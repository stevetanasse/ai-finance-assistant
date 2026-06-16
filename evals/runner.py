"""Evaluation runner for the AI Finance Assistant.

Orchestrates the full evaluation lifecycle:
  1. Validates required environment variables.
  2. Loads and validates the golden dataset from evaluation_dataset.json.
  3. Creates or reuses a named LangSmith Dataset and uploads examples.
  4. Defines a target function that invokes the LangGraph application.
  5. Runs a LangSmith Experiment with three custom evaluators.
  6. Prints a human-readable summary of results.

Usage:
    uv run python evals/runner.py [--experiment-prefix NAME]
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import json
import os
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import evaluate

from evals.evaluators.answer_correctness import answer_correctness_evaluator
from evals.evaluators.citation_presence import citation_presence_evaluator
from evals.evaluators.refusal_correctness import refusal_correctness_evaluator
from src.workflow.graph import build_graph

_REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_TRACING",
]

_REQUIRED_ENTRY_FIELDS = {"id", "question", "expected_behavior", "expected_facts"}


def _validate_env() -> None:
    """Raise EnvironmentError if any required environment variable is missing."""
    missing = [var for var in _REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        raise EnvironmentError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + "\nSee .env.example for the full list of required variables."
        )


def _load_dataset(dataset_path: str) -> list[dict]:
    """Load and validate the golden dataset from a JSON file.

    Args:
        dataset_path: Path to evaluation_dataset.json.

    Returns:
        List of validated dataset entry dicts.

    Raises:
        ValueError: If any entry is missing a required field.
    """
    with open(dataset_path, encoding="utf-8") as f:
        entries = json.load(f)

    for entry in entries:
        missing_fields = _REQUIRED_ENTRY_FIELDS - set(entry.keys())
        if missing_fields:
            raise ValueError(
                f"Entry '{entry.get('id', '<unknown>')}' is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )

    return entries


def _upload_dataset(
    client: Client,
    entries: list[dict],
    dataset_name: str,
) -> None:
    """Create or reuse a LangSmith Dataset and upload examples idempotently.

    Each example is assigned a stable UUID derived from its string ``id`` field
    using uuid5, so re-running the runner never creates duplicate examples.

    Args:
        client: Authenticated LangSmith Client.
        entries: Golden dataset entries loaded from evaluation_dataset.json.
        dataset_name: Name of the LangSmith Dataset to create or reuse.
    """
    datasets = list(client.list_datasets(dataset_name=dataset_name))
    if datasets:
        print(f"Using existing LangSmith dataset: {dataset_name!r}")
    else:
        client.create_dataset(dataset_name)
        print(f"Created new LangSmith dataset: {dataset_name!r}")

    existing_ids = {
        str(ex.id)
        for ex in client.list_examples(dataset_name=dataset_name)
    }

    created = 0
    skipped = 0
    for entry in entries:
        example_id = uuid.uuid5(uuid.NAMESPACE_URL, entry["id"])
        if str(example_id) in existing_ids:
            skipped += 1
            continue

        inputs = {"question": entry["question"]}
        outputs = {k: v for k, v in entry.items() if k != "question"}
        client.create_example(
            inputs=inputs,
            outputs=outputs,
            dataset_name=dataset_name,
            example_id=example_id,
        )
        created += 1

    print(f"Dataset sync: {created} created, {skipped} already exist.")


def _print_summary(results) -> None:
    """Print a human-readable experiment summary to stdout.

    Args:
        results: EvaluationResults returned by langsmith.evaluate().
    """
    rows = list(results)
    total = len(rows)

    metric_scores: dict[str, list[float]] = {}
    failures: list[str] = []

    for row in rows:
        example_id = str(row.get("example_id", "?"))
        feedback_list = row.get("feedback", []) or []
        for fb in feedback_list:
            key = fb.key if hasattr(fb, "key") else fb.get("key", "unknown")
            score = fb.score if hasattr(fb, "score") else fb.get("score", None)
            if score is not None:
                metric_scores.setdefault(key, []).append(float(score))
            if score == 0.0:
                comment = fb.comment if hasattr(fb, "comment") else fb.get("comment", "")
                failures.append(f"  example={example_id!r} metric={key!r}: {comment}")

    print("\n" + "=" * 60)
    print(f"Experiment complete | {total} example(s) evaluated")
    print("=" * 60)

    if metric_scores:
        print("\nPer-metric averages:")
        for metric, scores in sorted(metric_scores.items()):
            avg = sum(scores) / len(scores)
            print(f"  {metric}: {avg:.2f} ({len(scores)} result(s))")
    else:
        print("\n(No metric scores found in results — check LangSmith UI.)")

    if failures:
        print(f"\nExamples with score=0.0 ({len(failures)} failure(s)):")
        for f in failures:
            print(f)
    else:
        print("\nAll examples passed all metrics.")

    print("=" * 60)


def run_evaluation(
    dataset_path: str = "evals/evaluation_dataset.json",
    experiment_prefix: str = "finance-assistant",
    langsmith_dataset_name: str = "ai-finance-assistant-golden",
) -> None:
    """Run the full golden dataset evaluation against the LangGraph application.

    Args:
        dataset_path: Path to the golden dataset JSON file.
        experiment_prefix: Prefix for the LangSmith Experiment name.
        langsmith_dataset_name: Name of the LangSmith Dataset to create/reuse.

    Raises:
        EnvironmentError: If required environment variables are not set.
        ValueError: If any dataset entry is missing required fields.
    """
    _validate_env()

    entries = _load_dataset(dataset_path)
    print(f"Loaded {len(entries)} example(s) from {dataset_path!r}")

    client = Client()
    _upload_dataset(client, entries, langsmith_dataset_name)

    llm = ChatOpenAI(model="gpt-4o-mini")
    compiled_graph = build_graph(llm)

    def target(inputs: dict) -> dict:
        result = compiled_graph.invoke(
            {
                "messages": [HumanMessage(content=inputs["question"])],
                "call_counts": {},
            },
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
        return {"response": result["messages"][-1].content}

    print(f"\nStarting experiment with prefix {experiment_prefix!r} …")
    results = evaluate(
        target,
        data=langsmith_dataset_name,
        evaluators=[
            answer_correctness_evaluator,
            refusal_correctness_evaluator,
            citation_presence_evaluator,
        ],
        experiment_prefix=experiment_prefix,
        metadata={"model": "gpt-4o-mini", "graph_version": "fan-out-fan-in"},
    )

    _print_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run golden dataset evaluations against the AI Finance Assistant."
    )
    parser.add_argument(
        "--experiment-prefix",
        default="finance-assistant",
        help="Prefix for the LangSmith Experiment name (default: finance-assistant)",
    )
    args = parser.parse_args()
    run_evaluation(experiment_prefix=args.experiment_prefix)
