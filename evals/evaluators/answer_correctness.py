"""End-to-End Answer Correctness evaluator.

Uses an LLM-as-judge (gpt-4o-mini) to verify that the assistant's response
contains all expected facts and avoids all forbidden facts.

Metric key: "answer_correctness"
Score 1.0: all expected_facts present AND no forbidden_facts present.
Score 0.0: at least one expected fact missing OR at least one forbidden fact present.
Skipped (score 1.0): expected_behavior != "answer".
"""

import json

from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run

_SYSTEM_PROMPT = (
    "You are an evaluation assistant. You will be given a response from an AI finance "
    "assistant and a list of expected facts. Determine whether the response contains "
    "each expected fact (even if paraphrased) and whether it avoids any forbidden facts.\n"
    "Respond only in JSON with this structure:\n"
    '{\n'
    '  "expected_facts_present": [true/false, one per expected fact in order],\n'
    '  "forbidden_facts_present": [true/false, one per forbidden fact in order],\n'
    '  "reasoning": "brief explanation"\n'
    "}\n"
    "Do not include any text outside the JSON object."
)

_judge_llm = ChatOpenAI(model="gpt-4o-mini")


def answer_correctness_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Evaluate whether the assistant response contains all expected facts.

    Args:
        run: LangSmith Run whose outputs["response"] holds the assistant reply.
        example: LangSmith Example whose outputs hold expected_behavior,
            expected_facts, and forbidden_facts from the golden dataset.

    Returns:
        EvaluationResult with key="answer_correctness", score 0.0 or 1.0,
        and a comment detailing which facts passed or failed.
        Returns score=1.0 (skipped) when expected_behavior != "answer".
        Returns score=0.0 with an error comment on any exception.
    """
    try:
        response = run.outputs.get("response", "")
        expected_behavior = example.outputs.get("expected_behavior")

        if expected_behavior != "answer":
            return EvaluationResult(
                key="answer_correctness",
                score=1.0,
                comment="Skipped: expected_behavior is not 'answer'",
            )

        expected_facts = example.outputs.get("expected_facts", [])
        forbidden_facts = example.outputs.get("forbidden_facts", [])

        user_message = (
            f"Response:\n{response}\n\n"
            f"Expected facts: {json.dumps(expected_facts, ensure_ascii=False)}\n"
            f"Forbidden facts: {json.dumps(forbidden_facts, ensure_ascii=False)}"
        )
        judge_response = _judge_llm.invoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )

        judgment = json.loads(judge_response.content)
        expected_present = judgment.get("expected_facts_present", [])
        forbidden_present = judgment.get("forbidden_facts_present", [])
        reasoning = judgment.get("reasoning", "")

        all_expected_ok = all(expected_present) if expected_present else True
        no_forbidden_found = not any(forbidden_present) if forbidden_present else True
        passed = all_expected_ok and no_forbidden_found

        fact_details = []
        for fact, present in zip(expected_facts, expected_present):
            status = "FOUND" if present else "MISSING"
            fact_details.append(f"  [{status}] expected: {fact!r}")
        for fact, present in zip(forbidden_facts, forbidden_present):
            status = "FOUND (bad)" if present else "absent (ok)"
            fact_details.append(f"  [{status}] forbidden: {fact!r}")
        comment = (
            f"{'PASS' if passed else 'FAIL'} | {reasoning}\n"
            + "\n".join(fact_details)
        )

        return EvaluationResult(
            key="answer_correctness",
            score=1.0 if passed else 0.0,
            comment=comment,
        )

    except Exception as exc:
        return EvaluationResult(
            key="answer_correctness",
            score=0.0,
            comment=f"Evaluator error: {exc}",
        )
