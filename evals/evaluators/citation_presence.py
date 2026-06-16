"""Citation Presence evaluator.

Checks whether every URL listed in expected_source_ids appears (as a
normalized substring) somewhere in the assistant's response.  Normalization
strips trailing slashes and lowercases both the URL and the response text so
minor formatting differences (e.g. trailing punctuation after a URL) do not
cause false negatives.

Metric key: "citation_presence"
Score 1.0: all expected URLs found in the response.
Score 0.0: at least one expected URL is missing from the response.
Skipped (score 1.0): expected_behavior != "answer", or expected_source_ids
    is empty/absent.
"""

from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run


def _normalize(text: str) -> str:
    return text.lower().rstrip("/")


def citation_presence_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Evaluate whether the assistant response cites all expected source URLs.

    Args:
        run: LangSmith Run whose outputs["response"] holds the assistant reply.
        example: LangSmith Example whose outputs hold expected_behavior and
            expected_source_ids from the golden dataset.

    Returns:
        EvaluationResult with key="citation_presence", score 0.0 or 1.0,
        and a comment listing found/missing URLs and up to 100 characters of
        surrounding context for each found URL to aid debugging.
        Returns score=1.0 (skipped) when the metric does not apply.
    """
    response = run.outputs.get("response", "")
    expected_behavior = example.outputs.get("expected_behavior")
    expected_source_ids = example.outputs.get("expected_source_ids") or []

    if expected_behavior != "answer":
        return EvaluationResult(
            key="citation_presence",
            score=1.0,
            comment="Skipped: citation not expected for non-answer responses",
        )

    if not expected_source_ids:
        return EvaluationResult(
            key="citation_presence",
            score=1.0,
            comment="Skipped: no expected sources defined for this entry",
        )

    normalized_response = _normalize(response)
    found = []
    missing = []

    for url in expected_source_ids:
        needle = _normalize(url)
        idx = normalized_response.find(needle)
        if idx != -1:
            start = max(0, idx - 40)
            end = min(len(normalized_response), idx + len(needle) + 60)
            context = normalized_response[start:end]
            found.append(f"  FOUND {url!r} — context: ...{context!r}...")
        else:
            missing.append(f"  MISSING {url!r}")

    passed = len(missing) == 0
    parts = ["PASS" if passed else "FAIL"]
    if found:
        parts.append("Found:\n" + "\n".join(found))
    if missing:
        parts.append("Missing:\n" + "\n".join(missing))
    comment = " | ".join(parts)

    return EvaluationResult(
        key="citation_presence",
        score=1.0 if passed else 0.0,
        comment=comment,
    )
