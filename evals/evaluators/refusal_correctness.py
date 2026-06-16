"""Refusal Correctness evaluator.

Deterministic check: verifies that the assistant refuses when it should
(expected_behavior == "refuse") and answers when it should
(expected_behavior == "answer"), using the known canned refusal phrase
emitted by synchronizer_node.

Metric key: "refusal_correctness"
Score 1.0: the assistant's refusal/answer behavior matches expected_behavior.
Score 0.0: mismatch between expected and actual behavior.
"""

from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run

REFUSAL_PHRASE = "does not have the ability"


def refusal_correctness_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Evaluate whether the assistant refused or answered as expected.

    Args:
        run: LangSmith Run whose outputs["response"] holds the assistant reply.
        example: LangSmith Example whose outputs hold expected_behavior
            ("answer" or "refuse").

    Returns:
        EvaluationResult with key="refusal_correctness", score 0.0 or 1.0,
        and a comment describing expected behavior, detected behavior, and
        the pass/fail reason.
    """
    response = run.outputs.get("response", "")
    expected_behavior = example.outputs.get("expected_behavior", "answer")

    refused = REFUSAL_PHRASE in response
    detected_behavior = "refuse" if refused else "answer"

    if expected_behavior == "answer":
        passed = not refused
        reason = (
            "Response correctly does not contain a refusal."
            if passed
            else f"Response unexpectedly contains refusal phrase {REFUSAL_PHRASE!r}."
        )
    elif expected_behavior == "refuse":
        passed = refused
        reason = (
            f"Response correctly contains refusal phrase {REFUSAL_PHRASE!r}."
            if passed
            else f"Response does not contain expected refusal phrase {REFUSAL_PHRASE!r}."
        )
    else:
        passed = True
        reason = f"Unknown expected_behavior {expected_behavior!r}; skipped."

    comment = (
        f"expected={expected_behavior!r} | detected={detected_behavior!r} | "
        f"{'PASS' if passed else 'FAIL'}: {reason}"
    )

    return EvaluationResult(
        key="refusal_correctness",
        score=1.0 if passed else 0.0,
        comment=comment,
    )
