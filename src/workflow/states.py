from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator, model_validator


def merge_call_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Merge call_counts updates from parallel Send branches.

    Each branch starts from the same baseline dict (passed via Send) and only
    ever increments its own node's key, so colliding keys always carry equal
    values. max() combines the two dicts without per-key ownership tracking.
    """
    merged = dict(left)
    for key, value in right.items():
        merged[key] = max(merged.get(key, 0), value)
    return merged


class RouteDecision(BaseModel):
    """Routing decision produced by the router node.

    `next` may contain multiple routes when a request spans both financial
    concepts and real-time quotes, enabling fan-out via langgraph.types.Send.
    `out_of_scope` must never be combined with other routes.

    `financial_concepts_query` and `realtime_quotes_query` are per-node
    restatements of the user's request, decomposed so each node only sees the
    portion relevant to it (e.g. for "What is a dividend and what is TSLA's
    price?", `financial_concepts_query` is "What is a dividend?" and
    `realtime_quotes_query` is "TSLA"). Populated only when the corresponding
    route is present in `next`; both remain `None` when `next == ["out_of_scope"]`.
    """

    next: list[Literal["financial_concepts", "realtime_quotes", "out_of_scope"]] = Field(min_length=1)
    reasoning: str
    financial_concepts_query: str | None = None
    realtime_quotes_query: str | None = None

    @field_validator("next")
    @classmethod
    def validate_out_of_scope_singleton(cls, v: list[str]) -> list[str]:
        if "out_of_scope" in v and len(v) > 1:
            raise ValueError("'out_of_scope' must not appear alongside other routes")
        return v

    @model_validator(mode="after")
    def clear_sub_queries_for_out_of_scope(self) -> "RouteDecision":
        if self.next == ["out_of_scope"]:
            self.financial_concepts_query = None
            self.realtime_quotes_query = None
        return self


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    call_counts: Annotated[dict[str, int], merge_call_counts]
    route: list[str]
    route_decision: RouteDecision | None
