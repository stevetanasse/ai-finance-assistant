from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    call_counts: dict[str, int]


class RouteDecision(BaseModel):
    next: Literal["financial_concepts_node", "realtime_quotes_node", "out_of_scope"]
    reasoning: str
