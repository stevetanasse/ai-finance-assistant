from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from .states import AgentState

VALID_CATEGORIES = {"financial_concepts", "realtime_quotes", "out_of_scope"}

ROUTER_SYSTEM_PROMPT = (
    "You are a request router for a finance assistant. "
    "Classify the user's request into exactly one of these categories: "
    "financial_concepts, realtime_quotes, out_of_scope. "
    "Respond with ONLY one of these three exact strings and nothing else: "
    "financial_concepts, realtime_quotes, out_of_scope."
)


def router_node(state: AgentState) -> dict:
    call_counts = dict(state.get("call_counts", {}))
    call_counts["router_node"] = call_counts.get("router_node", 0) + 1

    user_text = state["messages"][-1].content

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", user_text),
    ])

    classification = response.content.strip()
    if classification not in VALID_CATEGORIES:
        classification = "out_of_scope"

    return {
        "call_counts": call_counts,
        "messages": [AIMessage(content=classification)],
    }


def financial_concepts_node(state: AgentState) -> dict:
    call_counts = dict(state.get("call_counts", {}))
    call_counts["financial_concepts_node"] = call_counts.get("financial_concepts_node", 0) + 1

    return {
        "call_counts": call_counts,
        "messages": [AIMessage(
            content="[financial_concepts_node] This node will answer financial concept questions. (stub)"
        )],
    }


def realtime_quotes_node(state: AgentState) -> dict:
    call_counts = dict(state.get("call_counts", {}))
    call_counts["realtime_quotes_node"] = call_counts.get("realtime_quotes_node", 0) + 1

    return {
        "call_counts": call_counts,
        "messages": [AIMessage(
            content="[realtime_quotes_node] This node will provide real-time stock quotes. (stub)"
        )],
    }
