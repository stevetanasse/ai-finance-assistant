from langchain_core.messages import AIMessage

from .states import AgentState, RouteDecision

ROUTER_SYSTEM_PROMPT = (
    "You are a request router for a finance assistant. "
    "Classify the user's request into exactly one of these categories: "
    "financial_concepts_node, realtime_quotes_node, out_of_scope. "
    "Use financial_concepts_node for questions about financial concepts, "
    "definitions, or general education (e.g. 'What is a P/E ratio?'). "
    "Use realtime_quotes_node for requests about current stock prices or quotes "
    "(e.g. 'What is the current price of AAPL?'). "
    "Use out_of_scope for anything unrelated to finance. "
    "Provide your classification in the 'next' field and a brief explanation "
    "of your reasoning in the 'reasoning' field."
)


def make_router_node(llm):
    def router_node(state: AgentState) -> dict:
        call_counts = dict(state.get("call_counts", {}))
        call_counts["router_node"] = call_counts.get("router_node", 0) + 1

        user_text = state["messages"][-1].content

        structured_llm = llm.with_structured_output(RouteDecision)
        decision = structured_llm.invoke([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", user_text),
        ])

        return {
            "call_counts": call_counts,
            "messages": [AIMessage(content=decision.next)],
        }

    return router_node


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
