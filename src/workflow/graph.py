from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .nodes import financial_concepts_node, make_realtime_quotes_node, make_router_node
from .states import AgentState


def route_after_router(state: AgentState) -> str:
    classification = state["messages"][-1].content
    if classification == "financial_concepts_node":
        return "financial_concepts"
    if classification == "realtime_quotes_node":
        return "realtime_quotes"
    return END


def build_graph(llm):
    router_node = make_router_node(llm)
    realtime_quotes_node = make_realtime_quotes_node(llm)

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("router", router_node)
    graph_builder.add_node("financial_concepts", financial_concepts_node)
    graph_builder.add_node("realtime_quotes", realtime_quotes_node)
    graph_builder.set_entry_point("router")
    graph_builder.add_conditional_edges(
        "router",
        route_after_router,
        {"financial_concepts": "financial_concepts", "realtime_quotes": "realtime_quotes", END: END},
    )
    graph_builder.add_edge("financial_concepts", END)
    graph_builder.add_edge("realtime_quotes", END)

    return graph_builder.compile(checkpointer=MemorySaver())
