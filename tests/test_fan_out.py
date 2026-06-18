from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from src.workflow.graph import route_after_router
from src.workflow.nodes import (
    OUT_OF_SCOPE_MESSAGE,
    SYNCHRONIZER_OUT_OF_SCOPE_MESSAGE,
    make_out_of_scope_node,
    make_router_node,
    make_synchronizer_node,
)
from src.workflow.states import AgentState, RouteDecision


class MockRouteNodeLLM:
    def __init__(self, route, financial_concepts_query=None, realtime_quotes_query=None):
        self.route = route
        self.financial_concepts_query = financial_concepts_query
        self.realtime_quotes_query = realtime_quotes_query

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        return RouteDecision(
            next=self.route,
            reasoning="mock reasoning",
            financial_concepts_query=self.financial_concepts_query,
            realtime_quotes_query=self.realtime_quotes_query,
        )


# --- route_after_router as a pure function ---


def test_route_after_router_returns_sends_for_multiple_routes():
    state = {"messages": [], "call_counts": {}, "route": ["financial_concepts", "realtime_quotes"]}
    result = route_after_router(state)

    assert isinstance(result, list)
    assert all(isinstance(s, Send) for s in result)
    assert {s.node for s in result} == {"financial_concepts", "realtime_quotes"}


def test_route_after_router_single_route_financial_concepts():
    state = {"messages": [], "call_counts": {}, "route": ["financial_concepts"]}
    result = route_after_router(state)

    assert len(result) == 1
    assert result[0].node == "financial_concepts"


def test_route_after_router_single_route_realtime_quotes():
    state = {"messages": [], "call_counts": {}, "route": ["realtime_quotes"]}
    result = route_after_router(state)

    assert len(result) == 1
    assert result[0].node == "realtime_quotes"


def test_route_after_router_single_route_out_of_scope():
    state = {"messages": [], "call_counts": {}, "route": ["out_of_scope"]}
    result = route_after_router(state)

    assert len(result) == 1
    assert result[0].node == "out_of_scope"


def test_route_after_router_returns_sends_for_all_three_routes():
    state = {
        "messages": [],
        "call_counts": {},
        "route": ["out_of_scope", "financial_concepts", "realtime_quotes"],
    }
    result = route_after_router(state)

    assert isinstance(result, list)
    assert all(isinstance(s, Send) for s in result)
    assert {s.node for s in result} == {"out_of_scope", "financial_concepts", "realtime_quotes"}


# --- make_router_node multi-route output ---


def test_router_node_emits_multiple_routes():
    node = make_router_node(MockRouteNodeLLM(
        route=["financial_concepts", "realtime_quotes"],
        financial_concepts_query="What is a P/E ratio?",
        realtime_quotes_query="AAPL",
    ))
    result = node({
        "messages": [HumanMessage(content="What is a P/E ratio and what is AAPL's price?")],
        "call_counts": {},
    })

    assert result["route"] == ["financial_concepts", "realtime_quotes"]
    assert result["call_counts"]["router_node"] == 1
    assert result["route_decision"]["financial_concepts_query"] == "What is a P/E ratio?"
    assert result["route_decision"]["realtime_quotes_query"] == "AAPL"


# --- make_out_of_scope_node ---


def test_out_of_scope_node_returns_fixed_message():
    node = make_out_of_scope_node()
    result = node({"messages": [], "call_counts": {}})

    assert result["messages"][-1].content == OUT_OF_SCOPE_MESSAGE
    assert result["messages"][-1].name == "out_of_scope"
    assert result["call_counts"]["out_of_scope_node"] == 1


# --- make_synchronizer_node ---


def test_synchronizer_out_of_scope_canned_response():
    node = make_synchronizer_node()
    result = node({"messages": [], "call_counts": {}, "route": ["out_of_scope"]})

    assert result["messages"][-1].content == SYNCHRONIZER_OUT_OF_SCOPE_MESSAGE
    assert result["call_counts"]["synchronizer_node"] == 1


def test_synchronizer_fan_in_combines_both_sections():
    node = make_synchronizer_node()
    state = {
        "messages": [
            HumanMessage(content="What is a P/E ratio and what is AAPL's price?"),
            AIMessage(content="A P/E ratio measures...", name="financial_concepts"),
            AIMessage(content="AAPL: 189.45", name="realtime_quotes"),
        ],
        "call_counts": {"router_node": 1, "financial_concepts_node": 1, "realtime_quotes_node": 1},
        "route": ["financial_concepts", "realtime_quotes"],
    }
    result = node(state)

    content = result["messages"][-1].content
    assert "A P/E ratio measures..." in content
    assert "AAPL: 189.45" in content
    assert "**Financial Concept**" in content
    assert "**Market Data**" in content
    assert result["call_counts"]["synchronizer_node"] == 1


def test_synchronizer_fan_in_orders_out_of_scope_first():
    node = make_synchronizer_node()
    state = {
        "messages": [
            HumanMessage(content="What is the price of TSLA and is Godzilla a real animal?"),
            AIMessage(content=OUT_OF_SCOPE_MESSAGE, name="out_of_scope"),
            AIMessage(content="TSLA: 392.77", name="realtime_quotes"),
        ],
        "call_counts": {"router_node": 1, "out_of_scope_node": 1, "realtime_quotes_node": 1},
        "route": ["out_of_scope", "realtime_quotes"],
    }
    result = node(state)

    content = result["messages"][-1].content
    assert content.index("**Out of Scope**") < content.index("**Market Data**")
    assert "TSLA: 392.77" in content


def test_synchronizer_fan_in_orders_all_three_sections():
    node = make_synchronizer_node()
    state = {
        "messages": [
            HumanMessage(content="..."),
            AIMessage(content=OUT_OF_SCOPE_MESSAGE, name="out_of_scope"),
            AIMessage(content="A P/E ratio measures...", name="financial_concepts"),
            AIMessage(content="AAPL: 189.45", name="realtime_quotes"),
        ],
        "call_counts": {
            "router_node": 1,
            "out_of_scope_node": 1,
            "financial_concepts_node": 1,
            "realtime_quotes_node": 1,
        },
        "route": ["out_of_scope", "financial_concepts", "realtime_quotes"],
    }
    result = node(state)

    content = result["messages"][-1].content
    assert (
        content.index("**Out of Scope**")
        < content.index("**Financial Concept**")
        < content.index("**Market Data**")
    )


def test_synchronizer_pass_through_single_route():
    node = make_synchronizer_node()
    state = {
        "messages": [
            HumanMessage(content="What is AAPL's price?"),
            AIMessage(content="AAPL: 189.45", name="realtime_quotes"),
        ],
        "call_counts": {"router_node": 1, "realtime_quotes_node": 1},
        "route": ["realtime_quotes"],
    }
    result = node(state)

    assert "messages" not in result
    assert result["call_counts"]["synchronizer_node"] == 1


# --- full mini-graph dispatch tests ---


def _build_test_graph(route_value, financial_concepts_spy, realtime_quotes_spy):
    def fake_financial_concepts(state):
        financial_concepts_spy()
        call_counts = dict(state.get("call_counts", {}))
        call_counts["financial_concepts_node"] = call_counts.get("financial_concepts_node", 0) + 1
        return {
            "call_counts": call_counts,
            "messages": [AIMessage(content="concept answer", name="financial_concepts")],
        }

    def fake_realtime_quotes(state):
        realtime_quotes_spy()
        call_counts = dict(state.get("call_counts", {}))
        call_counts["realtime_quotes_node"] = call_counts.get("realtime_quotes_node", 0) + 1
        return {
            "call_counts": call_counts,
            "messages": [AIMessage(content="quote answer", name="realtime_quotes")],
        }

    builder = StateGraph(AgentState)
    builder.add_node("router", make_router_node(MockRouteNodeLLM(route=route_value)))
    builder.add_node("financial_concepts", fake_financial_concepts)
    builder.add_node("realtime_quotes", fake_realtime_quotes)
    builder.add_node("out_of_scope", make_out_of_scope_node())
    builder.add_node("synchronizer", make_synchronizer_node())
    builder.set_entry_point("router")
    builder.add_conditional_edges("router", route_after_router)
    builder.add_edge("financial_concepts", "synchronizer")
    builder.add_edge("realtime_quotes", "synchronizer")
    builder.add_edge("out_of_scope", "synchronizer")
    builder.add_edge("synchronizer", END)
    return builder.compile(checkpointer=MemorySaver())


def test_dual_route_invokes_both_nodes():
    fc_spy, rq_spy = MagicMock(), MagicMock()
    graph = _build_test_graph(["financial_concepts", "realtime_quotes"], fc_spy, rq_spy)

    result = graph.invoke(
        {"messages": [HumanMessage(content="P/E ratio and AAPL price?")], "call_counts": {}, "route": []},
        config={"configurable": {"thread_id": "dual-route"}},
    )

    fc_spy.assert_called_once()
    rq_spy.assert_called_once()
    assert "concept answer" in result["messages"][-1].content
    assert "quote answer" in result["messages"][-1].content
    assert result["call_counts"] == {
        "router_node": 1,
        "financial_concepts_node": 1,
        "realtime_quotes_node": 1,
        "synchronizer_node": 1,
    }


def test_single_route_financial_concepts_only():
    fc_spy, rq_spy = MagicMock(), MagicMock()
    graph = _build_test_graph(["financial_concepts"], fc_spy, rq_spy)

    graph.invoke(
        {"messages": [HumanMessage(content="What is a P/E ratio?")], "call_counts": {}, "route": []},
        config={"configurable": {"thread_id": "single-fc"}},
    )

    fc_spy.assert_called_once()
    rq_spy.assert_not_called()


def test_single_route_realtime_quotes_only():
    fc_spy, rq_spy = MagicMock(), MagicMock()
    graph = _build_test_graph(["realtime_quotes"], fc_spy, rq_spy)

    graph.invoke(
        {"messages": [HumanMessage(content="What is AAPL's price?")], "call_counts": {}, "route": []},
        config={"configurable": {"thread_id": "single-rq"}},
    )

    rq_spy.assert_called_once()
    fc_spy.assert_not_called()


def test_out_of_scope_neither_node_invoked():
    fc_spy, rq_spy = MagicMock(), MagicMock()
    graph = _build_test_graph(["out_of_scope"], fc_spy, rq_spy)

    result = graph.invoke(
        {"messages": [HumanMessage(content="What's the weather?")], "call_counts": {}, "route": []},
        config={"configurable": {"thread_id": "oos"}},
    )

    fc_spy.assert_not_called()
    rq_spy.assert_not_called()
    assert result["messages"][-1].content == SYNCHRONIZER_OUT_OF_SCOPE_MESSAGE


def test_compound_out_of_scope_and_realtime_quotes():
    fc_spy, rq_spy = MagicMock(), MagicMock()
    graph = _build_test_graph(["out_of_scope", "realtime_quotes"], fc_spy, rq_spy)

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="What is the price of TSLA and is Godzilla a real animal?")],
            "call_counts": {},
            "route": [],
        },
        config={"configurable": {"thread_id": "oos-rq"}},
    )

    fc_spy.assert_not_called()
    rq_spy.assert_called_once()
    content = result["messages"][-1].content
    assert content.index("**Out of Scope**") < content.index("**Market Data**")
    assert "quote answer" in content
    assert result["call_counts"] == {
        "router_node": 1,
        "out_of_scope_node": 1,
        "realtime_quotes_node": 1,
        "synchronizer_node": 1,
    }


def test_compound_out_of_scope_and_financial_concepts():
    fc_spy, rq_spy = MagicMock(), MagicMock()
    graph = _build_test_graph(["out_of_scope", "financial_concepts"], fc_spy, rq_spy)

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="What is a dividend and is Godzilla a real animal?")],
            "call_counts": {},
            "route": [],
        },
        config={"configurable": {"thread_id": "oos-fc"}},
    )

    fc_spy.assert_called_once()
    rq_spy.assert_not_called()
    content = result["messages"][-1].content
    assert content.index("**Out of Scope**") < content.index("**Financial Concept**")
    assert "concept answer" in content
    assert result["call_counts"] == {
        "router_node": 1,
        "out_of_scope_node": 1,
        "financial_concepts_node": 1,
        "synchronizer_node": 1,
    }


def test_compound_all_three_routes():
    fc_spy, rq_spy = MagicMock(), MagicMock()
    graph = _build_test_graph(
        ["out_of_scope", "financial_concepts", "realtime_quotes"], fc_spy, rq_spy
    )

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="What is a dividend, the price of TSLA, and is Godzilla real?")],
            "call_counts": {},
            "route": [],
        },
        config={"configurable": {"thread_id": "oos-fc-rq"}},
    )

    fc_spy.assert_called_once()
    rq_spy.assert_called_once()
    content = result["messages"][-1].content
    assert (
        content.index("**Out of Scope**")
        < content.index("**Financial Concept**")
        < content.index("**Market Data**")
    )
    assert result["call_counts"] == {
        "router_node": 1,
        "out_of_scope_node": 1,
        "financial_concepts_node": 1,
        "realtime_quotes_node": 1,
        "synchronizer_node": 1,
    }


def test_call_counts_integrity_dual_route():
    fc_spy, rq_spy = MagicMock(), MagicMock()
    graph = _build_test_graph(["financial_concepts", "realtime_quotes"], fc_spy, rq_spy)

    result = graph.invoke(
        {"messages": [HumanMessage(content="...")], "call_counts": {}, "route": []},
        config={"configurable": {"thread_id": "counts"}},
    )

    assert result["call_counts"] == {
        "router_node": 1,
        "financial_concepts_node": 1,
        "realtime_quotes_node": 1,
        "synchronizer_node": 1,
    }


# --- sub-query decomposition ---


def _build_subquery_test_graph(route_value, financial_concepts_query, realtime_quotes_query):
    captured = {}

    def fake_financial_concepts(state):
        captured["financial_concepts_query"] = state["route_decision"]["financial_concepts_query"]
        call_counts = dict(state.get("call_counts", {}))
        call_counts["financial_concepts_node"] = call_counts.get("financial_concepts_node", 0) + 1
        return {
            "call_counts": call_counts,
            "messages": [AIMessage(content="concept answer", name="financial_concepts")],
        }

    def fake_realtime_quotes(state):
        captured["realtime_quotes_query"] = state["route_decision"]["realtime_quotes_query"]
        call_counts = dict(state.get("call_counts", {}))
        call_counts["realtime_quotes_node"] = call_counts.get("realtime_quotes_node", 0) + 1
        return {
            "call_counts": call_counts,
            "messages": [AIMessage(content="quote answer", name="realtime_quotes")],
        }

    builder = StateGraph(AgentState)
    builder.add_node("router", make_router_node(MockRouteNodeLLM(
        route=route_value,
        financial_concepts_query=financial_concepts_query,
        realtime_quotes_query=realtime_quotes_query,
    )))
    builder.add_node("financial_concepts", fake_financial_concepts)
    builder.add_node("realtime_quotes", fake_realtime_quotes)
    builder.add_node("out_of_scope", make_out_of_scope_node())
    builder.add_node("synchronizer", make_synchronizer_node())
    builder.set_entry_point("router")
    builder.add_conditional_edges("router", route_after_router)
    builder.add_edge("financial_concepts", "synchronizer")
    builder.add_edge("realtime_quotes", "synchronizer")
    builder.add_edge("out_of_scope", "synchronizer")
    builder.add_edge("synchronizer", END)
    return builder.compile(checkpointer=MemorySaver()), captured


def test_dual_route_nodes_receive_decomposed_sub_queries():
    graph, captured = _build_subquery_test_graph(
        route_value=["financial_concepts", "realtime_quotes"],
        financial_concepts_query="What is a dividend?",
        realtime_quotes_query="TSLA",
    )

    graph.invoke(
        {"messages": [HumanMessage(content="What is a dividend and what is the price of TSLA?")], "call_counts": {}, "route": []},
        config={"configurable": {"thread_id": "subquery-decomposition"}},
    )

    assert captured["financial_concepts_query"] == "What is a dividend?"
    assert captured["realtime_quotes_query"] == "TSLA"
