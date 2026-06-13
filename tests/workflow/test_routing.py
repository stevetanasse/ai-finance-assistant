from langchain_core.messages import HumanMessage

from src.workflow.graph import build_graph
from src.workflow.states import RouteDecision


class _MockLLMResponse(RouteDecision):
    """Doubles as the router's structured RouteDecision output (`.next`,
    `.reasoning`, sub-query fields - stored on AgentState.route_decision, so
    it must be a RouteDecision for checkpoint serialization) and as the
    realtime_quotes_node tool-call response (`.content`, `.tool_calls`)."""

    content: list[str] = []
    tool_calls: list = []

    def __init__(self, route, financial_concepts_query=None, realtime_quotes_query=None):
        super().__init__(
            next=route,
            reasoning="mock reasoning",
            content=route,
            tool_calls=[],
            financial_concepts_query=financial_concepts_query,
            realtime_quotes_query=realtime_quotes_query,
        )


class MockRouteNodeLLM:
    def __init__(self, route, financial_concepts_query=None, realtime_quotes_query=None):
        self.route = route
        self.financial_concepts_query = financial_concepts_query
        self.realtime_quotes_query = realtime_quotes_query

    def with_structured_output(self, schema):
        return self

    def bind_tools(self, tools):
        return self

    def invoke(self, prompt):
        return _MockLLMResponse(
            self.route,
            financial_concepts_query=self.financial_concepts_query,
            realtime_quotes_query=self.realtime_quotes_query,
        )


def test_routes_to_financial_concepts():
    graph = build_graph(MockRouteNodeLLM(
        route=["financial_concepts"],
        financial_concepts_query="What is a P/E ratio?",
    ))
    config = {"configurable": {"thread_id": "test-financial-concepts"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is a P/E ratio?")], "call_counts": {}, "route": []},
        config=config,
    )

    assert result["call_counts"]["financial_concepts_node"] == 1
    assert result["call_counts"]["router_node"] == 1
    assert result["call_counts"]["synchronizer_node"] == 1
    assert len(result["call_counts"]) == 3


def test_routes_to_realtime_quotes():
    graph = build_graph(MockRouteNodeLLM(
        route=["realtime_quotes"],
        realtime_quotes_query="AAPL",
    ))
    config = {"configurable": {"thread_id": "test-realtime-quotes"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is the current price of AAPL?")], "call_counts": {}, "route": []},
        config=config,
    )

    assert result["call_counts"]["realtime_quotes_node"] == 1
    assert result["call_counts"]["router_node"] == 1
    assert result["call_counts"]["synchronizer_node"] == 1
    assert len(result["call_counts"]) == 3

def test_routes_out_of_scope():
    graph = build_graph(MockRouteNodeLLM(route=["out_of_scope"]))
    config = {"configurable": {"thread_id": "test-out-of-scope"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is the weather in London?")], "call_counts": {}, "route": []},
        config=config,
    )

    assert result["call_counts"]["router_node"] == 1
    assert result["call_counts"]["synchronizer_node"] == 1
    assert "financial_concepts_node" not in result["call_counts"]
    assert "realtime_quotes_node" not in result["call_counts"]
    assert len(result["call_counts"]) == 2
