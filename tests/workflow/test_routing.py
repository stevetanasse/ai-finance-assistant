from langchain_core.messages import HumanMessage

from src.workflow.graph import build_graph
from src.workflow.states import RouteDecision


class MockLLM:
    def __init__(self, route):
        self.route = route

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        return RouteDecision(next=self.route, reasoning="mock reasoning")


def test_routes_to_financial_concepts():
    graph = build_graph(MockLLM(route="financial_concepts_node"))
    config = {"configurable": {"thread_id": "test-financial-concepts"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is a P/E ratio?")], "call_counts": {}},
        config=config,
    )

    assert result["call_counts"]["financial_concepts_node"] == 1
    assert result["call_counts"]["router_node"] == 1
    assert len(result["call_counts"]) == 2


def test_routes_to_realtime_quotes():
    graph = build_graph(MockLLM(route="realtime_quotes_node"))
    config = {"configurable": {"thread_id": "test-realtime-quotes"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is the current price of AAPL?")], "call_counts": {}},
        config=config,
    )
   
    assert result["call_counts"]["realtime_quotes_node"] == 1
    assert result["call_counts"]["router_node"] == 1
    assert len(result["call_counts"]) == 2

def test_routes_out_of_scope():
    graph = build_graph(MockLLM(route="out_of_scope"))
    config = {"configurable": {"thread_id": "test-out-of-scope"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is the weather in London?")], "call_counts": {}},
        config=config,
    )

    assert result["call_counts"]["router_node"] == 1
    assert "financial_concepts_node" not in result["call_counts"]
    assert "realtime_quotes_node" not in result["call_counts"]
    assert len(result["call_counts"]) == 1

