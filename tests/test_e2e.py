import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.workflow.graph import build_graph

load_dotenv()

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def llm():
    """Real LLM instance, matching run_workflow.py's production configuration."""
    return ChatOpenAI(model="gpt-4o-mini")


@pytest.fixture
def graph(llm):
    return build_graph(llm)


def _invoke(graph, text, thread_id):
    return graph.invoke(
        {"messages": [HumanMessage(content=text)], "call_counts": {}, "route": []},
        config={"configurable": {"thread_id": thread_id}},
    )


def test_pure_concept_query(graph):
    result = _invoke(graph, "What is a stock?", "e2e-pure-concept")
    response = result["messages"][-1].content.lower()

    assert any(keyword in response for keyword in ["ownership", "share", "equity"])
    assert result["call_counts"].get("financial_concepts_node") == 1
    assert "realtime_quotes_node" not in result["call_counts"]


def test_pure_quote_query(graph):
    result = _invoke(graph, "What is the price of CEC?", "e2e-pure-quote")
    response = result["messages"][-1].content

    assert "CEC" in response.upper()
    assert any(char.isdigit() for char in response) or "not found" in response.lower()
    assert result["call_counts"].get("realtime_quotes_node") == 1
    assert "financial_concepts_node" not in result["call_counts"]


def test_dual_route_query(graph):
    result = _invoke(graph, "What is a dividend and what is the price of TSLA?", "e2e-dual-route")
    response = result["messages"][-1].content
    response_lower = response.lower()

    assert any(keyword in response_lower for keyword in ["payment", "dividend", "shareholder", "profit"])
    assert "TSLA" in response.upper()
    assert any(char.isdigit() for char in response)
    assert result["call_counts"].get("financial_concepts_node") == 1
    assert result["call_counts"].get("realtime_quotes_node") == 1


def test_out_of_scope_query(graph):
    result = _invoke(graph, "Who is the best kung-fu master?", "e2e-out-of-scope")
    response = result["messages"][-1].content.lower()

    assert "does not have the ability" in response
    assert "financial_concepts_node" not in result["call_counts"]
    assert "realtime_quotes_node" not in result["call_counts"]


def test_unknown_ticker_query(graph):
    result = _invoke(graph, "What is the price of A3CD?", "e2e-unknown-ticker")
    response = result["messages"][-1].content
    response_lower = response.lower()

    assert "A3CD" in response.upper()
    assert "not found" in response_lower or "unavailable" in response_lower


def test_concept_not_in_knowledge_base_query(graph):
    result = _invoke(graph, "What is a speculative derivative?", "e2e-concept-not-in-kb")
    response = result["messages"][-1].content.lower()

    assert any(
        phrase in response
        for phrase in ["does not contain", "cannot answer", "not found", "couldn't find"]
    )
    assert result["call_counts"].get("financial_concepts_node") == 1
    assert "realtime_quotes_node" not in result["call_counts"]
