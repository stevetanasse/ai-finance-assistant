from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.rag.embedder.qdrant_manager import QdrantManager
from src.workflow.nodes import FINANCIAL_CONCEPTS_NOT_FOUND_MESSAGE, make_financial_concepts_node

DIM = 384
SPARSE_VEC = {"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]}


class MockConceptsNodeLLM:
    def __init__(self, response_content):
        self.response_content = response_content
        self.invoked_with = None

    def invoke(self, prompt):
        self.invoked_with = prompt
        return AIMessage(content=self.response_content)


def _make_mock_dense_embedder():
    e = MagicMock()
    e.embed_query.return_value = [0.1] * DIM
    return e


def _make_mock_sparse_embedder():
    e = MagicMock()
    e.embed_sparse_query.return_value = SPARSE_VEC
    return e


def test_successful_retrieval_grounds_llm_prompt_and_returns_response():
    qm = MagicMock(spec=QdrantManager)
    qm.query_dense.return_value = [
        {"id": "a", "score": 0.9, "payload": {"text": "A bond is a debt security.", "url": "https://investor.gov/bonds"}},
    ]
    qm.query_sparse.return_value = []

    llm = MockConceptsNodeLLM(response_content="A bond is a loan you make to an issuer.")

    node = make_financial_concepts_node(
        llm, qm, _make_mock_dense_embedder(), _make_mock_sparse_embedder(), "fin_test_collection"
    )

    result = node({
        "messages": [HumanMessage(content="What is a bond?")],
        "call_counts": {},
    })

    human_prompt = llm.invoked_with[1][1]
    assert "A bond is a debt security." in human_prompt
    assert "https://investor.gov/bonds" in human_prompt

    assert result["messages"][-1].content == "A bond is a loan you make to an issuer."
    assert result["call_counts"]["financial_concepts_node"] == 1


def test_empty_retrieval_returns_graceful_fallback_without_calling_llm():
    qm = MagicMock(spec=QdrantManager)
    qm.query_dense.return_value = []
    qm.query_sparse.return_value = []

    llm = MockConceptsNodeLLM(response_content="should not be used")

    node = make_financial_concepts_node(
        llm, qm, _make_mock_dense_embedder(), _make_mock_sparse_embedder(), "fin_test_collection"
    )

    result = node({
        "messages": [HumanMessage(content="What is a derivative?")],
        "call_counts": {},
    })

    assert result["messages"][-1].content == FINANCIAL_CONCEPTS_NOT_FOUND_MESSAGE
    assert llm.invoked_with is None
    assert result["call_counts"]["financial_concepts_node"] == 1
