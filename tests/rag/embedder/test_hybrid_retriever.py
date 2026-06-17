from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.rag.embedder.hybrid_retriever import HybridQdrantRetriever
from src.rag.embedder.qdrant_manager import QdrantManager

DIM = 384
SPARSE_VEC = {"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]}


def _make_mock_dense_embedder():
    e = MagicMock()
    e.embed_query.return_value = [0.1] * DIM
    return e


def _make_mock_sparse_embedder():
    e = MagicMock()
    e.embed_sparse_query.return_value = SPARSE_VEC
    return e


def test_invoke_returns_documents_with_text_and_metadata():
    qm = MagicMock(spec=QdrantManager)
    qm.query_dense.return_value = [
        {"id": "a", "score": 0.9, "payload": {"text": "Bonds are debt securities.", "url": "https://investor.gov/bonds"}},
    ]
    qm.query_sparse.return_value = []

    retriever = HybridQdrantRetriever(
        qdrant_manager=qm,
        dense_embedder=_make_mock_dense_embedder(),
        sparse_embedder=_make_mock_sparse_embedder(),
        collection_name="fin_test_collection",
    )

    docs = retriever.invoke("What is a bond?")

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "Bonds are debt securities."
    assert docs[0].metadata["source_url"] == "https://investor.gov/bonds"
    assert isinstance(docs[0].metadata["score"], float)


def test_invoke_returns_empty_list_when_no_results():
    qm = MagicMock(spec=QdrantManager)
    qm.query_dense.return_value = []
    qm.query_sparse.return_value = []

    retriever = HybridQdrantRetriever(
        qdrant_manager=qm,
        dense_embedder=_make_mock_dense_embedder(),
        sparse_embedder=_make_mock_sparse_embedder(),
        collection_name="fin_test_collection",
    )

    assert retriever.invoke("query") == []


def test_invoke_respects_top_k():
    qm = MagicMock(spec=QdrantManager)
    qm.query_dense.return_value = [
        {"id": str(i), "score": 1.0 - i * 0.1, "payload": {"text": f"chunk {i}", "url": f"https://investor.gov/{i}"}}
        for i in range(10)
    ]
    qm.query_sparse.return_value = []

    retriever = HybridQdrantRetriever(
        qdrant_manager=qm,
        dense_embedder=_make_mock_dense_embedder(),
        sparse_embedder=_make_mock_sparse_embedder(),
        collection_name="fin_test_collection",
        top_k=3,
    )

    assert len(retriever.invoke("query")) == 3
