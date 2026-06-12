from unittest.mock import MagicMock

from src.rag.embedder.hybrid_search import hybrid_search, reciprocal_rank_fusion
from src.rag.embedder.qdrant_manager import QdrantManager

DIM = 384
SPARSE_VEC = {"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]}


def _make_mock_dense_embedder():
    e = MagicMock()
    e.model_name = "bge-small"
    e.embed_query.return_value = [0.1] * DIM
    return e


def _make_mock_sparse_embedder():
    e = MagicMock()
    e.model_name = "bm42"
    e.embed_sparse_query.return_value = SPARSE_VEC
    return e


def test_rrf_orders_items_appearing_in_both_lists_first():
    dense = [
        {"id": "a", "score": 0.9, "payload": {"text": "A"}},
        {"id": "b", "score": 0.8, "payload": {"text": "B"}},
        {"id": "c", "score": 0.7, "payload": {"text": "C"}},
    ]
    sparse = [
        {"id": "c", "score": 5.0, "payload": {"text": "C"}},
        {"id": "a", "score": 4.0, "payload": {"text": "A"}},
        {"id": "d", "score": 3.0, "payload": {"text": "D"}},
    ]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    fused_ids = [item["id"] for item in fused]
    assert fused_ids[0] in {"a", "c"}
    assert fused_ids[1] in {"a", "c"}
    assert set(fused_ids) == {"a", "b", "c", "d"}


def test_rrf_computes_expected_scores():
    dense = [{"id": "a", "score": 1.0, "payload": {}}]
    sparse = [{"id": "a", "score": 1.0, "payload": {}}]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    assert fused[0]["id"] == "a"
    assert fused[0]["rrf_score"] == 1 / 61 + 1 / 61


def test_rrf_dedupes_by_id_keeping_first_seen_payload():
    dense = [{"id": "a", "score": 0.9, "payload": {"text": "dense payload"}}]
    sparse = [{"id": "a", "score": 4.0, "payload": {"text": "sparse payload"}}]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    assert len(fused) == 1
    assert fused[0]["payload"]["text"] == "dense payload"


def test_hybrid_search_returns_fused_text_and_source_url():
    qm = MagicMock(spec=QdrantManager)
    qm.query_dense.return_value = [
        {"id": "a", "score": 0.9, "payload": {"text": "Bonds are debt securities.", "url": "https://investor.gov/bonds"}},
    ]
    qm.query_sparse.return_value = [
        {"id": "b", "score": 5.0, "payload": {"text": "Stocks represent ownership.", "url": "https://investor.gov/stocks"}},
    ]

    dense_embedder = _make_mock_dense_embedder()
    sparse_embedder = _make_mock_sparse_embedder()

    results = hybrid_search(qm, dense_embedder, sparse_embedder, "fin_test_collection", "What is a bond?", top_k=5)

    dense_embedder.embed_query.assert_called_once_with("What is a bond?")
    sparse_embedder.embed_sparse_query.assert_called_once_with("What is a bond?")
    qm.query_dense.assert_called_once_with("fin_test_collection", [0.1] * DIM, top_k=10)
    qm.query_sparse.assert_called_once_with("fin_test_collection", SPARSE_VEC, top_k=10)

    assert len(results) == 2
    assert {"text", "source_url", "score"} <= results[0].keys()
    texts = {r["text"] for r in results}
    urls = {r["source_url"] for r in results}
    assert "Bonds are debt securities." in texts
    assert "https://investor.gov/stocks" in urls


def test_hybrid_search_respects_top_k():
    qm = MagicMock(spec=QdrantManager)
    qm.query_dense.return_value = [
        {"id": str(i), "score": 1.0 - i * 0.1, "payload": {"text": f"chunk {i}", "url": f"https://investor.gov/{i}"}}
        for i in range(10)
    ]
    qm.query_sparse.return_value = []

    dense_embedder = _make_mock_dense_embedder()
    sparse_embedder = _make_mock_sparse_embedder()

    results = hybrid_search(qm, dense_embedder, sparse_embedder, "fin_test_collection", "query", top_k=3)

    assert len(results) == 3


def test_hybrid_search_returns_empty_list_when_no_results():
    qm = MagicMock(spec=QdrantManager)
    qm.query_dense.return_value = []
    qm.query_sparse.return_value = []

    dense_embedder = _make_mock_dense_embedder()
    sparse_embedder = _make_mock_sparse_embedder()

    results = hybrid_search(qm, dense_embedder, sparse_embedder, "fin_test_collection", "query")

    assert results == []
