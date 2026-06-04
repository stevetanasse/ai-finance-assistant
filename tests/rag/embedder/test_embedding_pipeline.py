import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.rag.embedder.embedding_cache_manager import EmbeddingCacheManager
from src.rag.embedder.embedding_pipeline import EmbeddingPipeline
from src.rag.embedder.qdrant_manager import QdrantManager

URL = "https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks"
CHUNK_KEY = f"{URL}|c200|o20"
DIM = 384

SPARSE_VEC = {"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]}


def _make_mock_dense_embedder(dim: int = DIM):
    e = MagicMock()
    e.model_name = "bge-small"
    e.vector_size = dim
    e.embed.return_value = [[0.1] * dim]
    e.embed_query.return_value = [0.1] * dim
    return e


def _make_mock_sparse_embedder():
    e = MagicMock()
    e.model_name = "bm42"
    e.sparse_model_name = "bm42"
    e.vector_size = 0
    e.embed_sparse.return_value = [SPARSE_VEC]
    e.embed_sparse_query.return_value = SPARSE_VEC
    return e


def _make_mock_qdrant():
    qm = MagicMock(spec=QdrantManager)
    qm.collection_exists.return_value = False
    qm.query_dense.return_value = [{"id": "abc", "score": 0.9, "payload": {"text": "result"}}]
    qm.query_sparse.return_value = [{"id": "abc", "score": 1.2, "payload": {"text": "result"}}]
    return qm


def _write_jsonl(path: Path, chunks: list[dict]) -> None:
    lines = [json.dumps(c, ensure_ascii=False) for c in chunks]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_chunks(n: int = 3) -> list[dict]:
    return [
        {
            "chunk_id": f"investor_gov_stocks_{i:04d}",
            "url": URL,
            "source_domain": "investor.gov",
            "chunk_index": i,
            "total_chunks": n,
            "text": f"Financial text chunk {i}.",
            "char_count": 25,
            "chunk_size": 200,
            "chunk_overlap": 20,
            "strategy": "recursive",
            "chunked_at": "2026-05-29T10:00:00+00:00",
        }
        for i in range(n)
    ]


@pytest.fixture
def cm(tmp_path):
    return EmbeddingCacheManager(base_path=tmp_path)


@pytest.fixture
def pipeline(cm):
    return EmbeddingPipeline(
        embedding_cache_manager=cm,
        qdrant_manager=_make_mock_qdrant(),
        dense_embedder=_make_mock_dense_embedder(),
        sparse_embedder=_make_mock_sparse_embedder(),
    )


def _make_chunk_entry(file_path: str) -> dict:
    return {
        "url": URL,
        "file_path": file_path,
        "source_domain": "investor.gov",
        "chunk_size": 200,
        "chunk_overlap": 20,
        "total_chunks": 3,
        "strategy": "recursive",
        "status": "success",
        "error_message": None,
    }


# ---------------------------------------------------------------------------
# embed_chunk_file — caching
# ---------------------------------------------------------------------------

def test_embed_chunk_file_skips_when_cached_no_force_refresh(pipeline, cm):
    emb_key = cm.make_cache_key(CHUNK_KEY, "bge-small", "bm42")
    cached_entry = {"status": "success", "total_vectors": 5}
    mapping = {emb_key: cached_entry}
    result = pipeline.embed_chunk_file(CHUNK_KEY, {}, mapping, force_refresh=False)
    assert result is cached_entry
    pipeline.qdrant.upsert_points.assert_not_called()

def test_embed_chunk_file_processes_when_force_refresh(pipeline, cm, tmp_path):
    chunks = _make_chunks(2)
    jfile = tmp_path / "chunks.jsonl"
    _write_jsonl(jfile, chunks)
    entry = _make_chunk_entry(str(jfile))
    emb_key = cm.make_cache_key(CHUNK_KEY, "bge-small", "bm42")
    mapping = {emb_key: {"status": "success"}}
    pipeline.dense_embedder.embed.return_value = [[0.1] * DIM, [0.2] * DIM]
    pipeline.sparse_embedder.embed_sparse.return_value = [SPARSE_VEC, SPARSE_VEC]
    result = pipeline.embed_chunk_file(CHUNK_KEY, entry, mapping, force_refresh=True)
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# embed_chunk_file — success
# ---------------------------------------------------------------------------

def test_embed_chunk_file_returns_success_status(pipeline, tmp_path):
    chunks = _make_chunks(2)
    jfile = tmp_path / "chunks.jsonl"
    _write_jsonl(jfile, chunks)
    entry = _make_chunk_entry(str(jfile))
    pipeline.dense_embedder.embed.return_value = [[0.1] * DIM, [0.2] * DIM]
    pipeline.sparse_embedder.embed_sparse.return_value = [SPARSE_VEC, SPARSE_VEC]
    result = pipeline.embed_chunk_file(CHUNK_KEY, entry, {})
    assert result["status"] == "success"

def test_embed_chunk_file_calls_dense_embed_with_chunk_texts(pipeline, tmp_path):
    chunks = _make_chunks(2)
    jfile = tmp_path / "chunks.jsonl"
    _write_jsonl(jfile, chunks)
    entry = _make_chunk_entry(str(jfile))
    pipeline.dense_embedder.embed.return_value = [[0.1] * DIM, [0.2] * DIM]
    pipeline.sparse_embedder.embed_sparse.return_value = [SPARSE_VEC, SPARSE_VEC]
    pipeline.embed_chunk_file(CHUNK_KEY, entry, {})
    texts = [c["text"] for c in chunks]
    pipeline.dense_embedder.embed.assert_called_once_with(texts)

def test_embed_chunk_file_calls_sparse_embed(pipeline, tmp_path):
    chunks = _make_chunks(2)
    jfile = tmp_path / "chunks.jsonl"
    _write_jsonl(jfile, chunks)
    entry = _make_chunk_entry(str(jfile))
    pipeline.dense_embedder.embed.return_value = [[0.1] * DIM, [0.2] * DIM]
    pipeline.sparse_embedder.embed_sparse.return_value = [SPARSE_VEC, SPARSE_VEC]
    pipeline.embed_chunk_file(CHUNK_KEY, entry, {})
    texts = [c["text"] for c in chunks]
    pipeline.sparse_embedder.embed_sparse.assert_called_once_with(texts)

def test_embed_chunk_file_calls_upsert_points(pipeline, tmp_path):
    chunks = _make_chunks(1)
    jfile = tmp_path / "chunks.jsonl"
    _write_jsonl(jfile, chunks)
    entry = _make_chunk_entry(str(jfile))
    pipeline.embed_chunk_file(CHUNK_KEY, entry, {})
    pipeline.qdrant.upsert_points.assert_called_once()

def _get_upserted_points(pipeline) -> list[dict]:
    args, kwargs = pipeline.qdrant.upsert_points.call_args
    return kwargs.get("points") or args[1]

def test_embed_chunk_file_payload_includes_text(pipeline, tmp_path):
    chunks = _make_chunks(1)
    jfile = tmp_path / "chunks.jsonl"
    _write_jsonl(jfile, chunks)
    entry = _make_chunk_entry(str(jfile))
    pipeline.embed_chunk_file(CHUNK_KEY, entry, {})
    points = _get_upserted_points(pipeline)
    assert "text" in points[0]["payload"]

def test_embed_chunk_file_payload_includes_dense_model(pipeline, tmp_path):
    chunks = _make_chunks(1)
    jfile = tmp_path / "chunks.jsonl"
    _write_jsonl(jfile, chunks)
    entry = _make_chunk_entry(str(jfile))
    pipeline.embed_chunk_file(CHUNK_KEY, entry, {})
    points = _get_upserted_points(pipeline)
    assert points[0]["payload"]["dense_model"] == "bge-small"

def test_embed_chunk_file_payload_includes_sparse_model(pipeline, tmp_path):
    chunks = _make_chunks(1)
    jfile = tmp_path / "chunks.jsonl"
    _write_jsonl(jfile, chunks)
    entry = _make_chunk_entry(str(jfile))
    pipeline.embed_chunk_file(CHUNK_KEY, entry, {})
    points = _get_upserted_points(pipeline)
    assert points[0]["payload"]["sparse_model"] == "bm42"


# ---------------------------------------------------------------------------
# embed_chunk_file — failure
# ---------------------------------------------------------------------------

def test_embed_chunk_file_returns_failed_on_exception(pipeline):
    result = pipeline.embed_chunk_file(CHUNK_KEY, _make_chunk_entry("/nonexistent.jsonl"), {})
    assert result["status"] == "failed"
    assert result["error_message"] is not None

def test_embed_chunk_file_never_raises(pipeline):
    result = pipeline.embed_chunk_file(CHUNK_KEY, _make_chunk_entry("/nonexistent.jsonl"), {})
    assert result is not None


# ---------------------------------------------------------------------------
# embed_all
# ---------------------------------------------------------------------------

def test_embed_all_processes_pending_keys(pipeline, cm, tmp_path):
    chunks = _make_chunks(2)
    jfile = tmp_path / "chunks.jsonl"
    _write_jsonl(jfile, chunks)
    chunk_mapping = {CHUNK_KEY: _make_chunk_entry(str(jfile))}
    pipeline.dense_embedder.embed.return_value = [[0.1] * DIM, [0.2] * DIM]
    pipeline.sparse_embedder.embed_sparse.return_value = [SPARSE_VEC, SPARSE_VEC]
    result = pipeline.embed_all(chunk_mapping)
    emb_key = cm.make_cache_key(CHUNK_KEY, "bge-small", "bm42")
    assert emb_key in result

def test_embed_all_saves_mapping_after_each_file(pipeline, cm, tmp_path):
    chunks = _make_chunks(1)
    jfile = tmp_path / "chunks.jsonl"
    _write_jsonl(jfile, chunks)
    chunk_mapping = {CHUNK_KEY: _make_chunk_entry(str(jfile))}
    save_count = [0]
    original = cm.save_mapping
    def counting_save(m): save_count[0] += 1; original(m)
    cm.save_mapping = counting_save
    pipeline.embed_all(chunk_mapping)
    assert save_count[0] >= 1

def test_embed_all_skips_non_success_chunk_entries(pipeline):
    chunk_mapping = {CHUNK_KEY: {"status": "failed", "file_path": "x.jsonl"}}
    result = pipeline.embed_all(chunk_mapping)
    assert result == {}


# ---------------------------------------------------------------------------
# query_dense
# ---------------------------------------------------------------------------

def test_query_dense_calls_embed_query(pipeline):
    pipeline.query_dense("test_col", "what are stocks?", top_k=3)
    pipeline.dense_embedder.embed_query.assert_called_once_with("what are stocks?")

def test_query_dense_calls_qdrant_query_dense(pipeline):
    pipeline.query_dense("test_col", "stocks?", top_k=5)
    pipeline.qdrant.query_dense.assert_called_once()

def test_query_dense_passes_top_k(pipeline):
    pipeline.query_dense("test_col", "query", top_k=7)
    args, kwargs = pipeline.qdrant.query_dense.call_args
    top_k_val = kwargs.get("top_k", args[2] if len(args) > 2 else None)
    assert top_k_val == 7


# ---------------------------------------------------------------------------
# query_sparse
# ---------------------------------------------------------------------------

def test_query_sparse_calls_embed_sparse_query(pipeline):
    pipeline.query_sparse("test_col", "what are stocks?", top_k=3)
    pipeline.sparse_embedder.embed_sparse_query.assert_called_once_with("what are stocks?")

def test_query_sparse_calls_qdrant_query_sparse(pipeline):
    pipeline.query_sparse("test_col", "stocks?", top_k=5)
    pipeline.qdrant.query_sparse.assert_called_once()

def test_query_sparse_passes_top_k(pipeline):
    pipeline.query_sparse("test_col", "query", top_k=7)
    args, kwargs = pipeline.qdrant.query_sparse.call_args
    top_k_val = kwargs.get("top_k", args[2] if len(args) > 2 else None)
    assert top_k_val == 7
