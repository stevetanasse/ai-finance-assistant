import random
import pytest

from src.rag.embedder.qdrant_manager import QdrantManager

DIM = 384


def _random_vector(dim: int = DIM) -> list[float]:
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]


def _make_points(n: int = 5) -> list[dict]:
    return [
        {"id": f"chunk_{i:04d}", "vector": _random_vector(), "payload": {"text": f"text {i}", "chunk_index": i}}
        for i in range(n)
    ]


@pytest.fixture
def qm():
    return QdrantManager(in_memory=True)


@pytest.fixture
def qm_with_collection(qm):
    qm.create_collection("test_col", DIM)
    return qm


@pytest.fixture
def qm_with_data(qm_with_collection):
    points = _make_points(5)
    qm_with_collection.upsert_points("test_col", points)
    return qm_with_collection, points


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_in_memory_creates_client(qm):
    assert qm._client is not None

def test_collection_does_not_exist_for_new_client(qm):
    assert qm.collection_exists("nonexistent") is False


# ---------------------------------------------------------------------------
# create_collection / collection_exists
# ---------------------------------------------------------------------------

def test_create_collection_succeeds(qm):
    qm.create_collection("col1", DIM)
    assert qm.collection_exists("col1") is True

def test_create_collection_no_recreate_preserves_existing(qm_with_data):
    qm, original_points = qm_with_data
    qm.create_collection("test_col", DIM, recreate=False)
    info = qm.get_collection_info("test_col")
    assert info["vector_count"] == 5

def test_create_collection_recreate_true_resets_collection(qm_with_data):
    qm, _ = qm_with_data
    qm.create_collection("test_col", DIM, recreate=True)
    info = qm.get_collection_info("test_col")
    assert info["vector_count"] == 0

def test_collection_exists_true_after_creation(qm_with_collection):
    assert qm_with_collection.collection_exists("test_col") is True


# ---------------------------------------------------------------------------
# upsert_points / get_collection_info
# ---------------------------------------------------------------------------

def test_upsert_points_adds_points(qm_with_collection):
    qm_with_collection.upsert_points("test_col", _make_points(3))
    info = qm_with_collection.get_collection_info("test_col")
    assert info["vector_count"] == 3

def test_get_collection_info_returns_correct_vector_count(qm_with_data):
    qm, _ = qm_with_data
    info = qm.get_collection_info("test_col")
    assert info["vector_count"] == 5

def test_get_collection_info_returns_correct_vector_size(qm_with_data):
    qm, _ = qm_with_data
    info = qm.get_collection_info("test_col")
    assert info["vector_size"] == DIM


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

def test_query_returns_list_of_length_leq_top_k(qm_with_data):
    qm, _ = qm_with_data
    results = qm.query("test_col", _random_vector(), top_k=3)
    assert len(results) <= 3

def test_query_results_contain_required_keys(qm_with_data):
    qm, _ = qm_with_data
    results = qm.query("test_col", _random_vector(), top_k=1)
    assert len(results) >= 1
    for r in results:
        assert "id" in r
        assert "score" in r
        assert "payload" in r

def test_query_cosine_scores_between_neg1_and_1(qm_with_data):
    qm, _ = qm_with_data
    results = qm.query("test_col", _random_vector(), top_k=3)
    for r in results:
        assert -1.0 <= r["score"] <= 1.0

def test_query_top_k_3_returns_at_most_3(qm_with_data):
    qm, _ = qm_with_data
    results = qm.query("test_col", _random_vector(), top_k=3)
    assert len(results) <= 3


# ---------------------------------------------------------------------------
# _make_point_id
# ---------------------------------------------------------------------------

def test_make_point_id_returns_valid_uuid_string(qm):
    result = qm._make_point_id("test_chunk_0001")
    import uuid
    uuid.UUID(result)  # raises if invalid

def test_make_point_id_is_deterministic(qm):
    id1 = qm._make_point_id("same_input")
    id2 = qm._make_point_id("same_input")
    assert id1 == id2

def test_make_point_id_differs_for_different_inputs(qm):
    assert qm._make_point_id("chunk_0001") != qm._make_point_id("chunk_0002")


# ---------------------------------------------------------------------------
# list_collections / delete_collection
# ---------------------------------------------------------------------------

def test_list_collections_returns_collection_names(qm_with_collection):
    assert "test_col" in qm_with_collection.list_collections()

def test_delete_collection_removes_it(qm_with_collection):
    qm_with_collection.delete_collection("test_col")
    assert not qm_with_collection.collection_exists("test_col")

def test_delete_collection_nonexistent_does_not_raise(qm):
    qm.delete_collection("does_not_exist")  # must not raise
