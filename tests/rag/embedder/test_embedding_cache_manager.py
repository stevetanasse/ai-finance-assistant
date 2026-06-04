import pytest

from src.rag.embedder.embedding_cache_manager import EmbeddingCacheManager

URL = "https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks"
CHUNK_KEY = f"{URL}|c500|o50"
DENSE_MODEL = "bge-small"
SPARSE_MODEL = "bm42"


@pytest.fixture
def cm(tmp_path):
    return EmbeddingCacheManager(base_path=tmp_path)


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

def test_make_cache_key_returns_composite_format(cm):
    key = cm.make_cache_key(CHUNK_KEY, DENSE_MODEL, SPARSE_MODEL)
    assert key == f"{CHUNK_KEY}|{DENSE_MODEL}|{SPARSE_MODEL}"

def test_make_cache_key_includes_sparse_model(cm):
    key = cm.make_cache_key(CHUNK_KEY, DENSE_MODEL, SPARSE_MODEL)
    assert SPARSE_MODEL in key

def test_two_different_sparse_models_produce_different_keys(cm):
    key1 = cm.make_cache_key(CHUNK_KEY, DENSE_MODEL, "bm42")
    key2 = cm.make_cache_key(CHUNK_KEY, DENSE_MODEL, "other-sparse")
    assert key1 != key2


# ---------------------------------------------------------------------------
# make_collection_name
# ---------------------------------------------------------------------------

def test_make_collection_name_correct_format(cm):
    name = cm.make_collection_name("investor.gov", 500, 50, DENSE_MODEL, SPARSE_MODEL)
    assert name == "fin_investor_gov_c500_o50_bge-small_bm42"

def test_make_collection_name_includes_sparse_model(cm):
    name = cm.make_collection_name("investor.gov", 500, 50, DENSE_MODEL, SPARSE_MODEL)
    assert SPARSE_MODEL in name

def test_make_collection_name_slugifies_domain_dots(cm):
    name = cm.make_collection_name("investor.gov", 500, 50, DENSE_MODEL, SPARSE_MODEL)
    assert "." not in name

def test_make_collection_name_truncates_to_63_chars(cm):
    name = cm.make_collection_name("very.long.domain.name.example.com", 500, 50, DENSE_MODEL, SPARSE_MODEL)
    assert len(name) <= 63


# ---------------------------------------------------------------------------
# is_embedded
# ---------------------------------------------------------------------------

def test_is_embedded_returns_false_for_empty_mapping(cm):
    assert cm.is_embedded(CHUNK_KEY, DENSE_MODEL, {}, SPARSE_MODEL) is False

def test_is_embedded_returns_false_when_status_not_success(cm):
    key = cm.make_cache_key(CHUNK_KEY, DENSE_MODEL, SPARSE_MODEL)
    assert cm.is_embedded(CHUNK_KEY, DENSE_MODEL, {key: {"status": "failed"}}, SPARSE_MODEL) is False

def test_is_embedded_returns_true_when_status_success(cm):
    key = cm.make_cache_key(CHUNK_KEY, DENSE_MODEL, SPARSE_MODEL)
    assert cm.is_embedded(CHUNK_KEY, DENSE_MODEL, {key: {"status": "success"}}, SPARSE_MODEL) is True


# ---------------------------------------------------------------------------
# get_pending_chunk_keys
# ---------------------------------------------------------------------------

def test_get_pending_returns_key_in_chunk_but_not_embedding(cm):
    chunk_mapping = {CHUNK_KEY: {"status": "success"}}
    result = cm.get_pending_chunk_keys(chunk_mapping, {}, DENSE_MODEL, SPARSE_MODEL)
    assert CHUNK_KEY in result

def test_get_pending_excludes_already_embedded_key(cm):
    emb_key = cm.make_cache_key(CHUNK_KEY, DENSE_MODEL, SPARSE_MODEL)
    chunk_mapping = {CHUNK_KEY: {"status": "success"}}
    emb_mapping = {emb_key: {"status": "success"}}
    result = cm.get_pending_chunk_keys(chunk_mapping, emb_mapping, DENSE_MODEL, SPARSE_MODEL)
    assert result == []

def test_get_pending_includes_same_key_for_different_model(cm):
    emb_key = cm.make_cache_key(CHUNK_KEY, DENSE_MODEL, SPARSE_MODEL)
    chunk_mapping = {CHUNK_KEY: {"status": "success"}}
    emb_mapping = {emb_key: {"status": "success"}}
    result = cm.get_pending_chunk_keys(chunk_mapping, emb_mapping, DENSE_MODEL, "other-sparse")
    assert CHUNK_KEY in result

def test_get_pending_excludes_failed_chunk_keys(cm):
    chunk_mapping = {CHUNK_KEY: {"status": "failed"}}
    result = cm.get_pending_chunk_keys(chunk_mapping, {}, DENSE_MODEL, SPARSE_MODEL)
    assert result == []


# ---------------------------------------------------------------------------
# load_mapping / save_mapping
# ---------------------------------------------------------------------------

def test_load_mapping_returns_empty_dict_when_no_file(cm):
    assert cm.load_mapping() == {}

def test_save_and_load_mapping_round_trip(cm):
    emb_key = cm.make_cache_key(CHUNK_KEY, DENSE_MODEL, SPARSE_MODEL)
    mapping = {emb_key: {"status": "success", "total_vectors": 12}}
    cm.save_mapping(mapping)
    assert cm.load_mapping() == mapping
