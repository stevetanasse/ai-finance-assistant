import json
import pytest
from pathlib import Path

from src.rag.chunker.chunk_cache_manager import ChunkCacheManager

URL = (
    "https://www.investor.gov/introduction-investing/investing-basics"
    "/investment-products/stocks"
)
URL2 = (
    "https://www.investor.gov/introduction-investing/investing-basics"
    "/investment-products/bonds-or-fixed-income-products/bonds"
)

SAMPLE_CHUNKS = ["First chunk of text.", "Second chunk of text.", "Third chunk of text."]


@pytest.fixture
def cm(tmp_path):
    return ChunkCacheManager(base_path=tmp_path)


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------

def test_make_cache_key_returns_composite_format(cm):
    key = cm.make_cache_key(URL, 500, 50)
    assert key == f"{URL}|c500|o50"

def test_make_cache_key_different_sizes_produce_different_keys(cm):
    key1 = cm.make_cache_key(URL, 500, 50)
    key2 = cm.make_cache_key(URL, 1000, 100)
    assert key1 != key2


# ---------------------------------------------------------------------------
# get_chunk_filepath
# ---------------------------------------------------------------------------

def test_get_chunk_filepath_uses_domain_subdir(cm):
    path = cm.get_chunk_filepath(URL, 500, 50)
    assert path.parent.name == "investor.gov"

def test_get_chunk_filepath_filename_contains_chunk_params(cm):
    path = cm.get_chunk_filepath(URL, 500, 50)
    assert "_c500_o50" in path.name

def test_get_chunk_filepath_ends_with_jsonl(cm):
    path = cm.get_chunk_filepath(URL, 500, 50)
    assert path.suffix == ".jsonl"

def test_same_url_different_params_produce_different_paths(cm):
    path1 = cm.get_chunk_filepath(URL, 500, 50)
    path2 = cm.get_chunk_filepath(URL, 1000, 100)
    assert path1 != path2

def test_same_url_different_params_produce_different_keys(cm):
    key1 = cm.make_cache_key(URL, 500, 50)
    key2 = cm.make_cache_key(URL, 1000, 100)
    assert key1 != key2


# ---------------------------------------------------------------------------
# is_cached
# ---------------------------------------------------------------------------

def test_is_cached_returns_false_for_empty_mapping(cm):
    assert cm.is_cached(URL, 500, 50, {}) is False

def test_is_cached_returns_false_when_status_is_not_success(cm):
    key = cm.make_cache_key(URL, 500, 50)
    mapping = {key: {"status": "failed"}}
    assert cm.is_cached(URL, 500, 50, mapping) is False

def test_is_cached_returns_true_when_status_is_success(cm):
    key = cm.make_cache_key(URL, 500, 50)
    mapping = {key: {"status": "success"}}
    assert cm.is_cached(URL, 500, 50, mapping) is True


# ---------------------------------------------------------------------------
# write_chunks / read_chunks
# ---------------------------------------------------------------------------

def test_write_chunks_creates_jsonl_file(cm):
    path = cm.write_chunks(SAMPLE_CHUNKS, URL, 500, 50, "investor.gov", "scraper/stocks.txt", "recursive")
    assert path.exists()

def test_write_chunks_writes_one_json_object_per_line(cm):
    path = cm.write_chunks(SAMPLE_CHUNKS, URL, 500, 50, "investor.gov", "scraper/stocks.txt", "recursive")
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == len(SAMPLE_CHUNKS)
    for line in lines:
        obj = json.loads(line)
        assert isinstance(obj, dict)

def test_write_chunks_each_line_has_required_fields(cm):
    required = {"chunk_id", "url", "source_domain", "chunk_index", "total_chunks",
                "text", "char_count", "chunk_size", "chunk_overlap", "strategy", "chunked_at"}
    path = cm.write_chunks(SAMPLE_CHUNKS, URL, 500, 50, "investor.gov", "scraper/stocks.txt", "recursive")
    for line in [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]:
        obj = json.loads(line)
        assert required.issubset(obj.keys()), f"Missing fields: {required - obj.keys()}"

def test_write_chunks_chunk_index_is_sequential(cm):
    path = cm.write_chunks(SAMPLE_CHUNKS, URL, 500, 50, "investor.gov", "scraper/stocks.txt", "recursive")
    objs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    indices = [o["chunk_index"] for o in objs]
    assert indices == list(range(len(SAMPLE_CHUNKS)))

def test_write_chunks_total_chunks_is_consistent(cm):
    path = cm.write_chunks(SAMPLE_CHUNKS, URL, 500, 50, "investor.gov", "scraper/stocks.txt", "recursive")
    objs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for o in objs:
        assert o["total_chunks"] == len(SAMPLE_CHUNKS)

def test_read_chunks_returns_list_of_dicts(cm):
    path = cm.write_chunks(SAMPLE_CHUNKS, URL, 500, 50, "investor.gov", "scraper/stocks.txt", "recursive")
    result = cm.read_chunks(path)
    assert isinstance(result, list)
    assert all(isinstance(c, dict) for c in result)

def test_read_chunks_round_trip(cm):
    path = cm.write_chunks(SAMPLE_CHUNKS, URL, 500, 50, "investor.gov", "scraper/stocks.txt", "recursive")
    result = cm.read_chunks(path)
    assert len(result) == len(SAMPLE_CHUNKS)
    texts = [c["text"] for c in result]
    assert texts == SAMPLE_CHUNKS

def test_read_chunks_skips_blank_lines(cm, tmp_path):
    jsonl_path = tmp_path / "test.jsonl"
    lines = ['{"text": "a"}', "", '{"text": "b"}', "  ", '{"text": "c"}']
    jsonl_path.write_text("\n".join(lines), encoding="utf-8")
    result = cm.read_chunks(jsonl_path)
    assert len(result) == 3

def test_read_chunks_raises_value_error_on_bad_json(cm, tmp_path):
    jsonl_path = tmp_path / "bad.jsonl"
    jsonl_path.write_text('{"text": "ok"}\nnot valid json\n{"text": "ok2"}', encoding="utf-8")
    with pytest.raises(ValueError):
        cm.read_chunks(jsonl_path)


# ---------------------------------------------------------------------------
# load_mapping / save_mapping
# ---------------------------------------------------------------------------

def test_load_mapping_returns_empty_dict_when_no_file(cm):
    assert cm.load_mapping() == {}

def test_save_and_load_mapping_round_trip(cm):
    key = cm.make_cache_key(URL, 500, 50)
    mapping = {key: {"status": "success", "total_chunks": 5}}
    cm.save_mapping(mapping)
    assert cm.load_mapping() == mapping


# ---------------------------------------------------------------------------
# get_pending_urls
# ---------------------------------------------------------------------------

def test_get_pending_urls_returns_url_in_scraper_but_not_chunk(cm):
    scraper_mapping = {URL: {"status": "success", "file_path": "scraper/stocks.txt", "source_domain": "investor.gov"}}
    result = cm.get_pending_urls({}, scraper_mapping, 500, 50)
    assert len(result) == 1
    assert result[0]["url"] == URL

def test_get_pending_urls_excludes_url_already_cached(cm):
    key = cm.make_cache_key(URL, 500, 50)
    chunk_mapping = {key: {"status": "success"}}
    scraper_mapping = {URL: {"status": "success", "file_path": "scraper/stocks.txt", "source_domain": "investor.gov"}}
    result = cm.get_pending_urls(chunk_mapping, scraper_mapping, 500, 50)
    assert result == []

def test_get_pending_urls_includes_url_with_different_chunk_params(cm):
    key_500 = cm.make_cache_key(URL, 500, 50)
    chunk_mapping = {key_500: {"status": "success"}}
    scraper_mapping = {URL: {"status": "success", "file_path": "scraper/stocks.txt", "source_domain": "investor.gov"}}
    result = cm.get_pending_urls(chunk_mapping, scraper_mapping, 1000, 100)
    assert len(result) == 1
    assert result[0]["url"] == URL

def test_get_pending_urls_excludes_non_success_scraper_entries(cm):
    scraper_mapping = {URL: {"status": "failed", "file_path": "scraper/stocks.txt", "source_domain": "investor.gov"}}
    result = cm.get_pending_urls({}, scraper_mapping, 500, 50)
    assert result == []
