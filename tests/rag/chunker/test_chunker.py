import pytest
from pathlib import Path

from src.rag.chunker.chunk_cache_manager import ChunkCacheManager
from src.rag.chunker.chunker import Chunker, STRATEGY_REGISTRY

URL = (
    "https://www.investor.gov/introduction-investing/investing-basics"
    "/investment-products/stocks"
)
URL2 = (
    "https://www.investor.gov/introduction-investing/investing-basics"
    "/investment-products/bonds-or-fixed-income-products/bonds"
)

SAMPLE_TEXT = (
    "Stocks represent ownership in a corporation. "
    "When you buy a stock you become a shareholder. "
    "Shareholders may receive dividends and have voting rights. "
    "The stock market allows buyers and sellers to trade shares. "
    "Stock prices fluctuate based on supply and demand. "
    "Investors buy stocks hoping the price will rise over time. "
    "Diversification helps manage risk across multiple investments. "
    "Blue-chip stocks are shares of well-established companies. "
    "Growth stocks are expected to grow faster than the market average. "
    "Value stocks trade at a price below their intrinsic value. "
) * 4  # ~400 words, enough to produce multiple chunks at size=200


@pytest.fixture
def cm(tmp_path):
    return ChunkCacheManager(base_path=tmp_path)


@pytest.fixture
def chunker(cm):
    return Chunker(chunk_cache_manager=cm, chunk_size=200, chunk_overlap=20)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_unknown_strategy_raises_value_error(cm):
    with pytest.raises(ValueError, match="Unknown strategy"):
        Chunker(chunk_cache_manager=cm, chunk_size=200, chunk_overlap=20, strategy="invalid")

def test_strategy_registry_contains_recursive():
    assert "recursive" in STRATEGY_REGISTRY


# ---------------------------------------------------------------------------
# chunk_url — caching behaviour
# ---------------------------------------------------------------------------

def test_chunk_url_returns_cached_entry_when_cached_no_force_refresh(chunker, cm):
    key = cm.make_cache_key(URL, 200, 20)
    cached_entry = {"status": "success", "total_chunks": 7, "url": URL}
    mapping = {key: cached_entry}
    result = chunker.chunk_url(URL, SAMPLE_TEXT, "investor.gov", "scraper/stocks.txt", mapping)
    assert result is cached_entry
    jsonl_files = list(cm.chunk_cache_dir.rglob("*.jsonl"))
    assert len(jsonl_files) == 0

def test_chunk_url_rechunks_when_force_refresh_true(chunker, cm):
    key = cm.make_cache_key(URL, 200, 20)
    mapping = {key: {"status": "success", "total_chunks": 1}}
    result = chunker.chunk_url(URL, SAMPLE_TEXT, "investor.gov", "scraper/stocks.txt",
                               mapping, force_refresh=True)
    assert result["status"] == "success"
    assert result["total_chunks"] > 1


# ---------------------------------------------------------------------------
# chunk_url — success path
# ---------------------------------------------------------------------------

def test_chunk_url_returns_success_status(chunker):
    result = chunker.chunk_url(URL, SAMPLE_TEXT, "investor.gov", "scraper/stocks.txt", {})
    assert result["status"] == "success"

def test_chunk_url_populates_all_mapping_fields(chunker):
    result = chunker.chunk_url(URL, SAMPLE_TEXT, "investor.gov", "scraper/stocks.txt", {})
    required = {"url", "file_path", "chunked_at", "status", "chunk_size",
                "chunk_overlap", "strategy", "total_chunks", "source_domain",
                "scraper_cache_path", "error_message"}
    assert required.issubset(result.keys())
    assert result["chunk_size"] == 200
    assert result["chunk_overlap"] == 20
    assert result["strategy"] == "recursive"
    assert result["total_chunks"] > 0
    assert result["error_message"] is None


# ---------------------------------------------------------------------------
# chunk_url — failure paths
# ---------------------------------------------------------------------------

def test_chunk_url_returns_failed_for_empty_text(chunker):
    result = chunker.chunk_url(URL, "", "investor.gov", "scraper/stocks.txt", {})
    assert result["status"] == "failed"
    assert result["error_message"] is not None

def test_chunk_url_returns_failed_for_whitespace_only_text(chunker):
    result = chunker.chunk_url(URL, "   \n\n   ", "investor.gov", "scraper/stocks.txt", {})
    assert result["status"] == "failed"

def test_chunk_url_never_raises_exceptions(chunker, cm, tmp_path):
    result = chunker.chunk_url(
        URL, SAMPLE_TEXT, "investor.gov",
        scraped_path="/nonexistent/scraper/stocks.txt",
        mapping={},
    )
    assert result is not None


# ---------------------------------------------------------------------------
# chunk_all
# ---------------------------------------------------------------------------

def _make_scraper_entry(url: str, txt_file: Path) -> dict:
    return {
        "status": "success",
        "file_path": str(txt_file),
        "source_domain": "investor.gov",
        "scraped_at": "2026-05-29T10:00:00+00:00",
        "word_count": 50,
        "error_message": None,
    }


def test_chunk_all_processes_pending_urls(chunker, cm, tmp_path):
    txt = tmp_path / "stocks.txt"
    txt.write_text(SAMPLE_TEXT, encoding="utf-8")
    scraper_mapping = {URL: _make_scraper_entry(URL, txt)}

    result = chunker.chunk_all(scraper_mapping)
    key = cm.make_cache_key(URL, 200, 20)
    assert key in result
    assert result[key]["status"] == "success"

def test_chunk_all_skips_non_success_scraper_entries(chunker, cm, tmp_path):
    scraper_mapping = {URL: {"status": "failed", "file_path": "missing.txt", "source_domain": "investor.gov"}}
    result = chunker.chunk_all(scraper_mapping)
    assert result == {}

def test_chunk_all_sets_failed_when_scraped_file_missing(chunker, cm, tmp_path):
    scraper_mapping = {
        URL: {
            "status": "success",
            "file_path": str(tmp_path / "nonexistent.txt"),
            "source_domain": "investor.gov",
        }
    }
    result = chunker.chunk_all(scraper_mapping)
    key = cm.make_cache_key(URL, 200, 20)
    assert result[key]["status"] == "failed"
    assert "not found" in result[key]["error_message"].lower()

def test_chunk_all_saves_mapping_after_each_url(chunker, cm, tmp_path):
    txt1 = tmp_path / "stocks.txt"
    txt2 = tmp_path / "bonds.txt"
    txt1.write_text(SAMPLE_TEXT, encoding="utf-8")
    txt2.write_text(SAMPLE_TEXT, encoding="utf-8")
    scraper_mapping = {
        URL: _make_scraper_entry(URL, txt1),
        URL2: _make_scraper_entry(URL2, txt2),
    }

    # Patch save_mapping to track how many times it's called
    save_count = [0]
    original_save = cm.save_mapping
    def counting_save(m):
        save_count[0] += 1
        original_save(m)
    cm.save_mapping = counting_save

    chunker.chunk_all(scraper_mapping)
    assert save_count[0] >= 2

def test_chunk_all_returns_complete_mapping(chunker, cm, tmp_path):
    txt1 = tmp_path / "stocks.txt"
    txt2 = tmp_path / "bonds.txt"
    txt1.write_text(SAMPLE_TEXT, encoding="utf-8")
    txt2.write_text(SAMPLE_TEXT, encoding="utf-8")
    scraper_mapping = {
        URL: _make_scraper_entry(URL, txt1),
        URL2: _make_scraper_entry(URL2, txt2),
    }
    result = chunker.chunk_all(scraper_mapping)
    key1 = cm.make_cache_key(URL, 200, 20)
    key2 = cm.make_cache_key(URL2, 200, 20)
    assert key1 in result
    assert key2 in result
