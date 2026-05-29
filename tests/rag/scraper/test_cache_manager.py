import pytest
from pathlib import Path

from src.rag.scraper.cache_manager import CacheManager

URL_BONDS = (
    "https://www.investor.gov/introduction-investing/investing-basics"
    "/investment-products/bonds-or-fixed-income-products/bonds"
)
URL_STOCKS = (
    "https://www.investor.gov/introduction-investing/investing-basics"
    "/investment-products/stocks"
)


@pytest.fixture
def cm(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return CacheManager()


# ---------------------------------------------------------------------------
# get_cache_filepath
# ---------------------------------------------------------------------------

def test_get_cache_filepath_uses_domain_subdir():
    path = CacheManager().get_cache_filepath(URL_BONDS, "html_cache")
    assert path.parent.name == "investor.gov"

def test_get_cache_filepath_slugifies_last_two_segments():
    path = CacheManager().get_cache_filepath(URL_BONDS, "html_cache")
    assert path.name == "bonds-or-fixed-income-products_bonds.html"

def test_get_cache_filepath_single_segment_path():
    url = "https://www.example.com/stocks"
    path = CacheManager().get_cache_filepath(url, "html_cache")
    assert path.name == "stocks.html"

def test_get_cache_filepath_html_extension():
    path = CacheManager().get_cache_filepath(URL_BONDS, "html_cache")
    assert path.suffix == ".html"

def test_get_cache_filepath_txt_extension_for_scraper_cache():
    path = CacheManager().get_cache_filepath(URL_BONDS, "scraper_cache")
    assert path.suffix == ".txt"

def test_get_cache_filepath_strips_query_params():
    url = URL_BONDS + "?ref=home&utm_source=test"
    path = CacheManager().get_cache_filepath(url, "html_cache")
    assert "ref" not in path.name
    assert "utm" not in path.name


# ---------------------------------------------------------------------------
# is_url_cached
# ---------------------------------------------------------------------------

def test_is_url_cached_true_for_success_entry():
    assert CacheManager().is_url_cached(URL_BONDS, {URL_BONDS: {"status": "success"}}) is True

def test_is_url_cached_false_for_missing_url():
    assert CacheManager().is_url_cached(URL_BONDS, {}) is False

def test_is_url_cached_false_for_failed_entry():
    assert CacheManager().is_url_cached(URL_BONDS, {URL_BONDS: {"status": "failed"}}) is False


# ---------------------------------------------------------------------------
# get_failed_urls
# ---------------------------------------------------------------------------

def test_get_failed_urls_returns_only_failed():
    mapping = {
        "url1": {"status": "failed"},
        "url2": {"status": "success"},
        "url3": {"status": "failed"},
    }
    assert set(CacheManager().get_failed_urls(mapping)) == {"url1", "url3"}

def test_get_failed_urls_empty_mapping():
    assert CacheManager().get_failed_urls({}) == []


# ---------------------------------------------------------------------------
# get_pending_scrape_urls
# ---------------------------------------------------------------------------

def test_get_pending_scrape_urls_returns_pending_with_success_status():
    mapping = {
        "url1": {"status": "success", "scrape_status": "pending"},
        "url2": {"status": "success", "scrape_status": "success"},
        "url3": {"status": "failed", "scrape_status": "pending"},
    }
    assert CacheManager().get_pending_scrape_urls(mapping) == ["url1"]

def test_get_pending_scrape_urls_empty_mapping():
    assert CacheManager().get_pending_scrape_urls({}) == []


# ---------------------------------------------------------------------------
# load / save round-trips
# ---------------------------------------------------------------------------

def test_save_and_load_html_mapping_round_trip(cm):
    mapping = {URL_BONDS: {"status": "success", "file_path": "html_cache/test.html"}}
    cm.save_html_mapping(mapping)
    assert cm.load_html_mapping() == mapping

def test_save_html_mapping_creates_file_if_not_exists(cm):
    cm.save_html_mapping({"url": {"status": "success"}})
    assert cm.load_html_mapping() == {"url": {"status": "success"}}

def test_load_html_mapping_returns_empty_dict_when_no_file(cm):
    assert cm.load_html_mapping() == {}

def test_save_and_load_scraper_mapping_round_trip(cm):
    mapping = {URL_STOCKS: {"status": "success", "word_count": 42}}
    cm.save_scraper_mapping(mapping)
    assert cm.load_scraper_mapping() == mapping
