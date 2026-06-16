import re
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

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


@pytest.fixture
def cm(tmp_path):
    return CacheManager(base_path=tmp_path)


# ---------------------------------------------------------------------------
# CacheManager initialisation
# ---------------------------------------------------------------------------

def test_default_base_path_ends_with_rag_caches():
    cm = CacheManager()
    assert cm.base_path.name == "rag_caches"

def test_custom_base_path_is_used_when_provided(tmp_path):
    cm = CacheManager(base_path=tmp_path)
    assert cm.base_path == tmp_path

def test_html_cache_subdir_created_on_init(tmp_path):
    cm = CacheManager(base_path=tmp_path)
    assert cm.html_cache_dir.is_dir()

def test_scraper_cache_subdir_created_on_init(tmp_path):
    cm = CacheManager(base_path=tmp_path)
    assert cm.scraper_cache_dir.is_dir()

def test_two_instances_do_not_share_state(tmp_path):
    cm1 = CacheManager(base_path=tmp_path / "run1")
    cm2 = CacheManager(base_path=tmp_path / "run2")
    cm1.save_html_mapping({"url_a": {"status": "success"}})
    assert cm2.load_html_mapping() == {}


# ---------------------------------------------------------------------------
# get_cache_filepath — GUID-based flat structure
# ---------------------------------------------------------------------------

def test_get_cache_filepath_file_is_directly_in_cache_dir(cm):
    path = cm.get_cache_filepath(URL_BONDS, cm.html_cache_dir)
    assert path.parent == cm.html_cache_dir

def test_get_cache_filepath_stem_is_guid_format(cm):
    path = cm.get_cache_filepath(URL_BONDS, cm.html_cache_dir)
    assert UUID_RE.match(path.stem), f"Expected UUID v4 stem, got: {path.stem}"

def test_get_cache_filepath_same_url_same_cache_dir_returns_same_path(cm):
    path1 = cm.get_cache_filepath(URL_BONDS, cm.html_cache_dir)
    path2 = cm.get_cache_filepath(URL_BONDS, cm.html_cache_dir)
    assert path1 == path2

def test_get_cache_filepath_different_urls_return_different_paths(cm):
    path1 = cm.get_cache_filepath(URL_BONDS, cm.html_cache_dir)
    path2 = cm.get_cache_filepath(URL_STOCKS, cm.html_cache_dir)
    assert path1 != path2

def test_get_cache_filepath_html_extension(cm):
    path = cm.get_cache_filepath(URL_BONDS, cm.html_cache_dir)
    assert path.suffix == ".html"

def test_get_cache_filepath_txt_extension_for_scraper_cache(cm):
    path = cm.get_cache_filepath(URL_BONDS, cm.scraper_cache_dir)
    assert path.suffix == ".txt"

def test_get_cache_filepath_strips_query_params(cm):
    url = URL_BONDS + "?ref=home&utm_source=test"
    path = cm.get_cache_filepath(url, cm.html_cache_dir)
    assert "ref" not in path.name
    assert "utm" not in path.name

def test_get_cache_filepath_same_guid_for_html_and_scraper(cm):
    html_path = cm.get_cache_filepath(URL_BONDS, cm.html_cache_dir)
    scraper_path = cm.get_cache_filepath(URL_BONDS, cm.scraper_cache_dir)
    assert html_path.stem == scraper_path.stem


# ---------------------------------------------------------------------------
# get_guid
# ---------------------------------------------------------------------------

def test_get_guid_returns_valid_uuid4(cm):
    guid = cm.get_guid(URL_BONDS)
    assert UUID_RE.match(guid), f"Expected UUID v4, got: {guid}"

def test_get_guid_is_stable_across_calls(cm):
    assert cm.get_guid(URL_BONDS) == cm.get_guid(URL_BONDS)


# ---------------------------------------------------------------------------
# is_url_cached
# ---------------------------------------------------------------------------

def test_is_url_cached_true_for_success_entry(cm):
    assert cm.is_url_cached(URL_BONDS, {URL_BONDS: {"status": "success"}}) is True

def test_is_url_cached_false_for_missing_url(cm):
    assert cm.is_url_cached(URL_BONDS, {}) is False

def test_is_url_cached_false_for_failed_entry(cm):
    assert cm.is_url_cached(URL_BONDS, {URL_BONDS: {"status": "failed"}}) is False


# ---------------------------------------------------------------------------
# get_failed_urls
# ---------------------------------------------------------------------------

def test_get_failed_urls_returns_only_failed(cm):
    mapping = {
        "url1": {"status": "failed"},
        "url2": {"status": "success"},
        "url3": {"status": "failed"},
    }
    assert set(cm.get_failed_urls(mapping)) == {"url1", "url3"}

def test_get_failed_urls_empty_mapping(cm):
    assert cm.get_failed_urls({}) == []


# ---------------------------------------------------------------------------
# get_pending_scrape_urls
# ---------------------------------------------------------------------------

def test_get_pending_scrape_urls_returns_pending_with_success_status(cm):
    mapping = {
        "url1": {"status": "success", "scrape_status": "pending"},
        "url2": {"status": "success", "scrape_status": "success"},
        "url3": {"status": "failed", "scrape_status": "pending"},
    }
    assert cm.get_pending_scrape_urls(mapping) == ["url1"]

def test_get_pending_scrape_urls_empty_mapping(cm):
    assert cm.get_pending_scrape_urls({}) == []


# ---------------------------------------------------------------------------
# load / save round-trips
# ---------------------------------------------------------------------------

def test_save_and_load_html_mapping_round_trip(cm):
    mapping = {URL_BONDS: {"status": "success", "file_path": "some/path.html"}}
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
