import re
import pytest
import requests
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rag.scraper.cache_manager import CacheManager
from src.rag.scraper.url_downloader import UrlDownloader

UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

URL = (
    "https://www.investor.gov/introduction-investing/investing-basics"
    "/investment-products/stocks"
)
URL2 = (
    "https://www.investor.gov/introduction-investing/investing-basics"
    "/investment-products/bonds-or-fixed-income-products/bonds"
)
URL3 = (
    "https://www.investor.gov/introduction-investing/investing-basics"
    "/investment-products/mutual-funds"
)


def _mock_response(status_code=200, text="<html><body><p>Content</p></body></html>",
                   content_type="text/html"):
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    r.headers = {"Content-Type": content_type}
    return r


@pytest.fixture
def cm(tmp_path):
    return CacheManager(base_path=tmp_path)


@pytest.fixture
def downloader(cm):
    return UrlDownloader(cache_manager=cm, delay_seconds=0.0)


# ---------------------------------------------------------------------------
# download_url — success path
# ---------------------------------------------------------------------------

def test_successful_download_sets_status_success(downloader):
    with patch.object(downloader.session, "get", return_value=_mock_response()):
        entry = downloader.download_url(URL)
    assert entry["status"] == "success"

def test_successful_download_populates_all_rich_fields(downloader):
    with patch.object(downloader.session, "get", return_value=_mock_response()):
        entry = downloader.download_url(URL)
    assert entry["http_status_code"] == 200
    assert entry["downloaded_at"] is not None
    assert entry["scrape_status"] == "pending"
    assert entry["source_domain"] == "investor.gov"
    assert entry["error_message"] is None
    assert entry["file_path"] is not None
    assert "guid" in entry
    assert UUID4_RE.match(entry["guid"]), f"Expected UUID v4 guid, got: {entry['guid']}"
    assert entry["domain"] == "investor.gov"
    assert entry["url"] == URL

def test_successful_download_writes_html_file(downloader):
    with patch.object(downloader.session, "get", return_value=_mock_response()):
        entry = downloader.download_url(URL)
    assert Path(entry["file_path"]).exists()

def test_successful_download_saves_mapping(downloader, cm):
    with patch.object(downloader.session, "get", return_value=_mock_response()):
        downloader.download_url(URL)
    mapping = cm.load_html_mapping()
    assert URL in mapping
    assert mapping[URL]["status"] == "success"


# ---------------------------------------------------------------------------
# download_url — failure paths
# ---------------------------------------------------------------------------

def test_http_404_sets_failed(downloader):
    with patch.object(downloader.session, "get", return_value=_mock_response(404)):
        entry = downloader.download_url(URL)
    assert entry["status"] == "failed"
    assert "404" in entry["error_message"]

def test_http_500_sets_failed(downloader):
    with patch.object(downloader.session, "get", return_value=_mock_response(500)):
        entry = downloader.download_url(URL)
    assert entry["status"] == "failed"
    assert "500" in entry["error_message"]

def test_http_error_does_not_raise(downloader):
    with patch.object(downloader.session, "get", return_value=_mock_response(503)):
        entry = downloader.download_url(URL)
    assert entry is not None

def test_timeout_sets_failed(downloader):
    with patch.object(downloader.session, "get", side_effect=requests.exceptions.Timeout("timed out")):
        entry = downloader.download_url(URL)
    assert entry["status"] == "failed"
    assert entry["error_message"] is not None

def test_timeout_does_not_raise(downloader):
    with patch.object(downloader.session, "get", side_effect=requests.exceptions.Timeout()):
        entry = downloader.download_url(URL)
    assert entry is not None


# ---------------------------------------------------------------------------
# download_url — cache behaviour
# ---------------------------------------------------------------------------

def test_cached_url_is_skipped_when_force_refresh_false(cm):
    cm.save_html_mapping({URL: {"status": "success", "file_path": "cached.html"}})
    dl = UrlDownloader(cache_manager=cm, force_refresh=False)
    with patch.object(dl.session, "get") as mock_get:
        entry = dl.download_url(URL)
    mock_get.assert_not_called()
    assert entry["status"] == "success"

def test_cached_url_is_redownloaded_when_force_refresh_true(cm):
    cm.save_html_mapping({URL: {"status": "success"}})
    dl = UrlDownloader(cache_manager=cm, force_refresh=True)
    with patch.object(dl.session, "get", return_value=_mock_response()):
        entry = dl.download_url(URL)
    assert entry["downloaded_at"] is not None


# ---------------------------------------------------------------------------
# User-Agent
# ---------------------------------------------------------------------------

def test_user_agent_header_is_set(downloader):
    assert "User-Agent" in downloader.session.headers
    assert len(downloader.session.headers["User-Agent"]) > 10


# ---------------------------------------------------------------------------
# download_all
# ---------------------------------------------------------------------------

def test_download_all_calls_download_url_once_per_url(downloader):
    urls = [URL, URL2]
    with patch.object(downloader, "download_url", return_value={"status": "success"}) as mock:
        downloader.download_all(urls)
    assert mock.call_count == len(urls)

def test_download_all_respects_delay_between_requests(cm):
    dl = UrlDownloader(cache_manager=cm, delay_seconds=0.5)
    urls = [URL, URL2, URL3]
    with patch.object(dl, "download_url", return_value={"status": "success"}):
        with patch("src.rag.scraper.url_downloader.time.sleep") as mock_sleep:
            dl.download_all(urls)
    assert mock_sleep.call_count == len(urls) - 1
    mock_sleep.assert_called_with(0.5)

def test_download_all_returns_full_mapping(cm):
    dl = UrlDownloader(cache_manager=cm, delay_seconds=0.0)
    urls = [URL, URL2]
    with patch.object(dl.session, "get", return_value=_mock_response()):
        result = dl.download_all(urls)
    assert isinstance(result, dict)
    assert URL in result
    assert URL2 in result
