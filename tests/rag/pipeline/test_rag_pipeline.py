import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.rag.pipeline.rag_pipeline import _build_parser, resolve_urls, run

_MODULE = "src.rag.pipeline.rag_pipeline"

URL_STOCKS = (
    "https://www.investor.gov/introduction-investing/investing-basics"
    "/investment-products/stocks"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patched_run(tmp_path, action="download", urls=None, force_refresh=False,
                 verbose=False, dl_return=None, scraper_return=None,
                 dl_side_effect=None, scraper_side_effect=None):
    """Run run() with CacheManager, UrlDownloader, HtmlScraper all mocked."""
    if urls is None:
        urls = [URL_STOCKS]
    if dl_return is None:
        dl_return = {}
    if scraper_return is None:
        scraper_return = {}

    with patch(f"{_MODULE}.CacheManager") as MockCM, \
         patch(f"{_MODULE}.UrlDownloader") as MockDL, \
         patch(f"{_MODULE}.HtmlScraper") as MockScraper:

        MockDL.return_value.download_all.return_value = dl_return
        MockDL.return_value.download_all.side_effect = dl_side_effect

        MockScraper.return_value.scrape_all.return_value = scraper_return
        MockScraper.return_value.scrape_all.side_effect = scraper_side_effect

        result = run(tmp_path, action, urls, force_refresh=force_refresh, verbose=verbose)
        return result, MockCM, MockDL, MockScraper


# ---------------------------------------------------------------------------
# resolve_urls
# ---------------------------------------------------------------------------

def test_http_url_returns_single_item_list():
    assert resolve_urls("http://example.com/page") == ["http://example.com/page"]

def test_https_url_returns_single_item_list():
    assert resolve_urls("https://example.com/page") == ["https://example.com/page"]

def test_file_with_three_urls_returns_list_of_three(tmp_path):
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://a.com\nhttps://b.com\nhttps://c.com\n")
    assert resolve_urls(str(url_file)) == ["https://a.com", "https://b.com", "https://c.com"]

def test_file_skips_blank_lines(tmp_path):
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://a.com\n\nhttps://b.com\n\n")
    assert resolve_urls(str(url_file)) == ["https://a.com", "https://b.com"]

def test_file_skips_comment_lines(tmp_path):
    url_file = tmp_path / "urls.txt"
    url_file.write_text("# comment\nhttps://a.com\n# another comment\nhttps://b.com\n")
    assert resolve_urls(str(url_file)) == ["https://a.com", "https://b.com"]

def test_file_strips_whitespace_from_urls(tmp_path):
    url_file = tmp_path / "urls.txt"
    url_file.write_text("  https://a.com  \n  https://b.com  \n")
    assert resolve_urls(str(url_file)) == ["https://a.com", "https://b.com"]

def test_non_url_nonexistent_path_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        resolve_urls("/nonexistent/path/to/urls.txt")

def test_file_resolving_to_empty_raises_value_error(tmp_path):
    url_file = tmp_path / "empty.txt"
    url_file.write_text("# only comments\n\n  \n")
    with pytest.raises(ValueError):
        resolve_urls(str(url_file))


# ---------------------------------------------------------------------------
# run() — action routing
# ---------------------------------------------------------------------------

def test_download_action_calls_download_all_once(tmp_path):
    _, _, MockDL, _ = _patched_run(tmp_path, action="download")
    MockDL.return_value.download_all.assert_called_once()

def test_download_action_does_not_call_scrape_all(tmp_path):
    _, _, _, MockScraper = _patched_run(tmp_path, action="download")
    MockScraper.return_value.scrape_all.assert_not_called()

def test_scrape_action_calls_download_all_once(tmp_path):
    _, _, MockDL, _ = _patched_run(tmp_path, action="scrape")
    MockDL.return_value.download_all.assert_called_once()

def test_scrape_action_calls_scrape_all_once(tmp_path):
    _, _, _, MockScraper = _patched_run(tmp_path, action="scrape")
    MockScraper.return_value.scrape_all.assert_called_once()

def test_scrape_action_calls_download_before_scrape(tmp_path):
    call_order = []

    with patch(f"{_MODULE}.CacheManager"), \
         patch(f"{_MODULE}.UrlDownloader") as MockDL, \
         patch(f"{_MODULE}.HtmlScraper") as MockScraper:

        MockDL.return_value.download_all.side_effect = (
            lambda urls: call_order.append("download") or {}
        )
        MockScraper.return_value.scrape_all.side_effect = (
            lambda **kw: call_order.append("scrape") or {}
        )
        run(tmp_path, "scrape", [URL_STOCKS])

    assert call_order == ["download", "scrape"]


# ---------------------------------------------------------------------------
# run() — force_refresh passthrough
# ---------------------------------------------------------------------------

def test_force_refresh_passed_to_url_downloader_constructor(tmp_path):
    with patch(f"{_MODULE}.CacheManager") as MockCM, \
         patch(f"{_MODULE}.UrlDownloader") as MockDL, \
         patch(f"{_MODULE}.HtmlScraper"):

        MockDL.return_value.download_all.return_value = {}
        run(tmp_path, "download", [URL_STOCKS], force_refresh=True)

    _, kwargs = MockDL.call_args
    assert kwargs.get("force_refresh") is True

def test_force_refresh_passed_to_scrape_all(tmp_path):
    _, _, _, MockScraper = _patched_run(tmp_path, action="scrape", force_refresh=True)
    MockScraper.return_value.scrape_all.assert_called_once_with(force_refresh=True)


# ---------------------------------------------------------------------------
# run() — return codes
# ---------------------------------------------------------------------------

def test_run_returns_0_on_success(tmp_path):
    result, _, _, _ = _patched_run(tmp_path, action="scrape")
    assert result == 0

def test_run_returns_1_when_download_all_raises(tmp_path):
    result, _, _, _ = _patched_run(
        tmp_path, action="download",
        dl_side_effect=RuntimeError("network error"),
    )
    assert result == 1

def test_run_returns_1_when_scrape_all_raises(tmp_path):
    result, _, _, _ = _patched_run(
        tmp_path, action="scrape",
        scraper_side_effect=RuntimeError("parse error"),
    )
    assert result == 1

def test_run_never_raises_exceptions(tmp_path):
    result, _, _, _ = _patched_run(
        tmp_path, action="scrape",
        dl_side_effect=Exception("unexpected"),
    )
    assert result == 1


# ---------------------------------------------------------------------------
# run() — CacheManager initialization
# ---------------------------------------------------------------------------

def test_cache_manager_initialized_with_provided_cache_path(tmp_path):
    with patch(f"{_MODULE}.CacheManager") as MockCM, \
         patch(f"{_MODULE}.UrlDownloader") as MockDL, \
         patch(f"{_MODULE}.HtmlScraper"):

        MockDL.return_value.download_all.return_value = {}
        run(tmp_path, "download", [URL_STOCKS])

    MockCM.assert_called_once_with(base_path=tmp_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return _build_parser()


def test_missing_cache_path_causes_system_exit(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["--action", "download", "--url-path", URL_STOCKS])

def test_missing_action_causes_system_exit(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["--cache-path", "/tmp", "--url-path", URL_STOCKS])

def test_missing_url_path_causes_system_exit(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["--cache-path", "/tmp", "--action", "download"])

def test_invalid_action_causes_system_exit(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["--cache-path", "/tmp", "--action", "invalid", "--url-path", URL_STOCKS])

def test_force_refresh_defaults_to_false(parser):
    args = parser.parse_args(["--cache-path", "/tmp", "--action", "download", "--url-path", URL_STOCKS])
    assert args.force_refresh is False

def test_verbose_defaults_to_false(parser):
    args = parser.parse_args(["--cache-path", "/tmp", "--action", "download", "--url-path", URL_STOCKS])
    assert args.verbose is False

def test_force_refresh_flag_sets_true(parser):
    args = parser.parse_args([
        "--cache-path", "/tmp", "--action", "download",
        "--url-path", URL_STOCKS, "--force-refresh",
    ])
    assert args.force_refresh is True

def test_verbose_flag_sets_true(parser):
    args = parser.parse_args([
        "--cache-path", "/tmp", "--action", "download",
        "--url-path", URL_STOCKS, "--verbose",
    ])
    assert args.verbose is True
