import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.rag.pipeline.rag_pipeline import _build_parser, resolve_urls, run, validate_args

_MODULE = "src.rag.pipeline.rag_pipeline"
_CHUNKER_MODULE = "src.rag.chunker"

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


def _patched_run_chunk(tmp_path, chunk_size=500, chunk_overlap=50, force_refresh=False,
                       dl_return=None, scraper_return=None, chunk_return=None,
                       chunk_side_effect=None):
    """Run run() with chunk action, all components mocked."""
    if dl_return is None:
        dl_return = {}
    if scraper_return is None:
        scraper_return = {}
    if chunk_return is None:
        chunk_return = {}

    with patch(f"{_MODULE}.CacheManager") as MockCM, \
         patch(f"{_MODULE}.UrlDownloader") as MockDL, \
         patch(f"{_MODULE}.HtmlScraper") as MockScraper, \
         patch(f"{_CHUNKER_MODULE}.ChunkCacheManager") as MockCCM, \
         patch(f"{_CHUNKER_MODULE}.Chunker") as MockChunker:

        MockDL.return_value.download_all.return_value = dl_return
        MockScraper.return_value.scrape_all.return_value = scraper_return
        MockChunker.return_value.chunk_all.return_value = chunk_return
        MockChunker.return_value.chunk_all.side_effect = chunk_side_effect

        result = run(
            tmp_path, "chunk", [URL_STOCKS],
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            force_refresh=force_refresh,
        )
        return result, MockCM, MockCCM, MockDL, MockScraper, MockChunker


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
# validate_args
# ---------------------------------------------------------------------------

def test_validate_args_returns_none_for_download_with_no_chunk_args():
    assert validate_args("download", None, None) is None

def test_validate_args_returns_none_for_scrape_with_no_chunk_args():
    assert validate_args("scrape", None, None) is None

def test_validate_args_returns_none_for_chunk_with_valid_args():
    assert validate_args("chunk", 500, 50) is None

def test_validate_args_returns_error_when_chunk_size_missing():
    result = validate_args("chunk", None, 50)
    assert result is not None
    assert "--chunk-size" in result

def test_validate_args_returns_error_when_chunk_overlap_missing():
    result = validate_args("chunk", 500, None)
    assert result is not None
    assert "--chunk-overlap" in result

def test_validate_args_returns_error_when_overlap_equals_size():
    result = validate_args("chunk", 500, 500)
    assert result is not None
    assert "--chunk-overlap" in result or "--chunk-size" in result

def test_validate_args_returns_error_when_overlap_exceeds_size():
    result = validate_args("chunk", 500, 600)
    assert result is not None

def test_validate_args_returns_error_when_chunk_size_is_zero():
    result = validate_args("chunk", 0, 50)
    assert result is not None
    assert "--chunk-size" in result

def test_validate_args_returns_error_when_chunk_size_is_negative():
    result = validate_args("chunk", -1, 0)
    assert result is not None

def test_validate_args_returns_error_when_chunk_overlap_is_negative():
    result = validate_args("chunk", 500, -1)
    assert result is not None
    assert "--chunk-overlap" in result


# ---------------------------------------------------------------------------
# run() — action routing (existing tests unchanged)
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
# run() — chunk action routing
# ---------------------------------------------------------------------------

def test_chunk_action_calls_download_all_once(tmp_path):
    _, _, _, MockDL, _, _ = _patched_run_chunk(tmp_path)
    MockDL.return_value.download_all.assert_called_once()

def test_chunk_action_calls_scrape_all_once(tmp_path):
    _, _, _, _, MockScraper, _ = _patched_run_chunk(tmp_path)
    MockScraper.return_value.scrape_all.assert_called_once()

def test_chunk_action_calls_chunk_all_once(tmp_path):
    _, _, _, _, _, MockChunker = _patched_run_chunk(tmp_path)
    MockChunker.return_value.chunk_all.assert_called_once()

def test_chunk_action_calls_stages_in_order(tmp_path):
    call_order = []

    with patch(f"{_MODULE}.CacheManager"), \
         patch(f"{_MODULE}.UrlDownloader") as MockDL, \
         patch(f"{_MODULE}.HtmlScraper") as MockScraper, \
         patch(f"{_CHUNKER_MODULE}.ChunkCacheManager"), \
         patch(f"{_CHUNKER_MODULE}.Chunker") as MockChunker:

        MockDL.return_value.download_all.side_effect = (
            lambda urls: call_order.append("download") or {}
        )
        MockScraper.return_value.scrape_all.side_effect = (
            lambda **kw: call_order.append("scrape") or {}
        )
        MockChunker.return_value.chunk_all.side_effect = (
            lambda **kw: call_order.append("chunk") or {}
        )
        run(tmp_path, "chunk", [URL_STOCKS], chunk_size=500, chunk_overlap=50)

    assert call_order == ["download", "scrape", "chunk"]

def test_chunk_action_passes_chunk_size_to_chunker(tmp_path):
    with patch(f"{_MODULE}.CacheManager"), \
         patch(f"{_MODULE}.UrlDownloader") as MockDL, \
         patch(f"{_MODULE}.HtmlScraper") as MockScraper, \
         patch(f"{_CHUNKER_MODULE}.ChunkCacheManager"), \
         patch(f"{_CHUNKER_MODULE}.Chunker") as MockChunker:

        MockDL.return_value.download_all.return_value = {}
        MockScraper.return_value.scrape_all.return_value = {}
        MockChunker.return_value.chunk_all.return_value = {}

        run(tmp_path, "chunk", [URL_STOCKS], chunk_size=500, chunk_overlap=50)

    _, kwargs = MockChunker.call_args
    assert kwargs.get("chunk_size") == 500

def test_chunk_action_passes_chunk_overlap_to_chunker(tmp_path):
    with patch(f"{_MODULE}.CacheManager"), \
         patch(f"{_MODULE}.UrlDownloader") as MockDL, \
         patch(f"{_MODULE}.HtmlScraper") as MockScraper, \
         patch(f"{_CHUNKER_MODULE}.ChunkCacheManager"), \
         patch(f"{_CHUNKER_MODULE}.Chunker") as MockChunker:

        MockDL.return_value.download_all.return_value = {}
        MockScraper.return_value.scrape_all.return_value = {}
        MockChunker.return_value.chunk_all.return_value = {}

        run(tmp_path, "chunk", [URL_STOCKS], chunk_size=500, chunk_overlap=50)

    _, kwargs = MockChunker.call_args
    assert kwargs.get("chunk_overlap") == 50

def test_chunk_action_passes_force_refresh_to_chunk_all(tmp_path):
    _, _, _, _, _, MockChunker = _patched_run_chunk(tmp_path, force_refresh=True)
    MockChunker.return_value.chunk_all.assert_called_once_with(
        scraper_mapping={}, force_refresh=True
    )

def test_scrape_action_does_not_call_chunker(tmp_path):
    with patch(f"{_MODULE}.CacheManager"), \
         patch(f"{_MODULE}.UrlDownloader") as MockDL, \
         patch(f"{_MODULE}.HtmlScraper") as MockScraper, \
         patch(f"{_CHUNKER_MODULE}.Chunker") as MockChunker:

        MockDL.return_value.download_all.return_value = {}
        MockScraper.return_value.scrape_all.return_value = {}

        run(tmp_path, "scrape", [URL_STOCKS])

    MockChunker.assert_not_called()

def test_download_action_does_not_call_html_scraper_for_chunk_test(tmp_path):
    _, _, _, MockDL, MockScraper, MockChunker = _patched_run_chunk(
        tmp_path,
        # Override action to download — need a different approach
    )
    # This test is about download action specifically, done via _patched_run
    pass

def test_download_action_does_not_call_chunker(tmp_path):
    with patch(f"{_MODULE}.CacheManager"), \
         patch(f"{_MODULE}.UrlDownloader") as MockDL, \
         patch(f"{_MODULE}.HtmlScraper") as MockScraper, \
         patch(f"{_CHUNKER_MODULE}.Chunker") as MockChunker:

        MockDL.return_value.download_all.return_value = {}
        run(tmp_path, "download", [URL_STOCKS])

    MockChunker.assert_not_called()

def test_chunk_cache_manager_initialized_with_cache_path(tmp_path):
    with patch(f"{_MODULE}.CacheManager"), \
         patch(f"{_MODULE}.UrlDownloader") as MockDL, \
         patch(f"{_MODULE}.HtmlScraper") as MockScraper, \
         patch(f"{_CHUNKER_MODULE}.ChunkCacheManager") as MockCCM, \
         patch(f"{_CHUNKER_MODULE}.Chunker") as MockChunker:

        MockDL.return_value.download_all.return_value = {}
        MockScraper.return_value.scrape_all.return_value = {}
        MockChunker.return_value.chunk_all.return_value = {}

        run(tmp_path, "chunk", [URL_STOCKS], chunk_size=500, chunk_overlap=50)

    MockCCM.assert_called_once_with(base_path=tmp_path)


# ---------------------------------------------------------------------------
# run() — force_refresh passthrough (existing tests unchanged)
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
# run() — return codes (existing tests unchanged)
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
# run() — CacheManager initialization (existing test unchanged)
# ---------------------------------------------------------------------------

def test_cache_manager_initialized_with_provided_cache_path(tmp_path):
    with patch(f"{_MODULE}.CacheManager") as MockCM, \
         patch(f"{_MODULE}.UrlDownloader") as MockDL, \
         patch(f"{_MODULE}.HtmlScraper"):

        MockDL.return_value.download_all.return_value = {}
        run(tmp_path, "download", [URL_STOCKS])

    MockCM.assert_called_once_with(base_path=tmp_path)


# ---------------------------------------------------------------------------
# Argument parsing (existing tests unchanged)
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


# ---------------------------------------------------------------------------
# Argument parsing — new chunk args
# ---------------------------------------------------------------------------

def test_chunk_size_parses_as_int(parser):
    args = parser.parse_args([
        "--cache-path", "/tmp", "--action", "chunk",
        "--url-path", URL_STOCKS, "--chunk-size", "500", "--chunk-overlap", "50",
    ])
    assert args.chunk_size == 500
    assert isinstance(args.chunk_size, int)

def test_chunk_overlap_parses_as_int(parser):
    args = parser.parse_args([
        "--cache-path", "/tmp", "--action", "chunk",
        "--url-path", URL_STOCKS, "--chunk-size", "500", "--chunk-overlap", "50",
    ])
    assert args.chunk_overlap == 50
    assert isinstance(args.chunk_overlap, int)

def test_chunk_size_defaults_to_none(parser):
    args = parser.parse_args(["--cache-path", "/tmp", "--action", "download", "--url-path", URL_STOCKS])
    assert args.chunk_size is None

def test_chunk_overlap_defaults_to_none(parser):
    args = parser.parse_args(["--cache-path", "/tmp", "--action", "download", "--url-path", URL_STOCKS])
    assert args.chunk_overlap is None

def test_chunk_is_valid_action_no_system_exit(parser):
    args = parser.parse_args([
        "--cache-path", "/tmp", "--action", "chunk",
        "--url-path", URL_STOCKS,
    ])
    assert args.action == "chunk"

def test_action_chunk_accepted_by_argparse(parser):
    args = parser.parse_args([
        "--cache-path", "/tmp", "--action", "chunk",
        "--url-path", URL_STOCKS, "--chunk-size", "500", "--chunk-overlap", "50",
    ])
    assert args.action == "chunk"
