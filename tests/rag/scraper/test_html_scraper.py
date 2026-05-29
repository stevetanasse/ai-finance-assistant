import pytest
from pathlib import Path

from src.rag.scraper.cache_manager import CacheManager
from src.rag.scraper.html_scraper import HtmlScraper

URL = (
    "https://www.investor.gov/introduction-investing/investing-basics"
    "/investment-products/stocks"
)
URL_UNKNOWN = "https://www.example.com/page"

INVESTOR_HTML = """\
<html>
<head><title>Stocks | Investor.gov</title></head>
<body>
  <nav class="breadcrumb">Home / Stocks</nav>
  <article>
    <h1>Stocks</h1>
    <p>A stock is a share of ownership in a company.</p>
    <aside class="related">Related: Bonds</aside>
    <nav>In-page navigation</nav>
  </article>
  <aside class="sidebar">Sidebar ads</aside>
  <footer>Footer content</footer>
</body>
</html>
"""

GENERIC_HTML = """\
<html><body><p>Hello world from example.com</p></body></html>
"""


@pytest.fixture
def cm(tmp_path):
    return CacheManager(base_path=tmp_path)


@pytest.fixture
def scraper(cm):
    return HtmlScraper(cache_manager=cm)


def _make_entry(url: str, domain: str, html: str, cm: CacheManager) -> dict:
    html_path = cm.get_cache_filepath(url, cm.html_cache_dir)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    scraper_path = cm.get_cache_filepath(url, cm.scraper_cache_dir)
    return {
        "file_path": str(html_path),
        "scraped_path": str(scraper_path),
        "downloaded_at": "2026-05-28T10:00:00+00:00",
        "scraped_at": None,
        "status": "success",
        "scrape_status": "pending",
        "http_status_code": 200,
        "content_type": "text/html",
        "word_count": None,
        "source_domain": domain,
        "error_message": None,
    }


# ---------------------------------------------------------------------------
# scrape_url — extractor selection
# ---------------------------------------------------------------------------

def test_scrape_url_selects_investor_gov_extractor(scraper, cm):
    entry = _make_entry(URL, "investor.gov", INVESTOR_HTML, cm)
    result = scraper.scrape_url(URL, {URL: entry})
    assert result["scrape_status"] == "success"
    text = Path(result["scraped_path"]).read_text(encoding="utf-8")
    assert "Related: Bonds" not in text
    assert "In-page navigation" not in text

def test_scrape_url_uses_generic_extractor_for_unregistered_domain(scraper, cm):
    entry = _make_entry(URL_UNKNOWN, "example.com", GENERIC_HTML, cm)
    result = scraper.scrape_url(URL_UNKNOWN, {URL_UNKNOWN: entry})
    assert result["scrape_status"] == "success"
    text = Path(result["scraped_path"]).read_text(encoding="utf-8")
    assert "Hello world" in text


# ---------------------------------------------------------------------------
# scrape_url — entry updates
# ---------------------------------------------------------------------------

def test_scrape_url_sets_scrape_status_success(scraper, cm):
    entry = _make_entry(URL, "investor.gov", INVESTOR_HTML, cm)
    result = scraper.scrape_url(URL, {URL: entry})
    assert result["scrape_status"] == "success"

def test_scrape_url_sets_scrape_status_failed_on_missing_file(scraper, cm):
    bad_path = str(cm.html_cache_dir / "missing" / "file.html")
    bad_entry = {"file_path": bad_path, "source_domain": "investor.gov"}
    result = scraper.scrape_url(URL, {URL: bad_entry})
    assert result["scrape_status"] == "failed"
    assert result["error_message"] is not None

def test_scrape_url_does_not_raise_on_error(scraper, cm):
    bad_path = str(cm.html_cache_dir / "missing" / "file.html")
    bad_entry = {"file_path": bad_path, "source_domain": "investor.gov"}
    result = scraper.scrape_url(URL, {URL: bad_entry})
    assert result is not None

def test_scrape_url_populates_word_count(scraper, cm):
    entry = _make_entry(URL, "investor.gov", INVESTOR_HTML, cm)
    result = scraper.scrape_url(URL, {URL: entry})
    assert result["word_count"] is not None
    assert result["word_count"] > 0

def test_scrape_url_populates_scraped_at(scraper, cm):
    entry = _make_entry(URL, "investor.gov", INVESTOR_HTML, cm)
    result = scraper.scrape_url(URL, {URL: entry})
    assert result["scraped_at"] is not None


# ---------------------------------------------------------------------------
# scrape_all — filtering
# ---------------------------------------------------------------------------

def test_scrape_all_skips_entries_with_failed_download_status(scraper, cm):
    html_path = str(cm.html_cache_dir / "investor.gov" / "page.html")
    cm.save_html_mapping({
        URL: {
            "status": "failed",
            "scrape_status": "pending",
            "source_domain": "investor.gov",
            "file_path": html_path,
        }
    })
    result = scraper.scrape_all()
    assert result == {}

def test_scrape_all_skips_already_scraped_without_force_refresh(scraper, cm):
    entry = _make_entry(URL, "investor.gov", INVESTOR_HTML, cm)
    entry["scrape_status"] = "success"
    cm.save_html_mapping({URL: entry})
    result = scraper.scrape_all(force_refresh=False)
    assert result == {}

def test_scrape_all_rescapes_when_force_refresh_true(scraper, cm):
    entry = _make_entry(URL, "investor.gov", INVESTOR_HTML, cm)
    entry["scrape_status"] = "success"
    cm.save_html_mapping({URL: entry})
    result = scraper.scrape_all(force_refresh=True)
    assert URL in result
    assert result[URL]["status"] == "success"
