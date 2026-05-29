import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rag.scraper.cache_manager import CacheManager
from src.rag.scraper.html_scraper import HtmlScraper
from src.rag.scraper.url_downloader import UrlDownloader

URL_STOCKS = (
    "https://www.investor.gov/introduction-investing/investing-basics"
    "/investment-products/stocks"
)
URL_BONDS = (
    "https://www.investor.gov/introduction-investing/investing-basics"
    "/investment-products/bonds-or-fixed-income-products/bonds"
)

SAMPLE_HTML = """\
<html>
<head>
  <title>Stocks | Investor.gov</title>
  <meta name="description" content="Learn about stocks.">
</head>
<body>
  <nav class="breadcrumb"><a href="/">Home</a> / Stocks</nav>
  <article>
    <h1>Stocks</h1>
    <p>A stock is a share of ownership in a company. When you buy stock,
    you become a part owner of that company and may receive dividends.</p>
    <h2>Types of Stock</h2>
    <ul>
      <li>Common stock</li>
      <li>Preferred stock</li>
    </ul>
    <aside class="related">Related: Bonds, ETFs</aside>
  </article>
  <footer>U.S. Securities and Exchange Commission</footer>
</body>
</html>
"""


def _make_mock_response(status_code=200, html=SAMPLE_HTML):
    r = MagicMock()
    r.status_code = status_code
    r.text = html
    r.headers = {"Content-Type": "text/html; charset=utf-8"}
    return r


class TestPipelineEndToEnd:
    def test_full_pipeline_creates_correct_folder_structure(self, tmp_path):
        cm = CacheManager(base_path=tmp_path)
        dl = UrlDownloader(cache_manager=cm, delay_seconds=0.0)
        scraper = HtmlScraper(cache_manager=cm)

        with patch.object(dl.session, "get", return_value=_make_mock_response()):
            dl.download_all([URL_STOCKS])

        scraper.scrape_all()

        assert (tmp_path / "html_cache").is_dir()
        assert (tmp_path / "scraper_cache").is_dir()
        assert (tmp_path / "html_cache" / "html_cache_mapping.json").exists()
        assert (tmp_path / "scraper_cache" / "scraper_cache_mapping.json").exists()

        html_files = list((tmp_path / "html_cache" / "investor.gov").glob("*.html"))
        assert len(html_files) == 1

        txt_files = list((tmp_path / "scraper_cache" / "investor.gov").glob("*.txt"))
        assert len(txt_files) == 1

    def test_full_pipeline_mapping_files_are_consistent(self, tmp_path):
        cm = CacheManager(base_path=tmp_path)
        dl = UrlDownloader(cache_manager=cm, delay_seconds=0.0)
        scraper = HtmlScraper(cache_manager=cm)

        with patch.object(dl.session, "get", return_value=_make_mock_response()):
            dl.download_all([URL_STOCKS])

        scraper.scrape_all()

        html_mapping = cm.load_html_mapping()
        assert URL_STOCKS in html_mapping
        assert html_mapping[URL_STOCKS]["status"] == "success"
        html_file = Path(html_mapping[URL_STOCKS]["file_path"])
        assert html_file.exists()

        scraper_mapping = cm.load_scraper_mapping()
        assert URL_STOCKS in scraper_mapping
        assert scraper_mapping[URL_STOCKS]["status"] == "success"
        txt_file = Path(scraper_mapping[URL_STOCKS]["file_path"])
        assert txt_file.exists()

    def test_full_pipeline_isolation(self, tmp_path):
        cm1 = CacheManager(base_path=tmp_path / "run1")
        cm2 = CacheManager(base_path=tmp_path / "run2")
        dl1 = UrlDownloader(cache_manager=cm1, delay_seconds=0.0)
        dl2 = UrlDownloader(cache_manager=cm2, delay_seconds=0.0)

        with patch.object(dl1.session, "get", return_value=_make_mock_response()):
            dl1.download_all([URL_STOCKS])

        with patch.object(dl2.session, "get", return_value=_make_mock_response()):
            dl2.download_all([URL_BONDS])

        mapping1 = cm1.load_html_mapping()
        mapping2 = cm2.load_html_mapping()

        assert URL_STOCKS in mapping1
        assert URL_BONDS not in mapping1

        assert URL_BONDS in mapping2
        assert URL_STOCKS not in mapping2

        html_files1 = list((tmp_path / "run1" / "html_cache").rglob("*.html"))
        html_files2 = list((tmp_path / "run2" / "html_cache").rglob("*.html"))
        assert len(html_files1) == 1
        assert len(html_files2) == 1
        assert html_files1[0] != html_files2[0]

    def test_download_failure_does_not_break_pipeline(self, tmp_path):
        cm = CacheManager(base_path=tmp_path)
        dl = UrlDownloader(cache_manager=cm, delay_seconds=0.0)
        scraper = HtmlScraper(cache_manager=cm)

        def side_effect(url, **kwargs):
            if "bonds" in url:
                return _make_mock_response(status_code=404)
            return _make_mock_response(status_code=200)

        with patch.object(dl.session, "get", side_effect=side_effect):
            dl.download_all([URL_BONDS, URL_STOCKS])

        scraper.scrape_all()

        html_mapping = cm.load_html_mapping()
        assert html_mapping[URL_BONDS]["status"] == "failed"
        assert html_mapping[URL_STOCKS]["status"] == "success"

        scraper_mapping = cm.load_scraper_mapping()
        assert URL_STOCKS in scraper_mapping
        assert scraper_mapping[URL_STOCKS]["status"] == "success"
        assert URL_BONDS not in scraper_mapping
