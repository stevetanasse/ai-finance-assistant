import json
import re
import time
from pathlib import Path

import pytest

from src.rag.pipeline.rag_pipeline import run

INVESTOR_GOV_STOCKS_URL = (
    "https://www.investor.gov/introduction-investing/"
    "investing-basics/investment-products/stocks"
)


@pytest.mark.integration
class TestRagPipelineE2E:
    def test_download_action_creates_html_cache(self, tmp_path):
        code = run(
            cache_path=tmp_path,
            action="download",
            urls=[INVESTOR_GOV_STOCKS_URL],
        )

        assert code == 0
        assert (tmp_path / "html_cache").is_dir()
        assert (tmp_path / "html_cache" / "investor.gov").is_dir()

        html_files = list((tmp_path / "html_cache" / "investor.gov").glob("*.html"))
        assert len(html_files) >= 1

        mapping_file = tmp_path / "html_cache" / "html_cache_mapping.json"
        assert mapping_file.exists()
        mapping = json.loads(mapping_file.read_text())
        assert mapping[INVESTOR_GOV_STOCKS_URL]["status"] == "success"

        html_file = html_files[0]
        assert html_file.stat().st_size > 0
        content = html_file.read_text(encoding="utf-8", errors="replace").lower()
        assert "<html" in content or "<!doctype" in content

    def test_scrape_action_creates_both_caches(self, tmp_path):
        code = run(
            cache_path=tmp_path,
            action="scrape",
            urls=[INVESTOR_GOV_STOCKS_URL],
        )

        assert code == 0
        assert (tmp_path / "html_cache").is_dir()
        assert (tmp_path / "scraper_cache").is_dir()

        html_mapping_file = tmp_path / "html_cache" / "html_cache_mapping.json"
        scraper_mapping_file = tmp_path / "scraper_cache" / "scraper_cache_mapping.json"
        assert html_mapping_file.exists()
        assert scraper_mapping_file.exists()

        html_mapping = json.loads(html_mapping_file.read_text())
        scraper_mapping = json.loads(scraper_mapping_file.read_text())
        assert html_mapping[INVESTOR_GOV_STOCKS_URL]["status"] == "success"
        assert scraper_mapping[INVESTOR_GOV_STOCKS_URL]["status"] == "success"

        txt_files = list((tmp_path / "scraper_cache" / "investor.gov").glob("*.txt"))
        assert len(txt_files) >= 1

        txt_content = txt_files[0].read_text(encoding="utf-8")
        assert len(txt_content) > 0
        assert not re.search(r"<\s*/?\w", txt_content), "Extracted text should not contain HTML tags"

        lower = txt_content.lower()
        assert any(word in lower for word in ("stock", "share", "invest", "equity")), (
            "Extracted text should contain financial content"
        )

    def test_scrape_action_skips_cached_download(self, tmp_path):
        run(cache_path=tmp_path, action="download", urls=[INVESTOR_GOV_STOCKS_URL])

        html_files = list((tmp_path / "html_cache" / "investor.gov").glob("*.html"))
        assert len(html_files) >= 1
        html_file = html_files[0]
        mtime_before = html_file.stat().st_mtime

        time.sleep(1.0)

        run(cache_path=tmp_path, action="scrape", urls=[INVESTOR_GOV_STOCKS_URL], force_refresh=False)

        mtime_after = html_file.stat().st_mtime
        assert mtime_after == mtime_before, (
            "HTML file should not be re-downloaded when already cached and force_refresh=False"
        )

    def test_force_refresh_redownloads_content(self, tmp_path):
        run(cache_path=tmp_path, action="download", urls=[INVESTOR_GOV_STOCKS_URL])

        html_files = list((tmp_path / "html_cache" / "investor.gov").glob("*.html"))
        assert len(html_files) >= 1
        html_file = html_files[0]
        mtime_before = html_file.stat().st_mtime

        time.sleep(1.0)

        run(cache_path=tmp_path, action="download", urls=[INVESTOR_GOV_STOCKS_URL], force_refresh=True)

        mtime_after = html_file.stat().st_mtime
        assert mtime_after > mtime_before, (
            "HTML file modification time should change when force_refresh=True"
        )
