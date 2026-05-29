from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from .cache_manager import CacheManager
from .extractors.investor_gov import InvestorGovExtractor


class HtmlScraper:
    EXTRACTOR_REGISTRY = {
        "investor.gov": InvestorGovExtractor(),
    }

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager

    def scrape_url(self, url: str, html_mapping: dict) -> dict:
        entry = html_mapping[url].copy()
        try:
            html = Path(entry["file_path"]).read_text(encoding="utf-8")
            soup = BeautifulSoup(html, "lxml")

            extractor = self.EXTRACTOR_REGISTRY.get(entry.get("source_domain", ""))
            if extractor:
                text = extractor.extract(soup)
            else:
                text = self._generic_extract(soup)

            html_path = Path(entry["file_path"])
            scraper_path = (
                self.cache_manager.scraper_cache_dir
                / html_path.relative_to(self.cache_manager.html_cache_dir).with_suffix(".txt")
            )
            scraper_path.parent.mkdir(parents=True, exist_ok=True)
            scraper_path.write_text(text, encoding="utf-8")

            entry["scrape_status"] = "success"
            entry["scraped_at"] = datetime.now(timezone.utc).isoformat()
            entry["word_count"] = len(text.split())
            entry["scraped_path"] = str(scraper_path)
        except Exception as e:
            entry["scrape_status"] = "failed"
            entry["error_message"] = str(e)

        return entry

    def scrape_all(self, force_refresh: bool = False) -> dict:
        html_mapping = self.cache_manager.load_html_mapping()
        scraper_mapping = self.cache_manager.load_scraper_mapping()

        urls = [
            url for url, entry in html_mapping.items()
            if entry.get("status") == "success"
            and (force_refresh or entry.get("scrape_status") == "pending")
        ]

        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] Scraping: {url}")
            entry = self.scrape_url(url, html_mapping)
            html_mapping[url] = entry
            scraper_mapping[url] = {
                "file_path": entry.get("scraped_path"),
                "scraped_at": entry.get("scraped_at"),
                "status": entry.get("scrape_status"),
                "word_count": entry.get("word_count"),
                "source_domain": entry.get("source_domain"),
                "error_message": entry.get("error_message"),
            }
            self.cache_manager.save_html_mapping(html_mapping)
            self.cache_manager.save_scraper_mapping(scraper_mapping)

        return scraper_mapping

    def _generic_extract(self, soup: BeautifulSoup) -> str:
        return soup.get_text(separator="\n").strip()
