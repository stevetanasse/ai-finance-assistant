import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

from .cache_manager import CacheManager

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class UrlDownloader:
    def __init__(
        self,
        cache_manager: CacheManager,
        delay_seconds: float = 1.0,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        force_refresh: bool = False,
    ):
        self.cache_manager = cache_manager
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.force_refresh = force_refresh
        self.session = requests.Session()
        self.session.headers["User-Agent"] = _USER_AGENT

    def download_url(self, url: str) -> dict:
        mapping = self.cache_manager.load_html_mapping()

        if not self.force_refresh and self.cache_manager.is_url_cached(url, mapping):
            return mapping[url]

        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]

        cache_path = self.cache_manager.get_cache_filepath(url, self.cache_manager.html_cache_dir)

        entry: dict = {
            "file_path": str(cache_path),
            "scraped_path": str(self.cache_manager.get_cache_filepath(url, self.cache_manager.scraper_cache_dir)),
            "downloaded_at": None,
            "scraped_at": None,
            "status": "failed",
            "scrape_status": "pending",
            "http_status_code": None,
            "content_type": None,
            "word_count": None,
            "source_domain": domain,
            "error_message": None,
        }

        try:
            response = self._fetch(url)
            entry["http_status_code"] = response.status_code
            entry["content_type"] = response.headers.get("Content-Type", "")
            if response.status_code == 200:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(response.text, encoding="utf-8")
                entry["status"] = "success"
                entry["downloaded_at"] = datetime.now(timezone.utc).isoformat()
            else:
                entry["error_message"] = f"HTTP {response.status_code}"
        except Exception as e:
            entry["error_message"] = str(e)

        mapping[url] = entry
        self.cache_manager.save_html_mapping(mapping)
        return entry

    def download_all(self, urls: list[str]) -> dict:
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] Downloading: {url}")
            self.download_url(url)
            if i < len(urls):
                time.sleep(self.delay_seconds)
        return self.cache_manager.load_html_mapping()

    def _fetch(self, url: str) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.session.get(url, timeout=self.timeout_seconds)
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay_seconds)
        raise last_exc  # type: ignore[misc]
