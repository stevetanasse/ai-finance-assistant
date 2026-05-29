import json
from pathlib import Path
from urllib.parse import urlparse

from slugify import slugify

HTML_MAPPING_PATH = Path("html_cache") / "html_cache_mapping.json"
SCRAPER_MAPPING_PATH = Path("scraper_cache") / "scraper_cache_mapping.json"


class CacheManager:
    def load_html_mapping(self) -> dict:
        return self._load(HTML_MAPPING_PATH)

    def save_html_mapping(self, mapping: dict) -> None:
        self._save(HTML_MAPPING_PATH, mapping)

    def load_scraper_mapping(self) -> dict:
        return self._load(SCRAPER_MAPPING_PATH)

    def save_scraper_mapping(self, mapping: dict) -> None:
        self._save(SCRAPER_MAPPING_PATH, mapping)

    def get_cache_filepath(self, url: str, cache_dir: str) -> Path:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        path_parts = [p for p in parsed.path.split("/") if p]
        relevant = path_parts[-2:] if path_parts else ["index"]
        slug = "_".join(slugify(p) for p in relevant) or "index"
        ext = ".html" if "html" in str(cache_dir) else ".txt"
        return Path(cache_dir) / domain / f"{slug}{ext}"

    def is_url_cached(self, url: str, mapping: dict) -> bool:
        return url in mapping and mapping[url].get("status") == "success"

    def get_failed_urls(self, mapping: dict) -> list[str]:
        return [url for url, entry in mapping.items() if entry.get("status") == "failed"]

    def get_pending_scrape_urls(self, mapping: dict) -> list[str]:
        return [
            url for url, entry in mapping.items()
            if entry.get("status") == "success" and entry.get("scrape_status") == "pending"
        ]

    def _load(self, path: Path) -> dict:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
