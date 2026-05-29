import json
from pathlib import Path
from urllib.parse import urlparse

from slugify import slugify


class CacheManager:
    def __init__(self, base_path: Path | str | None = None):
        if base_path is None:
            # parents[3]: src/rag/scraper/ → src/rag/ → src/ → project root
            base_path = Path(__file__).resolve().parents[3] / "rag_caches"
        self.base_path = Path(base_path)
        self.html_cache_dir = self.base_path / "html_cache"
        self.scraper_cache_dir = self.base_path / "scraper_cache"
        self.html_mapping_file = self.html_cache_dir / "html_cache_mapping.json"
        self.scraper_mapping_file = self.scraper_cache_dir / "scraper_cache_mapping.json"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.html_cache_dir.mkdir(exist_ok=True)
        self.scraper_cache_dir.mkdir(exist_ok=True)

    def load_html_mapping(self) -> dict:
        return self._load(self.html_mapping_file)

    def save_html_mapping(self, mapping: dict) -> None:
        self._save(self.html_mapping_file, mapping)

    def load_scraper_mapping(self) -> dict:
        return self._load(self.scraper_mapping_file)

    def save_scraper_mapping(self, mapping: dict) -> None:
        self._save(self.scraper_mapping_file, mapping)

    def get_cache_filepath(self, url: str, cache_dir: Path | str) -> Path:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        path_parts = [p for p in parsed.path.split("/") if p]
        relevant = path_parts[-2:] if path_parts else ["index"]
        slug = "_".join(slugify(p) for p in relevant) or "index"
        ext = ".html" if "html" in str(Path(cache_dir).name) else ".txt"
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
