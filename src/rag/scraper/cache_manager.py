import json
from pathlib import Path
from urllib.parse import urlparse

from src.rag.guid_registry import GuidRegistry


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
        self._guid_registry = GuidRegistry(self.base_path)

    def load_html_mapping(self) -> dict:
        return self._load(self.html_mapping_file)

    def save_html_mapping(self, mapping: dict) -> None:
        self._save(self.html_mapping_file, mapping)

    def load_scraper_mapping(self) -> dict:
        return self._load(self.scraper_mapping_file)

    def save_scraper_mapping(self, mapping: dict) -> None:
        self._save(self.scraper_mapping_file, mapping)

    def get_cache_filepath(self, url: str, cache_dir: Path | str) -> Path:
        guid = self._guid_registry.get_or_create_guid(url)
        ext = ".html" if "html" in str(Path(cache_dir).name) else ".txt"
        return Path(cache_dir) / f"{guid}{ext}"

    def get_guid(self, url: str) -> str:
        return self._guid_registry.get_or_create_guid(url)

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
