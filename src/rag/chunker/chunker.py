from datetime import datetime, timezone
from pathlib import Path

from .chunk_cache_manager import ChunkCacheManager
from .strategies.recursive_chunker import RecursiveChunker

STRATEGY_REGISTRY = {
    "recursive": RecursiveChunker,
}


class Chunker:
    def __init__(
        self,
        chunk_cache_manager: ChunkCacheManager,
        chunk_size: int,
        chunk_overlap: int,
        strategy: str = "recursive",
    ):
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Available: {list(STRATEGY_REGISTRY.keys())}"
            )
        self.cache_manager = chunk_cache_manager
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy_name = strategy
        self._chunker = STRATEGY_REGISTRY[strategy](chunk_size, chunk_overlap)

    def chunk_url(
        self,
        url: str,
        scraped_text: str,
        source_domain: str,
        scraped_path: str,
        mapping: dict,
        force_refresh: bool = False,
    ) -> dict:
        key = self.cache_manager.make_cache_key(url, self.chunk_size, self.chunk_overlap)

        if not force_refresh and self.cache_manager.is_cached(url, self.chunk_size, self.chunk_overlap, mapping):
            return mapping[key]

        entry: dict = {
            "url": url,
            "file_path": None,
            "chunked_at": None,
            "status": "failed",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "strategy": self.strategy_name,
            "total_chunks": 0,
            "source_domain": source_domain,
            "scraper_cache_path": scraped_path,
            "error_message": None,
        }

        try:
            chunks = self._chunker.split(scraped_text)

            if not chunks:
                entry["error_message"] = "No chunks produced from input text"
                return entry

            filepath = self.cache_manager.write_chunks(
                chunks=chunks,
                url=url,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                source_domain=source_domain,
                scraped_path=scraped_path,
                strategy=self.strategy_name,
            )

            entry["status"] = "success"
            entry["file_path"] = str(filepath)
            entry["chunked_at"] = datetime.now(timezone.utc).isoformat()
            entry["total_chunks"] = len(chunks)
            entry["error_message"] = None
        except Exception as e:
            entry["error_message"] = str(e)

        return entry

    def chunk_all(
        self,
        scraper_mapping: dict,
        force_refresh: bool = False,
    ) -> dict:
        mapping = self.cache_manager.load_mapping()

        if force_refresh:
            pending = [
                {
                    "url": url,
                    "scraped_path": entry.get("file_path", ""),
                    "source_domain": entry.get("source_domain", ""),
                }
                for url, entry in scraper_mapping.items()
                if entry.get("status") == "success"
            ]
        else:
            pending = self.cache_manager.get_pending_urls(
                mapping, scraper_mapping, self.chunk_size, self.chunk_overlap
            )

        total = len(pending)
        for i, item in enumerate(pending, 1):
            url = item["url"]
            print(f"[{i}/{total}] Chunking: {url}")

            scraped_file = Path(item["scraped_path"])
            if not scraped_file.exists():
                key = self.cache_manager.make_cache_key(url, self.chunk_size, self.chunk_overlap)
                mapping[key] = {
                    "url": url,
                    "file_path": None,
                    "chunked_at": None,
                    "status": "failed",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "strategy": self.strategy_name,
                    "total_chunks": 0,
                    "source_domain": item["source_domain"],
                    "scraper_cache_path": item["scraped_path"],
                    "error_message": "Scraped file not found",
                }
                self.cache_manager.save_mapping(mapping)
                continue

            scraped_text = scraped_file.read_text(encoding="utf-8")
            entry = self.chunk_url(
                url=url,
                scraped_text=scraped_text,
                source_domain=item["source_domain"],
                scraped_path=item["scraped_path"],
                mapping=mapping,
                force_refresh=force_refresh,
            )
            key = self.cache_manager.make_cache_key(url, self.chunk_size, self.chunk_overlap)
            mapping[key] = entry
            self.cache_manager.save_mapping(mapping)

        return mapping
