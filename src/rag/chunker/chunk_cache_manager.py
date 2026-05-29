import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from slugify import slugify


class ChunkCacheManager:
    def __init__(self, base_path: Path | str | None = None):
        if base_path is None:
            # parents[0]=chunker/, [1]=rag/, [2]=src/, [3]=project root
            base_path = Path(__file__).resolve().parents[3] / "rag_caches"
        self.base_path = Path(base_path)
        self.chunk_cache_dir = self.base_path / "chunk_cache"
        self.mapping_file = self.chunk_cache_dir / "chunk_cache_mapping.json"
        self.chunk_cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Key / path helpers
    # ------------------------------------------------------------------

    def make_cache_key(self, url: str, chunk_size: int, chunk_overlap: int) -> str:
        return f"{url}|c{chunk_size}|o{chunk_overlap}"

    def get_chunk_filepath(self, url: str, chunk_size: int, chunk_overlap: int) -> Path:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        path_parts = [p for p in parsed.path.split("/") if p]
        slug = slugify(path_parts[-1]) if path_parts else "index"
        domain_dir = self.chunk_cache_dir / domain
        domain_dir.mkdir(exist_ok=True)
        return domain_dir / f"{slug}_c{chunk_size}_o{chunk_overlap}.jsonl"

    # ------------------------------------------------------------------
    # Cache status
    # ------------------------------------------------------------------

    def is_cached(self, url: str, chunk_size: int, chunk_overlap: int, mapping: dict) -> bool:
        key = self.make_cache_key(url, chunk_size, chunk_overlap)
        return key in mapping and mapping[key].get("status") == "success"

    def get_pending_urls(
        self,
        mapping: dict,
        scraper_mapping: dict,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        result = []
        for url, entry in scraper_mapping.items():
            if entry.get("status") != "success":
                continue
            if self.is_cached(url, chunk_size, chunk_overlap, mapping):
                continue
            result.append({
                "url": url,
                "scraped_path": entry.get("file_path", ""),
                "source_domain": entry.get("source_domain", ""),
            })
        return result

    # ------------------------------------------------------------------
    # JSONL I/O
    # ------------------------------------------------------------------

    def write_chunks(
        self,
        chunks: list[str],
        url: str,
        chunk_size: int,
        chunk_overlap: int,
        source_domain: str,
        scraped_path: str,
        strategy: str,
    ) -> Path:
        filepath = self.get_chunk_filepath(url, chunk_size, chunk_overlap)
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split("/") if p]
        slug = slugify(path_parts[-1]) if path_parts else "index"

        total = len(chunks)
        now = datetime.now(timezone.utc).isoformat()

        lines = []
        for i, text in enumerate(chunks):
            obj = {
                "chunk_id": f"{source_domain}_{slug}_{i:04d}",
                "url": url,
                "source_domain": source_domain,
                "chunk_index": i,
                "total_chunks": total,
                "text": text,
                "char_count": len(text),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "strategy": strategy,
                "chunked_at": now,
            }
            lines.append(json.dumps(obj, ensure_ascii=False))

        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return filepath

    def read_chunks(self, file_path: Path | str) -> list[dict]:
        text = Path(file_path).read_text(encoding="utf-8")
        result = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON in chunk file: {e}") from e
        return result

    # ------------------------------------------------------------------
    # Mapping I/O
    # ------------------------------------------------------------------

    def load_mapping(self) -> dict:
        if not self.mapping_file.exists():
            return {}
        return json.loads(self.mapping_file.read_text(encoding="utf-8"))

    def save_mapping(self, mapping: dict) -> None:
        self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
        self.mapping_file.write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8"
        )
