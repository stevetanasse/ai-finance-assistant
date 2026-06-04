import json
from pathlib import Path


class EmbeddingCacheManager:
    def __init__(self, base_path: Path | str | None = None):
        if base_path is None:
            # parents[3] = project root (src/rag/embedder/embedding_cache_manager.py)
            base_path = Path(__file__).resolve().parents[3] / "rag_caches"
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.mapping_file = self.base_path / "embedding_cache_mapping.json"

    def make_cache_key(
        self, chunk_key: str, dense_model_name: str, sparse_model_name: str = "bm42"
    ) -> str:
        return f"{chunk_key}|{dense_model_name}|{sparse_model_name}"

    def make_collection_name(
        self,
        source_domain: str,
        chunk_size: int,
        chunk_overlap: int,
        dense_model_name: str,
        sparse_model_name: str = "bm42",
    ) -> str:
        domain_slug = source_domain.replace(".", "_")
        name = (
            f"fin_{domain_slug}_c{chunk_size}_o{chunk_overlap}"
            f"_{dense_model_name}_{sparse_model_name}"
        )
        return name[:63]

    def load_mapping(self) -> dict:
        if not self.mapping_file.exists():
            return {}
        return json.loads(self.mapping_file.read_text(encoding="utf-8"))

    def save_mapping(self, mapping: dict) -> None:
        self.mapping_file.write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def is_embedded(
        self,
        chunk_key: str,
        dense_model_name: str,
        mapping: dict,
        sparse_model_name: str = "bm42",
    ) -> bool:
        key = self.make_cache_key(chunk_key, dense_model_name, sparse_model_name)
        return key in mapping and mapping[key].get("status") == "success"

    def get_pending_chunk_keys(
        self,
        chunk_mapping: dict,
        embedding_mapping: dict,
        dense_model_name: str,
        sparse_model_name: str = "bm42",
    ) -> list[str]:
        return [
            key for key, entry in chunk_mapping.items()
            if entry.get("status") == "success"
            and not self.is_embedded(
                key, dense_model_name, embedding_mapping, sparse_model_name
            )
        ]
