import json
from datetime import datetime, timezone
from pathlib import Path

from .embedding_cache_manager import EmbeddingCacheManager
from .qdrant_manager import QdrantManager
from .strategies.base_embedder import BaseEmbedder
from .strategies.fastembed_embedder import FastEmbedEmbedder
from .strategies.sparse_embedder import BM42Embedder

DENSE_EMBEDDER_REGISTRY = {
    "bge-small": FastEmbedEmbedder,
    "bge-small-en-v1.5": FastEmbedEmbedder,
}

SPARSE_EMBEDDER_REGISTRY = {
    "bm42": BM42Embedder,
}

BATCH_SIZE = 32


class EmbeddingPipeline:
    def __init__(
        self,
        embedding_cache_manager: EmbeddingCacheManager,
        qdrant_manager: QdrantManager,
        dense_embedder: BaseEmbedder,
        sparse_embedder: BaseEmbedder,
    ):
        self.cache_manager = embedding_cache_manager
        self.qdrant = qdrant_manager
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder

    def embed_chunk_file(
        self,
        chunk_key: str,
        chunk_entry: dict,
        embedding_mapping: dict,
        force_refresh: bool = False,
    ) -> dict:
        cache_key = self.cache_manager.make_cache_key(
            chunk_key, self.dense_embedder.model_name, self.sparse_embedder.model_name
        )

        if not force_refresh and self.cache_manager.is_embedded(
            chunk_key, self.dense_embedder.model_name, embedding_mapping,
            self.sparse_embedder.model_name
        ):
            return embedding_mapping[cache_key]

        source_domain = chunk_entry.get("source_domain", "")
        chunk_size = chunk_entry.get("chunk_size", 0)
        chunk_overlap = chunk_entry.get("chunk_overlap", 0)
        collection_name = self.cache_manager.make_collection_name(
            source_domain, chunk_size, chunk_overlap,
            self.dense_embedder.model_name, self.sparse_embedder.model_name
        )

        entry: dict = {
            "url": chunk_entry.get("url", ""),
            "collection_name": collection_name,
            "chunk_cache_key": chunk_key,
            "chunk_file_path": chunk_entry.get("file_path", ""),
            "dense_model": self.dense_embedder.model_name,
            "sparse_model": self.sparse_embedder.model_name,
            "vector_size": self.dense_embedder.vector_size,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_vectors": 0,
            "embedded_at": None,
            "status": "failed",
            "error_message": None,
        }

        try:
            self.qdrant.create_collection(
                collection_name,
                dense_vector_size=self.dense_embedder.vector_size,
                recreate=force_refresh,
            )

            chunk_file = Path(chunk_entry["file_path"])
            chunks = [
                json.loads(line)
                for line in chunk_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            texts = [c["text"] for c in chunks]
            dense_vecs: list[list[float]] = []
            sparse_vecs: list[dict] = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                dense_vecs.extend(self.dense_embedder.embed(batch))
                sparse_vecs.extend(self.sparse_embedder.embed_sparse(batch))

            points = [
                {
                    "id": chunk["chunk_id"],
                    "dense_vector": dvec,
                    "sparse_vector": svec,
                    "payload": {
                        **chunk,
                        "dense_model": self.dense_embedder.model_name,
                        "sparse_model": self.sparse_embedder.model_name,
                    },
                }
                for chunk, dvec, svec in zip(chunks, dense_vecs, sparse_vecs)
            ]
            self.qdrant.upsert_points(collection_name, points)

            entry["status"] = "success"
            entry["total_vectors"] = len(points)
            entry["embedded_at"] = datetime.now(timezone.utc).isoformat()
            entry["error_message"] = None
        except Exception as e:
            entry["error_message"] = str(e)

        return entry

    def embed_all(
        self,
        chunk_mapping: dict,
        force_refresh: bool = False,
    ) -> dict:
        embedding_mapping = self.cache_manager.load_mapping()

        if force_refresh:
            pending = [k for k, v in chunk_mapping.items() if v.get("status") == "success"]
        else:
            pending = self.cache_manager.get_pending_chunk_keys(
                chunk_mapping, embedding_mapping,
                self.dense_embedder.model_name, self.sparse_embedder.model_name
            )

        total = len(pending)
        for i, chunk_key in enumerate(pending, 1):
            chunk_entry = chunk_mapping[chunk_key]
            collection_name = self.cache_manager.make_collection_name(
                chunk_entry.get("source_domain", ""),
                chunk_entry.get("chunk_size", 0),
                chunk_entry.get("chunk_overlap", 0),
                self.dense_embedder.model_name,
                self.sparse_embedder.model_name,
            )
            print(f"[{i}/{total}] Embedding: {collection_name}")

            entry = self.embed_chunk_file(
                chunk_key, chunk_entry, embedding_mapping, force_refresh=force_refresh
            )
            cache_key = self.cache_manager.make_cache_key(
                chunk_key, self.dense_embedder.model_name, self.sparse_embedder.model_name
            )
            embedding_mapping[cache_key] = entry
            self.cache_manager.save_mapping(embedding_mapping)

        return embedding_mapping

    def query_dense(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 3,
        filters: dict | None = None,
    ) -> list[dict]:
        query_vec = self.dense_embedder.embed_query(query_text)
        return self.qdrant.query_dense(collection_name, query_vec, top_k, filters)

    def query_sparse(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 3,
        filters: dict | None = None,
    ) -> list[dict]:
        query_sv = self.sparse_embedder.embed_sparse_query(query_text)
        return self.qdrant.query_sparse(collection_name, query_sv, top_k, filters)
