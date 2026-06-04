import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

DISTANCE_MAP = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclidean": Distance.EUCLID,
}


class QdrantManager:
    def __init__(
        self,
        storage_path: Path | str | None = None,
        in_memory: bool = False,
    ):
        if in_memory:
            self._client = QdrantClient(":memory:")
        else:
            if storage_path is None:
                storage_path = self._default_path()
            storage_path = Path(storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(storage_path))

    def _default_path(self) -> Path:
        # parents[3] = project root (src/rag/embedder/qdrant_manager.py)
        return Path(__file__).resolve().parents[3] / "rag_caches" / "qdrant_storage"

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in [
            c.name for c in self._client.get_collections().collections
        ]

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "cosine",
        recreate: bool = False,
    ) -> None:
        if self.collection_exists(collection_name):
            if recreate:
                self._client.delete_collection(collection_name)
            else:
                return
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=DISTANCE_MAP[distance],
            ),
        )

    def upsert_points(self, collection_name: str, points: list[dict]) -> None:
        self._client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=self._make_point_id(p["id"]),
                    vector=p["vector"],
                    payload=p["payload"],
                )
                for p in points
            ],
        )

    def _make_point_id(self, id_str: str) -> str:
        return str(uuid.uuid5(_NAMESPACE, str(id_str)))

    def query(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict | None = None,
    ) -> list[dict]:
        qdrant_filter = None
        if filters:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filters.items()
                ]
            )
        response = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            {"id": str(r.id), "score": r.score, "payload": r.payload}
            for r in response.points
        ]

    def get_collection_info(self, collection_name: str) -> dict:
        info = self._client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vector_count": info.points_count or 0,
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
        }

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.get_collections().collections]

    def delete_collection(self, collection_name: str) -> None:
        if self.collection_exists(collection_name):
            self._client.delete_collection(collection_name)
