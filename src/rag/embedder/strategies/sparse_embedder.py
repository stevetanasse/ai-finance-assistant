from fastembed import SparseTextEmbedding

from .base_embedder import BaseEmbedder


class BM42Embedder(BaseEmbedder):
    """Sparse embedder using Qdrant's BM42 model via FastEmbed."""

    MODEL_CONFIGS = {
        "bm42": {
            "fastembed_model_name": "Qdrant/bm42-all-minilm-l6-v2-attentions",
            "short_name": "bm42",
        },
    }

    def __init__(self, model_name: str = "bm42"):
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(self.MODEL_CONFIGS.keys())}"
            )
        config = self.MODEL_CONFIGS[model_name]
        self._model_name = model_name
        self._short_name = config["short_name"]
        self._model = SparseTextEmbedding(model_name=config["fastembed_model_name"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "BM42Embedder does not support dense embeddings. Use FastEmbedEmbedder for dense."
        )

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError(
            "BM42Embedder does not support dense embeddings. Use FastEmbedEmbedder for dense."
        )

    def embed_sparse(self, texts: list[str]) -> list[dict]:
        if not texts:
            raise ValueError("texts list must not be empty")
        return [
            {"indices": r.indices.tolist(), "values": r.values.tolist()}
            for r in self._model.embed(texts)
        ]

    def embed_sparse_query(self, text: str) -> dict:
        result = list(self._model.query_embed(text))[0]
        return {"indices": result.indices.tolist(), "values": result.values.tolist()}

    @property
    def model_name(self) -> str:
        return self._short_name

    @property
    def sparse_model_name(self) -> str:
        return self._short_name

    @property
    def vector_size(self) -> int:
        return 0
