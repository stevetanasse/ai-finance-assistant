from fastembed import TextEmbedding

from .base_embedder import BaseEmbedder


class FastEmbedEmbedder(BaseEmbedder):
    MODEL_CONFIGS = {
        "bge-small-en-v1.5": {
            "fastembed_model_name": "BAAI/bge-small-en-v1.5",
            "vector_size": 384,
            "short_name": "bge-small",
        },
    }

    def __init__(self, model_name: str = "bge-small-en-v1.5"):
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(self.MODEL_CONFIGS.keys())}"
            )
        config = self.MODEL_CONFIGS[model_name]
        self._model_name = model_name
        self._short_name = config["short_name"]
        self._vector_size = config["vector_size"]
        self._model = TextEmbedding(model_name=config["fastembed_model_name"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts list must not be empty")
        return [e.tolist() for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._model.embed([text]))[0].tolist()

    @property
    def model_name(self) -> str:
        return self._short_name

    @property
    def vector_size(self) -> int:
        return self._vector_size
