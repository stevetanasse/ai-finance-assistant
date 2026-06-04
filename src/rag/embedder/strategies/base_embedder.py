from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into dense vectors.
        Returns one float vector per input text.
        Raises ValueError if texts list is empty.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for similarity search."""

    @abstractmethod
    def embed_sparse(self, texts: list[str]) -> list[dict]:
        """Generate sparse embeddings for a list of texts.
        Returns [{"indices": [...], "values": [...]}, ...].
        Raises ValueError if texts list is empty.
        """

    @abstractmethod
    def embed_sparse_query(self, text: str) -> dict:
        """Generate sparse embedding for a single query.
        Returns {"indices": [...], "values": [...]}.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short identifier used in collection names and mapping metadata."""

    @property
    @abstractmethod
    def sparse_model_name(self) -> str | None:
        """Short identifier for the sparse model, or None if not supported."""

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Dimensionality of the dense output vectors. Returns 0 for sparse-only."""
