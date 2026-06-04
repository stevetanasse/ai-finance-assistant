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

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short identifier used in collection names and mapping metadata."""

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Dimensionality of the output vectors."""
