from abc import ABC, abstractmethod


class BaseChunker(ABC):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split text into a list of chunk strings.
        Returns an empty list if text is empty or whitespace-only.
        Never raises exceptions for empty input.
        """

    @property
    def strategy_name(self) -> str:
        """Short identifier for this strategy used in cache filenames and mapping metadata."""
        raise NotImplementedError
