from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base_chunker import BaseChunker


class RecursiveChunker(BaseChunker):
    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(self, chunk_size: int, chunk_overlap: int):
        super().__init__(chunk_size, chunk_overlap)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.SEPARATORS,
            length_function=len,
        )

    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return self._splitter.split_text(text)

    @property
    def strategy_name(self) -> str:
        return "recursive"
