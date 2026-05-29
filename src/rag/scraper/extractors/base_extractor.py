import re
import unicodedata
from abc import ABC, abstractmethod

from bs4 import BeautifulSoup


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, soup: BeautifulSoup) -> str:
        """Extract clean article text from a BeautifulSoup object."""

    def clean_text(self, text: str) -> str:
        """Normalize whitespace, collapse blank lines, strip unicode."""
        text = unicodedata.normalize("NFKC", text)
        lines = [line.rstrip() for line in text.splitlines()]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
