from bs4 import BeautifulSoup, Tag

from .base_extractor import BaseExtractor

_NOISE_CLASSES = frozenset({
    "menu", "nav", "footer", "header", "sidebar", "breadcrumb",
    "social", "share", "related", "advertisement", "cookie", "banner",
})

_BLOCK_TAGS = ("nav", "header", "footer", "aside")


def _has_noise_class(tag: Tag) -> bool:
    classes = tag.get("class") or []
    return any(noise in cls for cls in classes for noise in _NOISE_CLASSES)


class InvestorGovExtractor(BaseExtractor):
    def extract(self, soup: BeautifulSoup) -> str:
        container = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_="content")
            or soup.find("body")
        )
        if container is None:
            return ""

        for tag_name in _BLOCK_TAGS:
            for tag in list(container.find_all(tag_name)):
                tag.decompose()

        noise_tags = [tag for tag in container.find_all(True) if _has_noise_class(tag)]
        for tag in noise_tags:
            try:
                tag.decompose()
            except Exception:
                pass

        return self.clean_text(container.get_text(separator="\n"))

    def get_title(self, soup: BeautifulSoup) -> str:
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        title = soup.find("title")
        if title:
            return title.get_text(strip=True)
        return ""

    def get_metadata(self, soup: BeautifulSoup) -> dict:
        description = ""
        meta = soup.find("meta", attrs={"name": "description"})
        if meta:
            description = meta.get("content", "")
        return {
            "title": self.get_title(soup),
            "description": description,
            "domain": "investor.gov",
        }
