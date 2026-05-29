import pytest
from bs4 import BeautifulSoup

from src.rag.scraper.extractors.investor_gov import InvestorGovExtractor

FULL_HTML = """\
<html>
<head>
  <title>Bonds | Investor.gov</title>
  <meta name="description" content="Learn about bonds and fixed income products.">
</head>
<body>
  <nav class="breadcrumb-nav">
    <a href="/">Home</a> / <a href="/introduction-investing">Investing Basics</a> / Bonds
  </nav>
  <article>
    <h1>Bonds</h1>
    <p>A bond is a debt security, similar to an IOU. When you purchase a bond,
    you are lending money to a government, municipality, or corporation.</p>
    <section>
      <h2>Types of Bonds</h2>
      <ul>
        <li>Government bonds</li>
        <li>Corporate bonds</li>
        <li>Municipal bonds</li>
      </ul>
    </section>
    <nav class="in-page-nav">Jump to: Types | Risks | Buying</nav>
    <div class="related">
      <h3>Related Topics</h3>
      <ul><li><a href="/stocks">Stocks</a></li></ul>
    </div>
  </article>
  <aside class="sidebar">
    <h3>Investor Resources</h3>
    <p>Visit SEC.gov for more information.</p>
  </aside>
  <footer>
    <p>U.S. Securities and Exchange Commission</p>
  </footer>
</body>
</html>
"""


@pytest.fixture
def extractor():
    return InvestorGovExtractor()


@pytest.fixture
def soup():
    return BeautifulSoup(FULL_HTML, "lxml")


# ---------------------------------------------------------------------------
# extract — container selection
# ---------------------------------------------------------------------------

def test_extract_uses_article_tag(extractor, soup):
    text = extractor.extract(soup)
    assert "A bond is a debt security" in text

def test_extract_falls_back_to_main_when_no_article(extractor):
    html = """\
    <html><body>
      <main>
        <h1>Stocks</h1>
        <p>A stock is ownership in a company.</p>
      </main>
      <aside class="sidebar">Sidebar content</aside>
    </body></html>"""
    text = extractor.extract(BeautifulSoup(html, "lxml"))
    assert "A stock is ownership" in text
    assert "Sidebar content" not in text

def test_extract_falls_back_to_div_content_when_no_article_or_main(extractor):
    html = """\
    <html><body>
      <div class="content"><p>Fallback content here.</p></div>
      <div class="sidebar">Ignore this.</div>
    </body></html>"""
    text = extractor.extract(BeautifulSoup(html, "lxml"))
    assert "Fallback content here" in text


# ---------------------------------------------------------------------------
# extract — noise removal
# ---------------------------------------------------------------------------

def test_extract_removes_nav_elements(extractor, soup):
    text = extractor.extract(soup)
    assert "Jump to: Types" not in text

def test_extract_removes_elements_with_related_class(extractor, soup):
    text = extractor.extract(soup)
    assert "Related Topics" not in text

def test_extract_removes_elements_with_sidebar_class(extractor):
    html = """\
    <html><body><article>
      <p>Main financial content.</p>
      <div class="sidebar">Sidebar noise.</div>
    </article></body></html>"""
    text = extractor.extract(BeautifulSoup(html, "lxml"))
    assert "Sidebar noise" not in text
    assert "Main financial content" in text

def test_extract_removes_elements_with_advertisement_class(extractor):
    html = """\
    <html><body><article>
      <p>Bond information here.</p>
      <div class="advertisement">Sponsored content</div>
    </article></body></html>"""
    text = extractor.extract(BeautifulSoup(html, "lxml"))
    assert "Sponsored content" not in text
    assert "Bond information here" in text

def test_extract_removes_elements_with_breadcrumb_class(extractor):
    html = """\
    <html><body><article>
      <div class="breadcrumb">Home / Products / Bonds</div>
      <p>Important content.</p>
    </article></body></html>"""
    text = extractor.extract(BeautifulSoup(html, "lxml"))
    assert "Home / Products / Bonds" not in text
    assert "Important content" in text

def test_extract_returns_nonempty_string_for_realistic_html(extractor, soup):
    text = extractor.extract(soup)
    assert len(text) > 50

def test_extract_preserves_main_article_content(extractor, soup):
    text = extractor.extract(soup)
    assert "Types of Bonds" in text
    assert "Government bonds" in text


# ---------------------------------------------------------------------------
# get_title
# ---------------------------------------------------------------------------

def test_get_title_extracts_h1(extractor, soup):
    assert extractor.get_title(soup) == "Bonds"

def test_get_title_falls_back_to_title_tag(extractor):
    html = "<html><head><title>Page Title | Investor.gov</title></head><body></body></html>"
    assert extractor.get_title(BeautifulSoup(html, "lxml")) == "Page Title | Investor.gov"

def test_get_title_returns_empty_string_when_no_title(extractor):
    html = "<html><head></head><body></body></html>"
    assert extractor.get_title(BeautifulSoup(html, "lxml")) == ""


# ---------------------------------------------------------------------------
# get_metadata
# ---------------------------------------------------------------------------

def test_get_metadata_extracts_description(extractor, soup):
    meta = extractor.get_metadata(soup)
    assert meta["description"] == "Learn about bonds and fixed income products."

def test_get_metadata_returns_empty_description_when_no_meta(extractor):
    html = "<html><head></head><body><h1>Title</h1></body></html>"
    meta = extractor.get_metadata(BeautifulSoup(html, "lxml"))
    assert meta["description"] == ""

def test_get_metadata_domain_is_investor_gov(extractor, soup):
    assert extractor.get_metadata(soup)["domain"] == "investor.gov"

def test_get_metadata_includes_title(extractor, soup):
    meta = extractor.get_metadata(soup)
    assert meta["title"] == "Bonds"


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

def test_clean_text_collapses_multiple_blank_lines(extractor):
    text = "Line 1\n\n\n\n\nLine 2"
    result = extractor.clean_text(text)
    assert "\n\n\n" not in result
    assert "Line 1" in result
    assert "Line 2" in result

def test_clean_text_strips_leading_and_trailing_whitespace(extractor):
    text = "  \n\n  Hello World  \n\n  "
    result = extractor.clean_text(text)
    assert result == result.strip()

def test_clean_text_strips_trailing_whitespace_from_lines(extractor):
    text = "Line 1   \nLine 2   "
    result = extractor.clean_text(text)
    for line in result.splitlines():
        assert line == line.rstrip()
