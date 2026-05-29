import pytest

from src.rag.chunker.strategies.recursive_chunker import RecursiveChunker

FINANCIAL_TEXT = """\
Stocks represent ownership in a corporation and constitute a claim on part of the
corporation's assets and earnings. There are two main types of stock: common and
preferred. Common stock usually entitles the owner to vote at shareholder meetings
and to receive dividends. Preferred stockholders generally do not have voting rights,
though they have a higher claim on assets and earnings than common stockholders.

Bonds are debt instruments issued by corporations and governments to raise capital.
When you buy a bond, you are lending money to the issuer. In return, the issuer
promises to pay you a specified rate of interest during the life of the bond and to
repay the face value of the bond when it matures. Government bonds are considered
low-risk investments, while corporate bonds carry higher risk but typically offer
higher yields to compensate investors for the additional risk.

Mutual funds pool money from many investors to purchase a diversified portfolio of
stocks, bonds, or other securities. Professional fund managers make investment
decisions on behalf of the fund's investors. Index funds are a type of mutual fund
that aims to replicate the performance of a specific market index, such as the
S&P 500. Exchange-traded funds (ETFs) are similar to index funds but trade on
stock exchanges like individual stocks, offering intraday liquidity.

Diversification is a risk management strategy that mixes a wide variety of
investments within a portfolio. The rationale behind this technique contends that
a portfolio constructed of different kinds of assets will, on average, yield higher
long-term returns and lower the risk of any individual holding or security.
"""


@pytest.fixture
def chunker():
    return RecursiveChunker(chunk_size=500, chunk_overlap=50)


# ---------------------------------------------------------------------------
# split() — basic behaviour
# ---------------------------------------------------------------------------

def test_split_returns_nonempty_list_for_normal_text(chunker):
    result = chunker.split("This is a sentence about stocks and bonds.")
    assert isinstance(result, list)
    assert len(result) > 0

def test_split_returns_empty_list_for_empty_string(chunker):
    assert chunker.split("") == []

def test_split_returns_empty_list_for_whitespace_only_string(chunker):
    assert chunker.split("   \n\n   \t  ") == []

def test_split_respects_chunk_size(chunker):
    result = chunker.split(FINANCIAL_TEXT)
    tolerance = 1.1
    for chunk in result:
        assert len(chunk) <= chunker.chunk_size * tolerance, (
            f"Chunk length {len(chunk)} exceeds {chunker.chunk_size * tolerance}"
        )

def test_split_produces_overlapping_chunks():
    # Use a single long paragraph with no natural separators to force mid-text overlap
    words = ["investment"] * 80
    dense_text = " ".join(words)  # ~880 chars, no \n separators
    c = RecursiveChunker(chunk_size=200, chunk_overlap=40)
    result = c.split(dense_text)
    assert len(result) >= 2
    # Last 40 chars of chunk[0] should appear somewhere in chunk[1]
    tail = result[0][-40:]
    assert tail in result[1], "Expected overlap: tail of chunk[0] should appear in chunk[1]"

def test_strategy_name_returns_recursive(chunker):
    assert chunker.strategy_name == "recursive"


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_constructor_raises_when_overlap_equals_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)

def test_constructor_raises_when_overlap_exceeds_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveChunker(chunk_size=100, chunk_overlap=200)

def test_constructor_raises_when_chunk_size_is_zero():
    with pytest.raises(ValueError, match="chunk_size"):
        RecursiveChunker(chunk_size=0, chunk_overlap=0)

def test_constructor_raises_when_chunk_size_is_negative():
    with pytest.raises(ValueError, match="chunk_size"):
        RecursiveChunker(chunk_size=-10, chunk_overlap=0)

def test_constructor_raises_when_chunk_overlap_is_negative():
    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveChunker(chunk_size=100, chunk_overlap=-1)


# ---------------------------------------------------------------------------
# Realistic chunking test
# ---------------------------------------------------------------------------

def test_financial_text_produces_at_least_three_chunks():
    c = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    result = c.split(FINANCIAL_TEXT)
    assert len(result) >= 3

def test_all_chunks_are_nonempty_strings():
    c = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    result = c.split(FINANCIAL_TEXT)
    for chunk in result:
        assert isinstance(chunk, str)
        assert len(chunk) > 0
