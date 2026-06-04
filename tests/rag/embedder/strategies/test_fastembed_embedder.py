import pytest
from unittest.mock import MagicMock, patch

_MODULE = "src.rag.embedder.strategies.fastembed_embedder"


def _fake_embedding(dim: int = 384):
    m = MagicMock()
    m.tolist.return_value = [0.1] * dim
    return m


@pytest.fixture
def mock_te():
    with patch(f"{_MODULE}.TextEmbedding") as MockTE:
        MockTE.return_value.embed.return_value = iter([_fake_embedding()])
        yield MockTE


@pytest.fixture
def embedder(mock_te):
    from src.rag.embedder.strategies.fastembed_embedder import FastEmbedEmbedder
    return FastEmbedEmbedder("bge-small-en-v1.5")


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_unknown_model_raises_value_error():
    with patch(f"{_MODULE}.TextEmbedding"):
        from src.rag.embedder.strategies.fastembed_embedder import FastEmbedEmbedder
        with pytest.raises(ValueError, match="Unknown model"):
            FastEmbedEmbedder("unknown-model")


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_model_name_returns_short_name(embedder):
    assert embedder.model_name == "bge-small"

def test_vector_size_returns_384(embedder):
    assert embedder.vector_size == 384


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------

def test_embed_raises_value_error_for_empty_list(embedder):
    with pytest.raises(ValueError, match="empty"):
        embedder.embed([])

def test_embed_returns_list_of_correct_length(mock_te, embedder):
    mock_te.return_value.embed.return_value = iter([_fake_embedding(), _fake_embedding()])
    result = embedder.embed(["text one", "text two"])
    assert len(result) == 2

def test_embed_each_vector_has_length_384(mock_te, embedder):
    mock_te.return_value.embed.return_value = iter([_fake_embedding(384)])
    result = embedder.embed(["hello"])
    assert len(result[0]) == 384

def test_embed_each_vector_element_is_float(mock_te, embedder):
    mock_te.return_value.embed.return_value = iter([_fake_embedding(384)])
    result = embedder.embed(["hello"])
    assert all(isinstance(v, float) for v in result[0])

def test_embed_calls_underlying_model_with_texts(mock_te, embedder):
    mock_te.return_value.embed.return_value = iter([_fake_embedding()])
    embedder.embed(["test text"])
    mock_te.return_value.embed.assert_called_once_with(["test text"])


# ---------------------------------------------------------------------------
# embed_query()
# ---------------------------------------------------------------------------

def test_embed_query_returns_single_vector_of_length_384(mock_te, embedder):
    mock_te.return_value.embed.return_value = iter([_fake_embedding(384)])
    result = embedder.embed_query("what are stocks?")
    assert isinstance(result, list)
    assert len(result) == 384
