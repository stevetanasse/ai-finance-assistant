import pytest
from unittest.mock import MagicMock, patch

_MODULE = "src.rag.embedder.strategies.sparse_embedder"


def _fake_sparse_emb(indices=None, values=None):
    if indices is None:
        indices = [0, 1, 5]
    if values is None:
        values = [0.5, 0.3, 0.8]
    m = MagicMock()
    m.indices = MagicMock()
    m.indices.tolist.return_value = indices
    m.values = MagicMock()
    m.values.tolist.return_value = values
    return m


@pytest.fixture
def mock_ste():
    with patch(f"{_MODULE}.SparseTextEmbedding") as MockSTE:
        MockSTE.return_value.embed.return_value = iter([_fake_sparse_emb()])
        MockSTE.return_value.query_embed.return_value = iter([_fake_sparse_emb()])
        yield MockSTE


@pytest.fixture
def embedder(mock_ste):
    from src.rag.embedder.strategies.sparse_embedder import BM42Embedder
    return BM42Embedder("bm42")


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_unknown_model_raises_value_error():
    with patch(f"{_MODULE}.SparseTextEmbedding"):
        from src.rag.embedder.strategies.sparse_embedder import BM42Embedder
        with pytest.raises(ValueError, match="Unknown model"):
            BM42Embedder("unknown-model")


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_model_name_returns_bm42(embedder):
    assert embedder.model_name == "bm42"

def test_sparse_model_name_returns_bm42(embedder):
    assert embedder.sparse_model_name == "bm42"

def test_vector_size_returns_0(embedder):
    assert embedder.vector_size == 0


# ---------------------------------------------------------------------------
# Dense methods raise NotImplementedError
# ---------------------------------------------------------------------------

def test_embed_raises_not_implemented(embedder):
    with pytest.raises(NotImplementedError):
        embedder.embed(["text"])

def test_embed_query_raises_not_implemented(embedder):
    with pytest.raises(NotImplementedError):
        embedder.embed_query("text")


# ---------------------------------------------------------------------------
# embed_sparse()
# ---------------------------------------------------------------------------

def test_embed_sparse_raises_value_error_for_empty_list(embedder):
    with pytest.raises(ValueError, match="empty"):
        embedder.embed_sparse([])

def test_embed_sparse_returns_list_of_correct_length(mock_ste, embedder):
    mock_ste.return_value.embed.return_value = iter(
        [_fake_sparse_emb(), _fake_sparse_emb()]
    )
    result = embedder.embed_sparse(["text one", "text two"])
    assert len(result) == 2

def test_embed_sparse_each_result_has_indices_and_values(mock_ste, embedder):
    mock_ste.return_value.embed.return_value = iter([_fake_sparse_emb()])
    result = embedder.embed_sparse(["hello"])
    assert "indices" in result[0]
    assert "values" in result[0]

def test_embed_sparse_indices_is_list_of_ints(mock_ste, embedder):
    mock_ste.return_value.embed.return_value = iter([_fake_sparse_emb([0, 1, 2], [0.5, 0.3, 0.2])])
    result = embedder.embed_sparse(["hello"])
    assert all(isinstance(i, int) for i in result[0]["indices"])

def test_embed_sparse_values_is_list_of_floats(mock_ste, embedder):
    mock_ste.return_value.embed.return_value = iter([_fake_sparse_emb([0, 1], [0.5, 0.3])])
    result = embedder.embed_sparse(["hello"])
    assert all(isinstance(v, float) for v in result[0]["values"])

def test_embed_sparse_indices_and_values_same_length(mock_ste, embedder):
    mock_ste.return_value.embed.return_value = iter([_fake_sparse_emb([0, 1, 2], [0.5, 0.3, 0.2])])
    result = embedder.embed_sparse(["hello"])
    assert len(result[0]["indices"]) == len(result[0]["values"])

def test_embed_sparse_calls_model_with_correct_texts(mock_ste, embedder):
    mock_ste.return_value.embed.return_value = iter([_fake_sparse_emb()])
    embedder.embed_sparse(["test text"])
    mock_ste.return_value.embed.assert_called_once_with(["test text"])


# ---------------------------------------------------------------------------
# embed_sparse_query()
# ---------------------------------------------------------------------------

def test_embed_sparse_query_returns_dict_with_required_keys(mock_ste, embedder):
    mock_ste.return_value.query_embed.return_value = iter([_fake_sparse_emb([0, 1], [0.5, 0.3])])
    result = embedder.embed_sparse_query("what are stocks?")
    assert "indices" in result
    assert "values" in result
    assert isinstance(result["indices"], list)
    assert isinstance(result["values"], list)
