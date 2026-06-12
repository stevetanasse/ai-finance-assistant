"""Hybrid (dense + sparse) retrieval over a Qdrant collection using Reciprocal Rank Fusion."""

from .qdrant_manager import QdrantManager
from .strategies.base_embedder import BaseEmbedder


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Fuse multiple ranked result lists into a single ranking via Reciprocal Rank Fusion.

    Each input list is expected to be a sequence of dicts with an "id" key, already
    sorted best-first (as returned by ``QdrantManager.query_dense``/``query_sparse``).
    For every item, ``1 / (k + rank)`` (1-indexed rank) is added to a running score per
    "id"; items appearing in multiple lists accumulate contributions from each. The
    constant ``k`` dampens the influence of any single ranking so that low-ranked items
    don't dominate the fused order.

    Returns the deduplicated items (first-seen payload kept) sorted by fused score
    descending, with the fused score attached under the "rrf_score" key.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            items.setdefault(item_id, item)

    ranked_ids = sorted(scores, key=lambda item_id: scores[item_id], reverse=True)
    return [{**items[item_id], "rrf_score": scores[item_id]} for item_id in ranked_ids]


def hybrid_search(
    qdrant_manager: QdrantManager,
    dense_embedder: BaseEmbedder,
    sparse_embedder: BaseEmbedder,
    collection_name: str,
    query: str,
    top_k: int = 5,
    candidate_k: int | None = None,
) -> list[dict]:
    """Run a hybrid dense + sparse search and return the top-k fused chunks.

    Embeds ``query`` with both ``dense_embedder`` and ``sparse_embedder``, retrieves
    ``candidate_k`` candidates from each of ``qdrant_manager.query_dense`` and
    ``query_sparse``, fuses them with :func:`reciprocal_rank_fusion`, and returns the
    top ``top_k`` results as dicts with "text", "source_url", and "score" keys taken
    from each result's Qdrant payload.

    ``candidate_k`` defaults to ``max(top_k * 2, 10)`` so the fusion step has a
    meaningful pool of candidates from each retrieval method to combine.
    """
    candidate_k = candidate_k or max(top_k * 2, 10)

    dense_vector = dense_embedder.embed_query(query)
    sparse_vector = sparse_embedder.embed_sparse_query(query)

    dense_results = qdrant_manager.query_dense(collection_name, dense_vector, top_k=candidate_k)
    sparse_results = qdrant_manager.query_sparse(collection_name, sparse_vector, top_k=candidate_k)

    fused = reciprocal_rank_fusion([dense_results, sparse_results])

    return [
        {
            "text": item["payload"].get("text", ""),
            "source_url": item["payload"].get("url", ""),
            "score": item["rrf_score"],
        }
        for item in fused[:top_k]
    ]
