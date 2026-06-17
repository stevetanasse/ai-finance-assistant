from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .hybrid_search import hybrid_search


class HybridQdrantRetriever(BaseRetriever):
    """Wraps hybrid_search() as a LangChain retriever so retrieval appears as its
    own span in LangSmith traces, instead of disappearing inside a bare function call."""

    # Typed Any (not the concrete classes) because callers — including existing
    # tests — pass plain MagicMock() instances with no spec for the embedders;
    # a concrete-class annotation would fail Pydantic's isinstance validation.
    qdrant_manager: Any
    dense_embedder: Any
    sparse_embedder: Any
    collection_name: str
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = hybrid_search(
            self.qdrant_manager,
            self.dense_embedder,
            self.sparse_embedder,
            self.collection_name,
            query,
            top_k=self.top_k,
        )
        return [
            Document(
                page_content=r["text"],
                metadata={"source_url": r["source_url"], "score": r["score"]},
            )
            for r in results
        ]
