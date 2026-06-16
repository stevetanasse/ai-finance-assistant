from pathlib import Path

import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from src.rag.embedder.embedding_cache_manager import EmbeddingCacheManager
from src.rag.embedder.qdrant_manager import QdrantManager
from src.rag.embedder.strategies.fastembed_embedder import FastEmbedEmbedder
from src.rag.embedder.strategies.sparse_embedder import BM42Embedder

from .nodes import (
    make_financial_concepts_node,
    make_realtime_quotes_node,
    make_router_node,
    make_synchronizer_node,
)
from .states import AgentState

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def route_after_router(state: AgentState) -> str | list[Send]:
    """Determine the next step(s) after the router node.

    Returns "synchronizer" directly for out-of-scope requests, or one Send
    per route for fan-out to financial_concepts and/or realtime_quotes. Each
    Send carries the full current state, since the target nodes only read
    `messages` and `call_counts`.
    """
    routes = state["route"]
    if routes == ["out_of_scope"]:
        return "synchronizer"
    return [Send(route, state) for route in routes]


def build_graph(llm, collection_name: str | None = None):
    router_node = make_router_node(llm)
    realtime_quotes_node = make_realtime_quotes_node(llm)
    synchronizer_node = make_synchronizer_node()

    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    chunking_cfg = config["chunking"]
    embedding_cfg = config["embedding"]

    qdrant_manager = QdrantManager()
    dense_embedder = FastEmbedEmbedder(embedding_cfg["default_model"])
    sparse_embedder = BM42Embedder()
    if collection_name is None:
        # TODO: update — domain removed from collection naming convention (see embedding_cache_manager.py).
        # source_domain arg is now ignored; collection name format is fin_c{size}_o{overlap}_{dense}_{sparse}
        collection_name = EmbeddingCacheManager().make_collection_name(
            embedding_cfg["source_domain"],
            chunking_cfg["default_chunk_size"],
            chunking_cfg["default_chunk_overlap"],
            dense_embedder.model_name,
            sparse_embedder.model_name,
        )
    financial_concepts_node = make_financial_concepts_node(
        llm, qdrant_manager, dense_embedder, sparse_embedder, collection_name
    )

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("router", router_node)
    graph_builder.add_node("financial_concepts", financial_concepts_node)
    graph_builder.add_node("realtime_quotes", realtime_quotes_node)
    graph_builder.add_node("synchronizer", synchronizer_node)
    graph_builder.set_entry_point("router")
    graph_builder.add_conditional_edges("router", route_after_router)
    graph_builder.add_edge("financial_concepts", "synchronizer")
    graph_builder.add_edge("realtime_quotes", "synchronizer")
    graph_builder.add_edge("synchronizer", END)

    return graph_builder.compile(checkpointer=MemorySaver())
