from pathlib import Path

import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.rag.embedder.embedding_cache_manager import EmbeddingCacheManager
from src.rag.embedder.qdrant_manager import QdrantManager
from src.rag.embedder.strategies.fastembed_embedder import FastEmbedEmbedder
from src.rag.embedder.strategies.sparse_embedder import BM42Embedder

from .nodes import make_financial_concepts_node, make_realtime_quotes_node, make_router_node
from .states import AgentState

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def route_after_router(state: AgentState) -> str:
    classification = state["messages"][-1].content
    if classification == "financial_concepts_node":
        return "financial_concepts"
    if classification == "realtime_quotes_node":
        return "realtime_quotes"
    return END


def build_graph(llm):
    router_node = make_router_node(llm)
    realtime_quotes_node = make_realtime_quotes_node(llm)

    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    chunking_cfg = config["chunking"]
    embedding_cfg = config["embedding"]

    qdrant_manager = QdrantManager()
    dense_embedder = FastEmbedEmbedder(embedding_cfg["default_model"])
    sparse_embedder = BM42Embedder()
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
    graph_builder.set_entry_point("router")
    graph_builder.add_conditional_edges(
        "router",
        route_after_router,
        {"financial_concepts": "financial_concepts", "realtime_quotes": "realtime_quotes", END: END},
    )
    graph_builder.add_edge("financial_concepts", END)
    graph_builder.add_edge("realtime_quotes", END)

    return graph_builder.compile(checkpointer=MemorySaver())
