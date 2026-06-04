import json
import pytest
from pathlib import Path

from src.rag.embedder.embedding_cache_manager import EmbeddingCacheManager
from src.rag.embedder.embedding_pipeline import EmbeddingPipeline
from src.rag.embedder.qdrant_manager import QdrantManager
from src.rag.embedder.strategies.fastembed_embedder import FastEmbedEmbedder

STOCKS_URL = (
    "https://www.investor.gov/introduction-investing/"
    "investing-basics/investment-products/stocks"
)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
QUERY = "What is the difference between preferred and common stock?"


@pytest.mark.integration
class TestEmbeddingE2E:
    @pytest.fixture
    def populated_qdrant(self, tmp_path):
        """Run full download → scrape → chunk → embed pipeline."""
        from src.rag.scraper.cache_manager import CacheManager
        from src.rag.scraper.url_downloader import UrlDownloader
        from src.rag.scraper.html_scraper import HtmlScraper
        from src.rag.chunker.chunk_cache_manager import ChunkCacheManager
        from src.rag.chunker.chunker import Chunker

        cm = CacheManager(base_path=tmp_path)
        dl = UrlDownloader(cache_manager=cm, delay_seconds=1.0)
        html_mapping = dl.download_all([STOCKS_URL])

        scraper = HtmlScraper(cache_manager=cm)
        scraper_mapping = scraper.scrape_all()

        ccm = ChunkCacheManager(base_path=tmp_path)
        chunker = Chunker(
            chunk_cache_manager=ccm,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunk_mapping = chunker.chunk_all(scraper_mapping)

        embedder = FastEmbedEmbedder("bge-small-en-v1.5")
        qdrant = QdrantManager(in_memory=True)
        ecm = EmbeddingCacheManager(base_path=tmp_path)
        pipeline = EmbeddingPipeline(
            embedding_cache_manager=ecm,
            qdrant_manager=qdrant,
            embedder=embedder,
        )
        pipeline.embed_all(chunk_mapping)

        collection_name = ecm.make_collection_name(
            "investor.gov", CHUNK_SIZE, CHUNK_OVERLAP, embedder.model_name
        )
        return pipeline, ecm, qdrant, collection_name, chunk_mapping

    def test_pipeline_populates_qdrant_collection(self, populated_qdrant):
        pipeline, ecm, qdrant, collection_name, _ = populated_qdrant
        assert qdrant.collection_exists(collection_name)
        info = qdrant.get_collection_info(collection_name)
        assert info["vector_count"] >= 1
        assert info["vector_size"] == 384
        assert collection_name == ecm.make_collection_name(
            "investor.gov", CHUNK_SIZE, CHUNK_OVERLAP, "bge-small"
        )

    def test_query_returns_top_3_results(self, populated_qdrant):
        pipeline, _, _, collection_name, _ = populated_qdrant
        results = pipeline.query_collection(collection_name, QUERY, top_k=3)
        assert len(results) == 3
        for r in results:
            assert "id" in r
            assert "score" in r
            assert "payload" in r
            assert -1.0 <= r["score"] <= 1.0
            assert len(r["payload"]["text"]) > 0
            assert r["payload"]["url"] == STOCKS_URL
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_results_are_relevant(self, populated_qdrant):
        pipeline, _, _, collection_name, _ = populated_qdrant
        results = pipeline.query_collection(collection_name, QUERY, top_k=3)
        relevant_terms = {"preferred", "common stock", "shareholder", "dividend", "voting"}
        found = any(
            any(term in r["payload"]["text"].lower() for term in relevant_terms)
            for r in results
        )
        assert found, (
            "Expected at least one top-3 result to contain a relevant financial term. "
            "Check that chunk text is stored in payload and embeddings are working."
        )

    def test_embedding_cache_mapping_is_populated(self, populated_qdrant, tmp_path):
        pipeline, ecm, _, collection_name, chunk_mapping = populated_qdrant
        emb_mapping = ecm.load_mapping()

        chunk_key = f"{STOCKS_URL}|c{CHUNK_SIZE}|o{CHUNK_OVERLAP}"
        emb_key = ecm.make_cache_key(chunk_key, "bge-small")
        assert emb_key in emb_mapping

        entry = emb_mapping[emb_key]
        assert entry["status"] == "success"
        assert entry["total_vectors"] >= 1
        assert entry["collection_name"] == collection_name
        assert entry["embedding_model"] == "bge-small"
