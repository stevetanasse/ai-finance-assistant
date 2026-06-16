import json
import re
import time
from pathlib import Path

import pytest

from src.rag.pipeline.rag_pipeline import run
from src.rag.chunker.chunk_cache_manager import ChunkCacheManager
from src.rag.embedder.embedding_cache_manager import EmbeddingCacheManager
from src.rag.embedder.qdrant_manager import QdrantManager

INVESTOR_GOV_STOCKS_URL = (
    "https://www.investor.gov/introduction-investing/"
    "investing-basics/investment-products/stocks"
)

DENSE_MODEL = "bge-small-en-v1.5"
SPARSE_MODEL = "bm42"

# FastEmbedEmbedder/BM42Embedder normalize the constructor model name to a
# short_name (see MODEL_CONFIGS) — that short_name is what ends up in
# collection names, embedding_mapping entries, and composite cache keys.
DENSE_SHORT_NAME = "bge-small"
SPARSE_SHORT_NAME = "bm42"


@pytest.mark.integration
class TestRagPipelineE2E:
    def test_download_action_creates_html_cache(self, tmp_path):
        code = run(
            cache_path=tmp_path,
            action="download",
            urls=[INVESTOR_GOV_STOCKS_URL],
        )

        assert code == 0
        assert (tmp_path / "html_cache").is_dir()

        html_files = list((tmp_path / "html_cache").glob("*.html"))
        assert len(html_files) >= 1

        mapping_file = tmp_path / "html_cache" / "html_cache_mapping.json"
        assert mapping_file.exists()
        mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
        assert mapping[INVESTOR_GOV_STOCKS_URL]["status"] == "success"

        html_file = html_files[0]
        assert html_file.stat().st_size > 0
        content = html_file.read_text(encoding="utf-8", errors="replace").lower()
        assert "<html" in content or "<!doctype" in content

    def test_scrape_action_creates_both_caches(self, tmp_path):
        code = run(
            cache_path=tmp_path,
            action="scrape",
            urls=[INVESTOR_GOV_STOCKS_URL],
        )

        assert code == 0
        assert (tmp_path / "html_cache").is_dir()
        assert (tmp_path / "scraper_cache").is_dir()

        html_mapping_file = tmp_path / "html_cache" / "html_cache_mapping.json"
        scraper_mapping_file = tmp_path / "scraper_cache" / "scraper_cache_mapping.json"
        assert html_mapping_file.exists()
        assert scraper_mapping_file.exists()

        html_mapping = json.loads(html_mapping_file.read_text(encoding="utf-8"))
        scraper_mapping = json.loads(scraper_mapping_file.read_text(encoding="utf-8"))
        assert html_mapping[INVESTOR_GOV_STOCKS_URL]["status"] == "success"
        assert scraper_mapping[INVESTOR_GOV_STOCKS_URL]["status"] == "success"

        txt_files = list((tmp_path / "scraper_cache").glob("*.txt"))
        assert len(txt_files) >= 1

        txt_content = txt_files[0].read_text(encoding="utf-8")
        assert len(txt_content) > 0
        assert not re.search(r"<\s*/?\w", txt_content), "Extracted text should not contain HTML tags"

        lower = txt_content.lower()
        assert any(word in lower for word in ("stock", "share", "invest", "equity")), (
            "Extracted text should contain financial content"
        )

    def test_scrape_action_skips_cached_download(self, tmp_path):
        run(cache_path=tmp_path, action="download", urls=[INVESTOR_GOV_STOCKS_URL])

        html_files = list((tmp_path / "html_cache").glob("*.html"))
        assert len(html_files) >= 1
        html_file = html_files[0]
        mtime_before = html_file.stat().st_mtime

        time.sleep(1.0)

        run(cache_path=tmp_path, action="scrape", urls=[INVESTOR_GOV_STOCKS_URL], force_refresh=False)

        mtime_after = html_file.stat().st_mtime
        assert mtime_after == mtime_before, (
            "HTML file should not be re-downloaded when already cached and force_refresh=False"
        )

    def test_force_refresh_redownloads_content(self, tmp_path):
        run(cache_path=tmp_path, action="download", urls=[INVESTOR_GOV_STOCKS_URL])

        html_files = list((tmp_path / "html_cache").glob("*.html"))
        assert len(html_files) >= 1
        html_file = html_files[0]
        mtime_before = html_file.stat().st_mtime

        time.sleep(1.0)

        run(cache_path=tmp_path, action="download", urls=[INVESTOR_GOV_STOCKS_URL], force_refresh=True)

        mtime_after = html_file.stat().st_mtime
        assert mtime_after > mtime_before, (
            "HTML file modification time should change when force_refresh=True"
        )

    def test_chunk_action_creates_all_three_caches(self, tmp_path):
        code = run(
            cache_path=tmp_path,
            action="chunk",
            urls=[INVESTOR_GOV_STOCKS_URL],
            chunk_size=500,
            chunk_overlap=50,
        )

        assert code == 0
        assert (tmp_path / "html_cache").is_dir()
        assert (tmp_path / "scraper_cache").is_dir()
        assert (tmp_path / "chunk_cache").is_dir()

        jsonl_files = list((tmp_path / "chunk_cache").glob("*.jsonl"))
        assert len(jsonl_files) >= 1

        mapping_file = tmp_path / "chunk_cache" / "chunk_cache_mapping.json"
        assert mapping_file.exists()

        ccm = ChunkCacheManager(base_path=tmp_path)
        composite_key = ccm.make_cache_key(INVESTOR_GOV_STOCKS_URL, 500, 50)
        mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
        assert composite_key in mapping
        entry = mapping[composite_key]
        assert entry["status"] == "success"
        assert entry["total_chunks"] >= 1

        jsonl_file = jsonl_files[0]
        assert jsonl_file.stat().st_size > 0
        lines = [l for l in jsonl_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) >= 1

        required_fields = {"chunk_id", "url", "text", "chunk_index", "total_chunks",
                           "chunk_size", "chunk_overlap"}
        for line in lines:
            obj = json.loads(line)
            assert required_fields.issubset(obj.keys()), f"Missing fields: {required_fields - obj.keys()}"
            assert obj["chunk_size"] == 500
            assert obj["chunk_overlap"] == 50
            assert len(obj["text"]) > 0, "text field must not be empty"

    def test_two_chunk_runs_create_separate_cache_entries(self, tmp_path):
        code1 = run(
            cache_path=tmp_path,
            action="chunk",
            urls=[INVESTOR_GOV_STOCKS_URL],
            chunk_size=500,
            chunk_overlap=50,
        )
        code2 = run(
            cache_path=tmp_path,
            action="chunk",
            urls=[INVESTOR_GOV_STOCKS_URL],
            chunk_size=1000,
            chunk_overlap=100,
        )

        assert code1 == 0
        assert code2 == 0

        ccm = ChunkCacheManager(base_path=tmp_path)
        key_500 = ccm.make_cache_key(INVESTOR_GOV_STOCKS_URL, 500, 50)
        key_1000 = ccm.make_cache_key(INVESTOR_GOV_STOCKS_URL, 1000, 100)

        chunk_mapping = json.loads((tmp_path / "chunk_cache" / "chunk_cache_mapping.json").read_text(encoding="utf-8"))
        assert key_500 in chunk_mapping
        assert key_1000 in chunk_mapping
        assert chunk_mapping[key_500]["status"] == "success"
        assert chunk_mapping[key_1000]["status"] == "success"

        jsonl_files = list((tmp_path / "chunk_cache").glob("*.jsonl"))
        assert len(jsonl_files) == 2
        names = {f.name for f in jsonl_files}
        assert any("_c500_o50" in n for n in names)
        assert any("_c1000_o100" in n for n in names)

        file_500 = next(f for f in jsonl_files if "_c500_o50" in f.name)
        file_1000 = next(f for f in jsonl_files if "_c1000_o100" in f.name)
        lines_500 = [l for l in file_500.read_text(encoding="utf-8").splitlines() if l.strip()]
        lines_1000 = [l for l in file_1000.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines_500) > 0
        assert len(lines_1000) > 0
        assert len(lines_1000) < len(lines_500), "Larger chunks should produce fewer chunks"

        html_mapping = json.loads((tmp_path / "html_cache" / "html_cache_mapping.json").read_text(encoding="utf-8"))
        assert len(html_mapping) == 1, "html_mapping should have exactly one entry (cache reuse)"

        scraper_mapping = json.loads((tmp_path / "scraper_cache" / "scraper_cache_mapping.json").read_text(encoding="utf-8"))
        assert len(scraper_mapping) == 1, "scraper_mapping should have exactly one entry (cache reuse)"

    def test_embed_action_creates_qdrant_collection(self, tmp_path):
        code = run(
            cache_path=tmp_path,
            action="embed",
            urls=[INVESTOR_GOV_STOCKS_URL],
            chunk_size=500,
            chunk_overlap=50,
            dense_embed=DENSE_MODEL,
            sparse_embed=SPARSE_MODEL,
        )

        assert code == 0
        assert list((tmp_path / "html_cache").glob("*.html"))
        assert list((tmp_path / "scraper_cache").glob("*.txt"))
        assert list((tmp_path / "chunk_cache").glob("*.jsonl"))
        assert (tmp_path / "qdrant_storage").is_dir()

        embedding_mapping_file = tmp_path / "embedding_cache_mapping.json"
        assert embedding_mapping_file.exists()
        embedding_mapping = json.loads(embedding_mapping_file.read_text(encoding="utf-8"))

        ccm = ChunkCacheManager(base_path=tmp_path)
        chunk_key = ccm.make_cache_key(INVESTOR_GOV_STOCKS_URL, 500, 50)
        ecm = EmbeddingCacheManager(base_path=tmp_path)
        composite_key = ecm.make_cache_key(chunk_key, DENSE_SHORT_NAME, SPARSE_SHORT_NAME)

        assert composite_key in embedding_mapping
        entry = embedding_mapping[composite_key]
        assert entry["status"] == "success"
        assert entry["total_vectors"] >= 1
        assert entry["dense_model"] == DENSE_SHORT_NAME
        assert entry["sparse_model"] == SPARSE_SHORT_NAME

        collection_name = entry["collection_name"]
        assert collection_name == f"fin_investor_gov_c500_o50_{DENSE_SHORT_NAME}_{SPARSE_SHORT_NAME}"

        qm = QdrantManager(storage_path=tmp_path / "qdrant_storage")
        info = qm.get_collection_info(collection_name)
        assert info["has_sparse_vectors"] is True
        assert info["dense_vector_size"] == 384
        assert info["vector_count"] >= 1

    def test_embed_action_supports_dense_query(self, tmp_path):
        code = run(
            cache_path=tmp_path,
            action="embed",
            urls=[INVESTOR_GOV_STOCKS_URL],
            chunk_size=500,
            chunk_overlap=50,
            dense_embed=DENSE_MODEL,
            sparse_embed=SPARSE_MODEL,
        )
        assert code == 0

        from src.rag.embedder.embedding_pipeline import EmbeddingPipeline
        from src.rag.embedder.strategies.fastembed_embedder import FastEmbedEmbedder
        from src.rag.embedder.strategies.sparse_embedder import BM42Embedder

        ccm = ChunkCacheManager(base_path=tmp_path)
        chunk_key = ccm.make_cache_key(INVESTOR_GOV_STOCKS_URL, 500, 50)
        ecm = EmbeddingCacheManager(base_path=tmp_path)
        composite_key = ecm.make_cache_key(chunk_key, DENSE_SHORT_NAME, SPARSE_SHORT_NAME)
        embedding_mapping = json.loads(
            (tmp_path / "embedding_cache_mapping.json").read_text(encoding="utf-8")
        )
        collection_name = embedding_mapping[composite_key]["collection_name"]

        pipeline = EmbeddingPipeline(
            embedding_cache_manager=EmbeddingCacheManager(base_path=tmp_path),
            qdrant_manager=QdrantManager(storage_path=tmp_path / "qdrant_storage"),
            dense_embedder=FastEmbedEmbedder(DENSE_MODEL),
            sparse_embedder=BM42Embedder(SPARSE_MODEL),
        )
        results = pipeline.query_dense(
            collection_name=collection_name,
            query_text="What is the difference between preferred and common stock?",
            top_k=3,
        )

        assert len(results) == 3
        keywords = ("preferred", "common", "shareholder", "dividend", "voting")
        found_keyword = False
        for result in results:
            assert -1.0 <= result["score"] <= 1.0
            payload = result["payload"]
            assert payload["text"]
            assert payload["url"] == INVESTOR_GOV_STOCKS_URL
            if any(k in payload["text"].lower() for k in keywords):
                found_keyword = True
        assert found_keyword, "At least one result should contain a relevant keyword"

    def test_embed_action_supports_sparse_query(self, tmp_path):
        code = run(
            cache_path=tmp_path,
            action="embed",
            urls=[INVESTOR_GOV_STOCKS_URL],
            chunk_size=500,
            chunk_overlap=50,
            dense_embed=DENSE_MODEL,
            sparse_embed=SPARSE_MODEL,
        )
        assert code == 0

        from src.rag.embedder.embedding_pipeline import EmbeddingPipeline
        from src.rag.embedder.strategies.fastembed_embedder import FastEmbedEmbedder
        from src.rag.embedder.strategies.sparse_embedder import BM42Embedder

        ccm = ChunkCacheManager(base_path=tmp_path)
        chunk_key = ccm.make_cache_key(INVESTOR_GOV_STOCKS_URL, 500, 50)
        ecm = EmbeddingCacheManager(base_path=tmp_path)
        composite_key = ecm.make_cache_key(chunk_key, DENSE_SHORT_NAME, SPARSE_SHORT_NAME)
        embedding_mapping = json.loads(
            (tmp_path / "embedding_cache_mapping.json").read_text(encoding="utf-8")
        )
        collection_name = embedding_mapping[composite_key]["collection_name"]

        pipeline = EmbeddingPipeline(
            embedding_cache_manager=EmbeddingCacheManager(base_path=tmp_path),
            qdrant_manager=QdrantManager(storage_path=tmp_path / "qdrant_storage"),
            dense_embedder=FastEmbedEmbedder(DENSE_MODEL),
            sparse_embedder=BM42Embedder(SPARSE_MODEL),
        )
        results = pipeline.query_sparse(
            collection_name=collection_name,
            query_text="What is the difference between preferred and common stock?",
            top_k=3,
        )

        assert len(results) == 3
        keywords = ("preferred", "common", "shareholder", "dividend", "voting")
        found_keyword = False
        for result in results:
            assert result["score"] >= 0.0
            payload = result["payload"]
            assert payload["text"]
            assert payload["url"] == INVESTOR_GOV_STOCKS_URL
            if any(k in payload["text"].lower() for k in keywords):
                found_keyword = True
        assert found_keyword, "At least one result should contain a relevant keyword"

    def test_two_embed_runs_create_separate_collections(self, tmp_path):
        code1 = run(
            cache_path=tmp_path,
            action="embed",
            urls=[INVESTOR_GOV_STOCKS_URL],
            chunk_size=500,
            chunk_overlap=50,
            dense_embed=DENSE_MODEL,
            sparse_embed=SPARSE_MODEL,
        )
        code2 = run(
            cache_path=tmp_path,
            action="embed",
            urls=[INVESTOR_GOV_STOCKS_URL],
            chunk_size=1000,
            chunk_overlap=100,
            dense_embed=DENSE_MODEL,
            sparse_embed=SPARSE_MODEL,
        )

        assert code1 == 0
        assert code2 == 0

        collection_500 = f"fin_investor_gov_c500_o50_{DENSE_SHORT_NAME}_{SPARSE_SHORT_NAME}"
        collection_1000 = f"fin_investor_gov_c1000_o100_{DENSE_SHORT_NAME}_{SPARSE_SHORT_NAME}"

        qm = QdrantManager(storage_path=tmp_path / "qdrant_storage")
        info_500 = qm.get_collection_info(collection_500)
        info_1000 = qm.get_collection_info(collection_1000)

        assert info_500["has_sparse_vectors"] is True
        assert info_1000["has_sparse_vectors"] is True
        assert info_500["vector_count"] >= 1
        assert info_1000["vector_count"] >= 1
        assert info_500["vector_count"] > info_1000["vector_count"], (
            "Smaller chunk size should produce more vectors"
        )

        embedding_mapping = json.loads(
            (tmp_path / "embedding_cache_mapping.json").read_text(encoding="utf-8")
        )
        successful = [e for e in embedding_mapping.values() if e["status"] == "success"]
        assert len(successful) == 2

        chunk_mapping = json.loads(
            (tmp_path / "chunk_cache" / "chunk_cache_mapping.json").read_text(encoding="utf-8")
        )
        assert len(chunk_mapping) == 2

        html_mapping = json.loads(
            (tmp_path / "html_cache" / "html_cache_mapping.json").read_text(encoding="utf-8")
        )
        assert len(html_mapping) == 1, "html_mapping should have exactly one entry (cache reuse)"

        scraper_mapping = json.loads(
            (tmp_path / "scraper_cache" / "scraper_cache_mapping.json").read_text(encoding="utf-8")
        )
        assert len(scraper_mapping) == 1, "scraper_mapping should have exactly one entry (cache reuse)"
