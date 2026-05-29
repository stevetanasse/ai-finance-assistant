import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

from src.rag.scraper.cache_manager import CacheManager
from src.rag.scraper.html_scraper import HtmlScraper
from src.rag.scraper.url_downloader import UrlDownloader


def resolve_urls(url_path: str) -> list[str]:
    """
    Accepts a single URL string or a file path string.
    Returns a list of URL strings.
    Raises FileNotFoundError if url_path is not a URL and the file does not exist.
    Raises ValueError if the resolved list is empty.
    """
    parsed = urlparse(url_path)
    if parsed.scheme in ("http", "https"):
        return [url_path]

    path = Path(url_path)
    if not path.exists():
        raise FileNotFoundError(f"URL file not found: {url_path}")

    urls = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not urls:
        raise ValueError(f"No URLs found in {url_path}")

    return urls


def validate_args(
    action: str,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> str | None:
    """
    Validates cross-argument constraints.
    Returns an error message string if validation fails, None if valid.
    """
    if action != "chunk":
        return None
    if chunk_size is None:
        return "Error: --chunk-size is required when --action is 'chunk'"
    if chunk_overlap is None:
        return "Error: --chunk-overlap is required when --action is 'chunk'"
    if chunk_size is not None and chunk_overlap is not None and chunk_overlap >= chunk_size:
        return "Error: --chunk-overlap must be less than --chunk-size"
    if chunk_size is not None and chunk_size <= 0:
        return "Error: --chunk-size must be greater than 0"
    if chunk_overlap is not None and chunk_overlap < 0:
        return "Error: --chunk-overlap must be non-negative"
    return None


def run(
    cache_path: Path,
    action: str,
    urls: list[str],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    force_refresh: bool = False,
    verbose: bool = False,
) -> int:
    """
    Orchestrates the pipeline.
    Returns 0 on success, 1 on any failure.
    Never raises exceptions to the caller.
    """
    try:
        cm = CacheManager(base_path=cache_path)

        if verbose:
            print(f"[rag_pipeline] Action: {action}")
            print(f"[rag_pipeline] URLs to process: {len(urls)}")
            print(f"[rag_pipeline] Cache path: {cm.base_path}")

        dl = UrlDownloader(cache_manager=cm, force_refresh=force_refresh)
        html_mapping = dl.download_all(urls)

        scraper_mapping: dict = {}
        if action in ("scrape", "chunk"):
            scraper = HtmlScraper(cache_manager=cm)
            scraper_mapping = scraper.scrape_all(force_refresh=force_refresh)

        chunk_mapping: dict = {}
        if action == "chunk":
            from src.rag.chunker import Chunker, ChunkCacheManager  # lazy import
            chunk_cache_mgr = ChunkCacheManager(base_path=cache_path)
            chunker = Chunker(
                chunk_cache_manager=chunk_cache_mgr,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunk_mapping = chunker.chunk_all(
                scraper_mapping=scraper_mapping,
                force_refresh=force_refresh,
            )

        downloaded = sum(1 for e in html_mapping.values() if e.get("status") == "success")
        failed = sum(1 for e in html_mapping.values() if e.get("status") == "failed")
        scraped = sum(1 for e in scraper_mapping.values() if e.get("status") == "success")
        chunked = sum(1 for e in chunk_mapping.values() if e.get("status") == "success")
        total_chunks = sum(
            e.get("total_chunks", 0)
            for e in chunk_mapping.values()
            if e.get("status") == "success"
        )

        if action == "download":
            print(f"[rag_pipeline] Done. Downloaded: {downloaded}, Failed: {failed}")
        elif action == "scrape":
            print(f"[rag_pipeline] Done. Downloaded: {downloaded}, Scraped: {scraped}, Failed: {failed}")
        else:
            print(
                f"[rag_pipeline] Done. Downloaded: {downloaded}, Scraped: {scraped}, "
                f"Chunked: {chunked}, Total chunks: {total_chunks}, Failed: {failed}"
            )

        return 0
    except Exception as e:
        print(f"[rag_pipeline] Error: {e}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_pipeline",
        description="Orchestrate the RAG download and scrape pipeline.",
    )
    parser.add_argument(
        "--cache-path",
        required=True,
        help="Path to the parent folder that will contain rag_caches/.",
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["download", "scrape", "chunk"],
        help="Pipeline action: 'download', 'scrape', or 'chunk'.",
    )
    parser.add_argument(
        "--url-path",
        required=True,
        help="A single URL or a path to a file containing URLs (one per line).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Number of characters per chunk. Required when --action is 'chunk'.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Number of overlapping characters between chunks. Required when --action is 'chunk'.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Re-download and re-scrape URLs even if already cached.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed progress to stdout.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    err = validate_args(args.action, args.chunk_size, args.chunk_overlap)
    if err:
        print(err, file=sys.stderr)
        sys.exit(1)

    try:
        urls = resolve_urls(args.url_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[rag_pipeline] Error: {e}", file=sys.stderr)
        sys.exit(1)

    code = run(
        cache_path=Path(args.cache_path),
        action=args.action,
        urls=urls,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_refresh=args.force_refresh,
        verbose=args.verbose,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
