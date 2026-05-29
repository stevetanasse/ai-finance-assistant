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


def run(
    cache_path: Path,
    action: str,
    urls: list[str],
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
        if action == "scrape":
            scraper = HtmlScraper(cache_manager=cm)
            scraper_mapping = scraper.scrape_all(force_refresh=force_refresh)

        downloaded = sum(1 for e in html_mapping.values() if e.get("status") == "success")
        failed = sum(1 for e in html_mapping.values() if e.get("status") == "failed")
        scraped = sum(1 for e in scraper_mapping.values() if e.get("status") == "success")

        print(f"[rag_pipeline] Done. Downloaded: {downloaded}, Scraped: {scraped}, Failed: {failed}")
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
        choices=["download", "scrape"],
        help="Pipeline action: 'download' or 'scrape'.",
    )
    parser.add_argument(
        "--url-path",
        required=True,
        help="A single URL or a path to a file containing URLs (one per line).",
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

    try:
        urls = resolve_urls(args.url_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[rag_pipeline] Error: {e}", file=sys.stderr)
        sys.exit(1)

    code = run(
        cache_path=Path(args.cache_path),
        action=args.action,
        urls=urls,
        force_refresh=args.force_refresh,
        verbose=args.verbose,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
