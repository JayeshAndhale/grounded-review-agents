import re
from pathlib import Path
import urllib.request

import arxiv

from .models import PaperMetadata

ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


class PaperNotFoundError(Exception):
    """Raised when an arXiv ID does not resolve to a real paper."""


def normalize_arxiv_id(raw: str) -> str:
    """Accept a URL, a bare ID, or an ID with a version suffix; return a clean ID.

    >>> normalize_arxiv_id("https://arxiv.org/abs/2211.11501v2")
    '2211.11501'
    """
    match = ARXIV_ID_PATTERN.search(raw.strip())
    if not match:
        raise ValueError(f"Could not parse an arXiv ID from: {raw!r}")
    return match.group(1)


def fetch_metadata(raw_id: str) -> PaperMetadata:
    """Look up a paper on arXiv. Raises PaperNotFoundError if it does not exist."""
    arxiv_id = normalize_arxiv_id(raw_id)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])

    try:
        result = next(client.results(search))
    except StopIteration:
        raise PaperNotFoundError(f"No arXiv paper found with ID {arxiv_id}") from None

    return PaperMetadata(
        arxiv_id=arxiv_id,
        title=result.title.strip().replace("\n", " "),
        authors=[a.name for a in result.authors],
        abstract=result.summary.strip().replace("\n", " "),
        published=result.published.date() if result.published else None,
        pdf_url=result.pdf_url,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def download_pdf(metadata: PaperMetadata, cache_dir: str = "./data/pdfs") -> Path:
    """Download the PDF, skipping the network call if we already have it.

    Fetches the URL directly rather than going through the arxiv client,
    which keeps this independent of that library's changing API.
    """
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{metadata.arxiv_id}.pdf"

    if target.exists():
        return target

    # arXiv rejects the default urllib User-Agent, so identify ourselves.
    request = urllib.request.Request(
        metadata.pdf_url,
        headers={"User-Agent": "grounded-review-agents/0.1 (research prototype)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        target.write_bytes(response.read())

    return target