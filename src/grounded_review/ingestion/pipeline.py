from .arxiv_client import download_pdf, fetch_metadata
from .chunker import chunk_sections
from .models import Paper
from .pdf_parser import extract_sections


def ingest(raw_id: str) -> Paper:
    """Fetch, download, parse, and chunk a single arXiv paper."""
    metadata = fetch_metadata(raw_id)
    pdf_path = download_pdf(metadata)
    sections = extract_sections(pdf_path)
    chunks = chunk_sections(metadata.arxiv_id, sections)
    return Paper(metadata=metadata, chunks=chunks)