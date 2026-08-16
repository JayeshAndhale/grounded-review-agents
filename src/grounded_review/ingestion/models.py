from datetime import date

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Bibliographic data for a single arXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: date | None = None
    pdf_url: str
    abs_url: str

    def citation(self) -> str:
        """Markdown link in first-author-et-al form, for use by the writer agent."""
        first = self.authors[0].split()[-1] if self.authors else "Unknown"
        suffix = " et al." if len(self.authors) > 1 else ""
        year = self.published.year if self.published else "n.d."
        return f"[{first}{suffix} ({year})]({self.abs_url})"


class Chunk(BaseModel):
    """One retrievable unit of a paper, with the provenance needed to cite it."""

    chunk_id: str
    arxiv_id: str
    section: str
    text: str
    token_count: int
    chunk_index: int = Field(description="Position within the paper, 0-based")


class Paper(BaseModel):
    """A fully ingested paper: metadata plus its chunked body text."""

    metadata: PaperMetadata
    chunks: list[Chunk] = []

    @property
    def total_tokens(self) -> int:
        return sum(c.token_count for c in self.chunks)