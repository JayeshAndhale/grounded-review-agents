"""Retrieval tool the research agent calls.

Wraps Phase 2's VectorStore.search() as a LangChain tool: a function with
a Pydantic input schema and a docstring the LLM reads as the tool's
description. The schema means whatever arguments the model emits get
validated before they ever touch the vector store - malformed tool-call
arguments fail here, not three lines into a Chroma query.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from grounded_review.config import get_settings
from grounded_review.retrieval.vector_store import PaperStore

# Module-level singleton: one Chroma connection for the process. Same
# reasoning as the embedder in Phase 2 - loading it isn't free, don't repeat it.
_store = PaperStore()


class SearchPapersInput(BaseModel):
    query: str = Field(description="Natural-language search query for the sub-topic being researched")
    arxiv_id: str | None = Field(
        default=None,
        description="Restrict the search to one paper's arXiv ID. Leave unset to search across all ingested papers.",
    )
    section: str | None = Field(
        default=None,
        description="Restrict to a section name (e.g. 'results', 'method') if relevant. Leave unset otherwise.",
    )


@tool("search_papers", args_schema=SearchPapersInput)
def search_papers(query: str, arxiv_id: str | None = None, section: str | None = None) -> list[dict]:
    """Semantic search over ingested paper chunks.

    Returns each match's chunk_id, arxiv_id, section, text, and similarity
    score. chunk_id is required for every downstream citation - never drop
    it when summarising a result.
    """
    settings = get_settings()
    results = _store.search(
        query=query,
        top_k=settings.retrieval_top_k,
        arxiv_id=arxiv_id,
        sections=[section] if section else None,
    )
    return [
        {
            "chunk_id": r.chunk.chunk_id,
            "arxiv_id": r.chunk.arxiv_id,
            "section": r.chunk.section,
            "text": r.chunk.text,
            "score": round(r.score, 3),
        }
        for r in results
    ]