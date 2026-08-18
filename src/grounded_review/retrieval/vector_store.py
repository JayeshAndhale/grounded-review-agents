from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import get_settings
from ..ingestion.models import Chunk, Paper
from .embedder import embed_query, embed_texts


@dataclass
class RetrievedChunk:
    """A chunk returned from search, with its similarity score."""

    chunk: Chunk
    score: float

    def __repr__(self) -> str:
        return f"<{self.chunk.arxiv_id} §{self.chunk.section} score={self.score:.3f}>"


class PaperStore:
    """Chroma-backed store for paper chunks, with provenance preserved."""

    COLLECTION = "paper_chunks"

    def __init__(self, path: str | None = None):
        settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=path or settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add_paper(self, paper: Paper) -> int:
        """Index every chunk of a paper. Idempotent — re-adding overwrites."""
        if not paper.chunks:
            return 0

        texts = [c.text for c in paper.chunks]
        self._collection.upsert(
            ids=[c.chunk_id for c in paper.chunks],
            embeddings=embed_texts(texts),
            documents=texts,
            metadatas=[
                {
                    "arxiv_id": c.arxiv_id,
                    "section": c.section,
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "title": paper.metadata.title,
                }
                for c in paper.chunks
            ],
        )
        return len(paper.chunks)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        arxiv_id: str | None = None,
        sections: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Semantic search, optionally scoped to one paper or given sections."""
        settings = get_settings()

        conditions = []
        if arxiv_id:
            conditions.append({"arxiv_id": arxiv_id})
        if sections:
            conditions.append({"section": {"$in": sections}})

        where = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        results = self._collection.query(
            query_embeddings=[embed_query(query)],
            n_results=top_k or settings.retrieval_top_k,
            where=where,
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        retrieved = []
        for cid, doc, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append(
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id=cid,
                        arxiv_id=meta["arxiv_id"],
                        section=meta["section"],
                        text=doc,
                        token_count=meta["token_count"],
                        chunk_index=meta["chunk_index"],
                    ),
                    # Chroma returns cosine *distance*; convert to similarity.
                    score=1.0 - distance,
                )
            )
        return retrieved

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, str]:
        """Exact-ID lookup of chunk text, for verification against the
        original source rather than a research-note summary.

        Batches into one Chroma call rather than looping - a single
        verifier pass may need a dozen-plus chunks' text at once.
        """
        if not chunk_ids:
            return {}
        results = self._collection.get(ids=chunk_ids)
        return dict(zip(results["ids"], results["documents"]))

    def has_paper(self, arxiv_id: str) -> bool:
        return bool(self._collection.get(where={"arxiv_id": arxiv_id}, limit=1)["ids"])

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop everything. Used between evaluation runs."""
        self._client.delete_collection(self.COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )