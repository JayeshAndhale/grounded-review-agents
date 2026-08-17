import tiktoken

from ..config import get_settings
from .models import Chunk

_encoder = tiktoken.get_encoding("cl100k_base")

# Sections excluded from the index.
#   preamble — title block and author list; matches queries lexically via the
#              title while carrying no evidence, so it crowds out real content.
#   abstract — already stored on PaperMetadata; indexing it creates a chunk
#              that competes with the body text it summarises.
SKIP_SECTIONS = {"preamble", "abstract"}

MIN_CHUNK_TOKENS = 32


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def _split_by_tokens(text: str, size: int, overlap: int) -> list[str]:
    """Split text into token-bounded windows with overlap, snapping to
    sentence boundaries so chunks stay readable."""
    sentences = [s.strip() + " " for s in text.split(". ") if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current_tokens + sentence_tokens > size and current:
            chunks.append("".join(current).strip())
            # Carry the tail of this chunk into the next one for context.
            carry: list[str] = []
            carry_tokens = 0
            for s in reversed(current):
                t = count_tokens(s)
                if carry_tokens + t > overlap:
                    break
                carry.insert(0, s)
                carry_tokens += t
            current, current_tokens = carry, carry_tokens
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append("".join(current).strip())
    return chunks


def chunk_sections(arxiv_id: str, sections: dict[str, str]) -> list[Chunk]:
    """Turn parsed sections into retrievable chunks.

    Section boundaries are respected first; only sections longer than the
    token budget are split further, with overlap.
    """
    settings = get_settings()
    chunks: list[Chunk] = []
    index = 0

    for section_name, text in sections.items():
        if section_name in SKIP_SECTIONS:
            continue

        pieces = (
            [text]
            if count_tokens(text) <= settings.chunk_size_tokens
            else _split_by_tokens(
                text, settings.chunk_size_tokens, settings.chunk_overlap_tokens
            )
        )
        for piece in pieces:
            if count_tokens(piece) < MIN_CHUNK_TOKENS:
                continue  # too small to be useful in retrieval
            chunks.append(
                Chunk(
                    chunk_id=f"{arxiv_id}::{index}",
                    arxiv_id=arxiv_id,
                    section=section_name,
                    text=piece,
                    token_count=count_tokens(piece),
                    chunk_index=index,
                )
            )
            index += 1

    return chunks