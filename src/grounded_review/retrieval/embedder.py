from functools import lru_cache

from sentence_transformers import SentenceTransformer

from ..config import get_settings


@lru_cache
def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it.

    Loading takes several seconds and allocates real memory, so this must
    not happen per-call.
    """
    return SentenceTransformer(get_settings().embedding_model)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a batch of texts into normalised vectors.

    Normalising means cosine similarity reduces to a dot product, which is
    what Chroma computes internally.
    """
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 64,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]


def embedding_dimension() -> int:
    return get_model().get_sentence_embedding_dimension()