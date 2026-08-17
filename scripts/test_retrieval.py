from grounded_review.ingestion.pipeline import ingest
from grounded_review.retrieval.embedder import embedding_dimension
from grounded_review.retrieval.vector_store import PaperStore

PAPERS = [
    "https://arxiv.org/abs/2211.11501",  # DS-1000
    "https://arxiv.org/abs/2406.15877",  # BigCodeBench
]

store = PaperStore()
store.reset()
print(f"Embedding dimension: {embedding_dimension()}\n")

for url in PAPERS:
    paper = ingest(url)
    n = store.add_paper(paper)
    print(f"Indexed {n:>3} chunks — {paper.metadata.title[:60]}")

print(f"\nTotal chunks in store: {store.count()}\n")

queries = [
    "how was the benchmark constructed",
    "what are the limitations of this evaluation",
    "how do models perform on the hardest tasks",
]

for q in queries:
    print(f"Q: {q}")
    for hit in store.search(q, top_k=3):
        print(f"   {hit.score:.3f}  [{hit.chunk.arxiv_id} §{hit.chunk.section}]")
        print(f"          {hit.chunk.text[:110]}...")
    print()

print("Scoped to one paper, Results sections only:")
for hit in store.search(
    "accuracy of the best model",
    arxiv_id="2211.11501",
    sections=["results", "experiments", "evaluation"],
    top_k=2,
):
    print(f"   {hit.score:.3f}  [{hit.chunk.section}] {hit.chunk.text[:110]}...")