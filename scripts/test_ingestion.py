from collections import Counter

from grounded_review.ingestion.pipeline import ingest

paper = ingest("https://arxiv.org/abs/2211.11501")

print(f"Title:  {paper.metadata.title}")
print(f"Chunks: {len(paper.chunks)}  |  Tokens: {paper.total_tokens}")
print()

for section, n in Counter(c.section for c in paper.chunks).items():
    print(f"  {section:<24} {n} chunk(s)")

print("\n--- sample chunk ---")
sample = paper.chunks[len(paper.chunks) // 2]
print(f"[{sample.section}] {sample.token_count} tokens")
print(sample.text[:400], "...")

assert "references" not in paper.chunks[-1].section.lower()
print("\nReference stripping: OK")