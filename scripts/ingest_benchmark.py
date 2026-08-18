# scripts/ingest_benchmark.py
from rich.console import Console
from rich.table import Table

from grounded_review.evaluation.benchmark import all_arxiv_ids
from grounded_review.ingestion.pipeline import ingest
from grounded_review.ingestion.arxiv_client import PaperNotFoundError
from grounded_review.retrieval.vector_store import PaperStore

console = Console()

def main():
    store = PaperStore()
    ids = all_arxiv_ids()

    table = Table(title="Benchmark ingestion")
    table.add_column("arXiv ID")
    table.add_column("Status")
    table.add_column("Chunks")
    table.add_column("Sections seen")

    for arxiv_id in ids:
        if store.has_paper(arxiv_id):
            table.add_row(arxiv_id, "already ingested", "-", "-")
            continue
        try:
            paper = ingest(arxiv_id)
            count = store.add_paper(paper)
            sections = sorted({c.section for c in paper.chunks})
            table.add_row(arxiv_id, "ingested", str(count), ", ".join(sections))
        except PaperNotFoundError as e:
            table.add_row(arxiv_id, f"[red]NOT FOUND: {e}[/red]", "-", "-")
        except Exception as e:
            table.add_row(arxiv_id, f"[red]FAILED: {type(e).__name__}: {e}[/red]", "-", "-")

    console.print(table)
    console.print(f"\nTotal chunks in store: {store.count()}")

if __name__ == "__main__":
    main()