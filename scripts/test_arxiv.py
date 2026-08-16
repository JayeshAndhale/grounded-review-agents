from grounded_review.ingestion.arxiv_client import (
    PaperNotFoundError,
    download_pdf,
    fetch_metadata,
)

meta = fetch_metadata("https://arxiv.org/abs/2211.11501")
print(f"Title:    {meta.title}")
print(f"Authors:  {len(meta.authors)} — {meta.authors[0]}, ...")
print(f"Citation: {meta.citation()}")

path = download_pdf(meta)
print(f"PDF:      {path} ({path.stat().st_size // 1024} KB)")

try:
    fetch_metadata("9999.99999")
except PaperNotFoundError as e:
    print(f"Refusal:  {e}")