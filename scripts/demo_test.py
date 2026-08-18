# scratch_check_survey.py
from grounded_review.ingestion.arxiv_client import fetch_metadata, download_pdf
from grounded_review.ingestion.pdf_parser import extract_sections

meta = fetch_metadata("2202.03629")
pdf_path = download_pdf(meta)
sections = extract_sections(pdf_path)

for name, text in sections.items():
    print(f"--- {name} ({len(text)} chars) ---")
    print(text[:150])
    print()