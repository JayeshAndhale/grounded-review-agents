import re
from pathlib import Path

import fitz  # PyMuPDF

# Headings that mark the end of body content. Everything after the first
# match is discarded — this is the reference-stripping step.
END_MARKERS = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography|acknowledgements|acknowledgments)\s*$",
    re.IGNORECASE,
)

# Section headings: optional numbering, then 1-6 title-case or upper-case words.
SECTION_HEADING = re.compile(
    r"^\s*(?:(?:\d+|[IVXLC]+)[\.\)]?\s+)?"
    r"([A-Z][A-Za-z]*(?:\s+[A-Za-z&\-]+){0,5})\s*$"
)

# Lines that are figure/table captions, which pollute retrieval.
CAPTION = re.compile(r"^\s*(fig(?:ure)?|table|algorithm)\.?\s*\d+", re.IGNORECASE)

# Canonical section names we try to snap headings onto.
KNOWN_SECTIONS = {
    "abstract", "introduction", "background", "related work", "motivation",
    "method", "methods", "methodology", "approach", "model", "architecture",
    "experiments", "experimental setup", "evaluation", "results",
    "analysis", "discussion", "ablation", "limitations",
    "conclusion", "conclusions", "future work",
}


class PDFParseError(Exception):
    """Raised when a PDF yields no usable body text."""


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not (3 <= len(stripped) <= 60):
        return False
    if stripped.endswith((".", ",", ";", ":")):
        return False
    if not SECTION_HEADING.match(stripped):
        return False
    # Either it matches a known section name, or it is numbered.
    normalized = re.sub(r"^[\d\.\)IVXLC\s]+", "", stripped).strip().lower()
    return normalized in KNOWN_SECTIONS or bool(re.match(r"^\d", stripped))


def _clean_line(line: str) -> str:
    # Rejoin words hyphenated across a line break, collapse whitespace.
    return re.sub(r"\s+", " ", line).strip()


def extract_sections(pdf_path: Path) -> dict[str, str]:
    """Parse a paper PDF into {section_name: text}, dropping references
    and figure captions.

    Returns at minimum a 'body' key if no sections could be identified.
    """
    with fitz.open(pdf_path) as doc:
        raw = "\n".join(page.get_text("text") for page in doc)

    if not raw.strip():
        raise PDFParseError(f"No extractable text in {pdf_path} — likely a scanned PDF")

    # Repair words split by end-of-line hyphenation before we split on newlines.
    raw = re.sub(r"-\n(\w)", r"\1", raw)

    sections: dict[str, list[str]] = {}
    current = "preamble"
    sections[current] = []

    for line in raw.split("\n"):
        if END_MARKERS.match(line):
            break  # everything from here is references — discard
        if CAPTION.match(line):
            continue  # drop caption lines
        if _looks_like_heading(line):
            current = re.sub(r"^[\d\.\)IVXLC\s]+", "", line.strip()).strip().lower()
            sections.setdefault(current, [])
            continue
        cleaned = _clean_line(line)
        if cleaned:
            sections[current].append(cleaned)

    merged = {
        name: " ".join(lines)
        for name, lines in sections.items()
        if len(" ".join(lines)) > 200  # drop fragments and stray headings
    }

    if not merged:
        raise PDFParseError(f"Parsed {pdf_path} but found no substantial sections")

    return merged