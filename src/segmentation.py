"""Clause Segmentation — splits legal documents into individual clauses.

Feature 4: Clause-by-Clause Breakdown Tab.
"""

import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Patterns that indicate clause/section boundaries
SECTION_PATTERNS = [
    # "Section 1.", "Section 1:", "SECTION 1"
    r"(?:^|\n)\s*(?:SECTION|Section)\s+\d+[\.\:\s]",
    # "Article 1.", "ARTICLE I"
    r"(?:^|\n)\s*(?:ARTICLE|Article)\s+[IVXLCDM\d]+[\.\:\s]",
    # "Clause 1.", "CLAUSE 1"
    r"(?:^|\n)\s*(?:CLAUSE|Clause)\s+\d+[\.\:\s]",
    # "1.", "2.", "3." at start of line (numbered list)
    r"(?:^|\n)\s*\d+\.\s+[A-Z]",
    # "1)", "2)", "(1)", "(2)" at start of line
    r"(?:^|\n)\s*[\(\[]?\d+[\)\]]\s+",
    # "(a)", "(b)", "(i)", "(ii)" at start of line
    r"(?:^|\n)\s*[\(\[]?[a-z][\)\]]\s+",
    # Roman numerals: "(i)", "(ii)", "(iii)"
    r"(?:^|\n)\s*[\(\[]?(?:i{1,3}|iv|v|vi{0,3}|ix|x)[\)\]]\s+",
]

# Combined pattern for splitting
SPLIT_PATTERN = re.compile(
    "|".join(SECTION_PATTERNS),
    re.MULTILINE | re.IGNORECASE,
)

# Pattern to extract heading from a clause
HEADING_PATTERN = re.compile(
    r"^(?:(?:SECTION|Section|ARTICLE|Article|CLAUSE|Clause)\s+[\dIVXLCDM]+[\.\:\s]*)?(.+?)(?:\.|:|\n)",
    re.IGNORECASE,
)


def segment_into_clauses(text: str, min_length: int = 30) -> List[Dict]:
    """Split legal text into individual clauses.

    Uses regex patterns to identify clause boundaries based on numbered
    sections, articles, and common legal formatting.

    Args:
        text: Full legal document text.
        min_length: Minimum character length for a valid clause.

    Returns:
        List of dicts: {"index": int, "text": str, "heading": str}
    """
    if not text or len(text.strip()) < min_length:
        return []

    # Find all split points
    split_points = []
    for match in SPLIT_PATTERN.finditer(text):
        split_points.append(match.start())

    # If no structured sections found, fall back to paragraph splitting
    if len(split_points) < 2:
        return _segment_by_paragraphs(text, min_length)

    # Add start and end
    if split_points[0] != 0:
        split_points.insert(0, 0)
    split_points.append(len(text))

    clauses = []
    for i in range(len(split_points) - 1):
        chunk = text[split_points[i]:split_points[i + 1]].strip()
        if len(chunk) < min_length:
            continue

        heading = _extract_heading(chunk)

        clauses.append({
            "index": len(clauses) + 1,
            "text": chunk,
            "heading": heading,
        })

    if not clauses:
        return _segment_by_paragraphs(text, min_length)

    logger.info(f"Segmented document into {len(clauses)} clauses")
    return clauses


def _segment_by_paragraphs(text: str, min_length: int = 30) -> List[Dict]:
    """Fallback: split by double newlines (paragraphs)."""
    paragraphs = re.split(r"\n\s*\n", text)
    clauses = []

    for para in paragraphs:
        para = para.strip()
        if len(para) < min_length:
            continue

        heading = _extract_heading(para)

        clauses.append({
            "index": len(clauses) + 1,
            "text": para,
            "heading": heading,
        })

    logger.info(f"Segmented document into {len(clauses)} paragraphs (fallback)")
    return clauses


def _extract_heading(text: str) -> str:
    """Extract a short heading from the first line of a clause."""
    first_line = text.split("\n")[0].strip()

    # Remove leading numbers/letters like "1.", "(a)", "Section 3."
    cleaned = re.sub(
        r"^(?:(?:SECTION|ARTICLE|CLAUSE)\s+)?[\d\(\)\[\]IVXLCDMivxlcdma-z\.]+[\.\:\)\]\s]+",
        "",
        first_line,
        flags=re.IGNORECASE,
    ).strip()

    # Truncate to reasonable heading length
    if len(cleaned) > 80:
        cleaned = cleaned[:77] + "…"

    return cleaned if cleaned else f"Clause {text[:30].strip()}…"