"""Clause segmentation utilities."""

import re

def segment_clauses(text: str):
    """Split text into clauses using punctuation patterns."""
    clauses = re.split(r"\.\s+|;\s+|\n+|\:\s+", text)
    clauses = [c.strip() for c in clauses if len(c.strip()) > 20]
    return clauses