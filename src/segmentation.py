import re

def segment_clauses(text: str):
    """
    Splits cleaned legal text into individual clauses.
    """
    clauses = re.split(r"\.\s+|;\s+|\n+|\:\s+", text)
    clauses = [c.strip() for c in clauses if len(c.strip()) > 20]
    return clauses
