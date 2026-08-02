"""Contract Comparison — clause-level diff with risk delta analysis.

Feature 5: Compare Contracts mode.
"""

import logging
import re
from difflib import SequenceMatcher
from typing import List, Dict

logger = logging.getLogger(__name__)


def _simple_segment(text: str) -> List[str]:
    """Split text into clause-like segments for alignment."""
    # Split on numbered sections, double newlines, or paragraph boundaries
    parts = re.split(
        r"(?:\n\s*\n)|(?:\n\s*\d+[\.\)]\s+)|(?:\n\s*(?:Section|Article|Clause)\s+\d+)",
        text,
        flags=re.IGNORECASE,
    )
    return [p.strip() for p in parts if p and len(p.strip()) > 20]


def segment_and_align(text_a: str, text_b: str) -> List[Dict]:
    """Align clauses between two contracts and produce a diff.

    Uses SequenceMatcher to find matching, added, removed, and modified clauses.

    Args:
        text_a: Original contract text.
        text_b: Revised contract text.

    Returns:
        List of dicts with keys:
            status: "unchanged" | "modified" | "added" | "removed"
            original: str (empty for added)
            revised: str (empty for removed)
            similarity: float (0-1, only for modified)
    """
    clauses_a = _simple_segment(text_a)
    clauses_b = _simple_segment(text_b)

    if not clauses_a and not clauses_b:
        return []

    matcher = SequenceMatcher(None, clauses_a, clauses_b)
    results = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                results.append({
                    "status": "unchanged",
                    "original": clauses_a[k],
                    "revised": clauses_b[j1 + (k - i1)],
                    "similarity": 1.0,
                })
        elif tag == "replace":
            # Modified clauses: pair them up
            a_chunk = clauses_a[i1:i2]
            b_chunk = clauses_b[j1:j2]
            max_len = max(len(a_chunk), len(b_chunk))
            for k in range(max_len):
                orig = a_chunk[k] if k < len(a_chunk) else ""
                rev = b_chunk[k] if k < len(b_chunk) else ""

                if orig and rev:
                    sim = SequenceMatcher(None, orig, rev).ratio()
                    results.append({
                        "status": "modified",
                        "original": orig,
                        "revised": rev,
                        "similarity": round(sim, 3),
                    })
                elif orig:
                    results.append({
                        "status": "removed",
                        "original": orig,
                        "revised": "",
                        "similarity": 0.0,
                    })
                else:
                    results.append({
                        "status": "added",
                        "original": "",
                        "revised": rev,
                        "similarity": 0.0,
                    })
        elif tag == "delete":
            for k in range(i1, i2):
                results.append({
                    "status": "removed",
                    "original": clauses_a[k],
                    "revised": "",
                    "similarity": 0.0,
                })
        elif tag == "insert":
            for k in range(j1, j2):
                results.append({
                    "status": "added",
                    "original": "",
                    "revised": clauses_b[k],
                    "similarity": 0.0,
                })

    logger.info(
        f"Compared {len(clauses_a)} vs {len(clauses_b)} clauses → "
        f"{len(results)} aligned pairs"
    )
    return results


def calculate_risk_delta(
    clauses_a_risks: List[Dict],
    clauses_b_risks: List[Dict],
) -> Dict:
    """Compare risk scores between two sets of classified clauses.

    Args:
        clauses_a_risks: List of classification dicts for original contract.
        clauses_b_risks: List of classification dicts for revised contract.

    Returns:
        Dict with risk delta summary.
    """
    def _count_levels(risks):
        counts = {"High": 0, "Medium": 0, "Low": 0}
        for r in risks:
            level = r.get("risk_level", "Low")
            if level in counts:
                counts[level] += 1
        return counts

    def _avg_score(risks):
        scores = [r.get("risk_score", 0) for r in risks]
        return sum(scores) / max(len(scores), 1)

    counts_a = _count_levels(clauses_a_risks)
    counts_b = _count_levels(clauses_b_risks)
    avg_a = _avg_score(clauses_a_risks)
    avg_b = _avg_score(clauses_b_risks)

    delta = avg_b - avg_a
    if delta > 0.1:
        direction = "increased"
    elif delta < -0.1:
        direction = "decreased"
    else:
        direction = "unchanged"

    return {
        "original_counts": counts_a,
        "revised_counts": counts_b,
        "original_avg_score": round(avg_a, 3),
        "revised_avg_score": round(avg_b, 3),
        "delta": round(delta, 3),
        "direction": direction,
        "new_high_risk": max(0, counts_b["High"] - counts_a["High"]),
    }
