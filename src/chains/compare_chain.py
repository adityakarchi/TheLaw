"""Compare Chain — LLM-powered change summarization for contract comparison.

Feature 5: Contract Comparison mode.
"""

import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.utils.config import get_llm

logger = logging.getLogger(__name__)

CHANGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a contract comparison expert. Summarize what changed between the "
               "original and revised clause in 1-2 sentences. Note any risk implications."),
    ("human", """ORIGINAL CLAUSE:
\"\"\"{original}\"\"\"

REVISED CLAUSE:
\"\"\"{revised}\"\"\"

What changed and what is the risk impact?"""),
])

OVERALL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a senior contract analyst. Summarize the overall differences between "
               "two versions of a contract. Focus on risk changes and key modifications."),
    ("human", """The following changes were found between the original and revised contract:

{changes_summary}

RISK DELTA: {risk_delta}

Provide a 3-5 paragraph executive summary of:
1. Key changes made
2. How the overall risk profile changed
3. Clauses that need attention
4. Recommendation (accept/negotiate/reject the changes)

COMPARISON REPORT:"""),
])


def get_change_chain(llm: Optional[ChatGroq] = None):
    _llm = llm or get_llm()
    return CHANGE_PROMPT | _llm | StrOutputParser()


def get_overall_chain(llm: Optional[ChatGroq] = None):
    _llm = llm or get_llm()
    return OVERALL_PROMPT | _llm | StrOutputParser()


def summarize_change(
    original_clause: str,
    revised_clause: str,
    llm: Optional[ChatGroq] = None,
) -> str:
    """Summarize what changed between two clause versions."""
    if not original_clause and not revised_clause:
        return "Both clauses are empty."
    if not original_clause:
        return "New clause added in revision."
    if not revised_clause:
        return "Clause was removed in revision."

    try:
        chain = get_change_chain(llm)
        return chain.invoke({
            "original": original_clause[:1500],
            "revised": revised_clause[:1500],
        }).strip()
    except Exception as e:
        logger.error(f"Change summarization failed: {e}")
        return f"[Error summarizing change: {e}]"


def overall_comparison_summary(
    changes: list,
    risk_delta: dict,
    llm: Optional[ChatGroq] = None,
) -> str:
    """Generate an overall comparison report.

    Args:
        changes: List of change dicts from segment_and_align().
        risk_delta: Risk delta dict from calculate_risk_delta().
        llm: Optional LLM instance.

    Returns:
        Full comparison report string.
    """
    # Build a concise changes summary for the prompt
    summary_lines = []
    modified = [c for c in changes if c["status"] == "modified"]
    added = [c for c in changes if c["status"] == "added"]
    removed = [c for c in changes if c["status"] == "removed"]
    unchanged = [c for c in changes if c["status"] == "unchanged"]

    summary_lines.append(f"- {len(unchanged)} clauses unchanged")
    summary_lines.append(f"- {len(modified)} clauses modified")
    summary_lines.append(f"- {len(added)} clauses added")
    summary_lines.append(f"- {len(removed)} clauses removed")

    # Include top 5 modified clauses for context
    for i, change in enumerate(modified[:5]):
        orig_snippet = change["original"][:100]
        rev_snippet = change["revised"][:100]
        summary_lines.append(
            f"\nModified {i+1}: \"{orig_snippet}...\" → \"{rev_snippet}...\""
        )

    changes_text = "\n".join(summary_lines)

    delta_text = (
        f"Risk {risk_delta.get('direction', 'unchanged')}: "
        f"avg score {risk_delta.get('original_avg_score', 0):.2f} → "
        f"{risk_delta.get('revised_avg_score', 0):.2f} "
        f"({risk_delta.get('new_high_risk', 0)} new high-risk clauses)"
    )

    try:
        chain = get_overall_chain(llm)
        return chain.invoke({
            "changes_summary": changes_text,
            "risk_delta": delta_text,
        }).strip()
    except Exception as e:
        logger.error(f"Overall comparison failed: {e}")
        return f"[Error generating comparison: {e}]"
