"""Clause Simplification Chain — plain-English explanation per clause.

Feature 4: Clause-by-Clause Breakdown Tab.
"""

import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.utils.config import get_llm

logger = logging.getLogger(__name__)

CLAUSE_SYSTEM_PROMPT = """You are a legal plain-language expert.
Given a legal clause and its detected type, explain it in 1-3 simple sentences
that any non-lawyer can understand. Also provide one brief negotiation tip."""

CLAUSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CLAUSE_SYSTEM_PROMPT),
    ("human", """Explain this {clause_type} clause in plain English (1-3 sentences),
then give one negotiation tip.

CLAUSE TEXT:
\"\"\"{clause_text}\"\"\"

Format your response as:
PLAIN ENGLISH: <explanation>
NEGOTIATION TIP: <tip>"""),
])


def get_clause_chain(llm: Optional[ChatGroq] = None):
    """Build a clause simplification chain."""
    _llm = llm or get_llm()
    return CLAUSE_PROMPT | _llm | StrOutputParser()


def simplify_clause(
    clause_text: str,
    clause_type: str = "General",
    llm: Optional[ChatGroq] = None,
) -> str:
    """Simplify a single clause into plain English with a negotiation tip.

    Args:
        clause_text: The raw clause text.
        clause_type: Detected clause type (from classifier).
        llm: Optional pre-configured LLM instance.

    Returns:
        Plain English explanation + negotiation tip.
    """
    if not clause_text or len(clause_text.strip()) < 15:
        return "PLAIN ENGLISH: Clause too short to summarize.\nNEGOTIATION TIP: N/A"

    try:
        chain = get_clause_chain(llm)
        result = chain.invoke({
            "clause_text": clause_text[:1500],  # limit per-clause context
            "clause_type": clause_type,
        })
        return result.strip()
    except Exception as e:
        logger.error(f"Clause simplification failed: {e}")
        return f"PLAIN ENGLISH: [Error simplifying clause]\nNEGOTIATION TIP: N/A"


def simplify_clauses_batch(
    clauses: list[dict],
    llm: Optional[ChatGroq] = None,
    max_clauses: int = 15,
) -> list[dict]:
    """Simplify multiple clauses sequentially.

    Args:
        clauses: List of dicts with "text" and "type" keys.
        llm: Optional pre-configured LLM instance.
        max_clauses: Max clauses to process (avoids rate limits).

    Returns:
        List of dicts with added "summary" and "negotiation_tip" keys.
    """
    _llm = llm or get_llm()
    results = []

    for clause in clauses[:max_clauses]:
        text = clause.get("text", "")
        ctype = clause.get("type", "General")

        raw = simplify_clause(text, ctype, _llm)

        # Parse the structured response
        summary = raw
        tip = ""
        if "PLAIN ENGLISH:" in raw and "NEGOTIATION TIP:" in raw:
            parts = raw.split("NEGOTIATION TIP:")
            summary = parts[0].replace("PLAIN ENGLISH:", "").strip()
            tip = parts[1].strip() if len(parts) > 1 else ""

        results.append({
            **clause,
            "summary": summary,
            "negotiation_tip": tip,
        })

    return results
