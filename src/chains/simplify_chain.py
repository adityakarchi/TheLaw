"""Simplification Chain — converts legal text to plain English using RAG context.

Uses LangChain + Groq to produce a structured simplified version that
preserves all legal obligations, rights, deadlines, and conditions.
"""

import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.utils.config import get_llm, LLM_MODEL

logger = logging.getLogger(__name__)

# ── Prompt Templates ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior legal translator with 20 years of experience converting
complex legal documents into clear, plain English that any non-lawyer can understand.

CRITICAL RULES:
1. Preserve ALL substantive legal meaning — obligations, rights, conditions, deadlines.
2. Replace legal jargon with everyday words; if a term MUST stay, define it inline.
3. Break long compound sentences into short, clear statements.
4. Use bullet points for lists of conditions, obligations, or exceptions.
5. Group related clauses under descriptive headings.
6. Highlight any red-flag clauses (unlimited liability, unilateral termination, etc.).
7. Be concise but complete — do NOT omit information.
8. End with a brief "Key Takeaways" section (3-5 bullet points).
9. If the text references other sections, note the reference without fabricating content."""

SIMPLIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Simplify the following legal text into plain English.

RELEVANT CONTEXT FROM THE SAME DOCUMENT (use to improve accuracy):
---
{context}
---

LEGAL TEXT TO SIMPLIFY:
\"\"\"
{text}
\"\"\"

SIMPLIFIED VERSION:"""),
])

SIMPLIFY_NO_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Simplify the following legal text into plain English.

LEGAL TEXT TO SIMPLIFY:
\"\"\"
{text}
\"\"\"

SIMPLIFIED VERSION:"""),
])


# ── Chain Construction ───────────────────────────────────────────────

def get_simplify_chain(llm: Optional[ChatGroq] = None):
    """Build a LangChain simplification chain with RAG context support."""
    _llm = llm or get_llm()
    return SIMPLIFY_PROMPT | _llm | StrOutputParser()


def get_simplify_chain_no_context(llm: Optional[ChatGroq] = None):
    """Simplification chain without retrieved context (fallback)."""
    _llm = llm or get_llm()
    return SIMPLIFY_NO_CONTEXT_PROMPT | _llm | StrOutputParser()


# ── High-Level API ──────────────────────────────────────────────────

def simplify_with_context(
    text: str,
    context: str = "",
    llm: Optional[ChatGroq] = None,
) -> str:
    """Simplify legal text, optionally enriched with retrieved context.

    Args:
        text:    The legal text to simplify.
        context: RAG-retrieved context from the same document.
        llm:     Optional pre-configured LLM instance.

    Returns:
        Simplified plain-English version of the text.
    """
    if not text or len(text.strip()) < 20:
        return "Text too short to simplify."

    try:
        if context.strip():
            chain = get_simplify_chain(llm)
            result = chain.invoke({"text": text, "context": context})
        else:
            chain = get_simplify_chain_no_context(llm)
            result = chain.invoke({"text": text})

        return result.strip()

    except Exception as e:
        logger.error(f"Simplification chain failed: {e}")
        raise
