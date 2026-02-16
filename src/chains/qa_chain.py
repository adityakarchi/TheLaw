"""Contract QA Chain — answers natural-language questions using RAG.

Retrieves relevant clauses from the FAISS index and feeds them to the
LLM to produce grounded, cited answers.
"""

import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.utils.config import get_llm

logger = logging.getLogger(__name__)

# ── Prompt Templates ────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """You are a helpful legal assistant who answers questions about contracts
and legal documents. You ONLY answer based on the provided document context.

RULES:
1. Answer ONLY using information from the provided context.
2. If the answer is not in the context, say "I cannot find this information in the document."
3. Quote relevant clauses when possible.
4. Explain legal terms in plain English.
5. Be precise — do not speculate or add information not present in the document.
6. If the question is ambiguous, address all possible interpretations.
7. Structure longer answers with bullet points or numbered lists."""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    ("human", """Based on the following context from a legal document, answer the question.

DOCUMENT CONTEXT:
---
{context}
---

QUESTION: {question}

ANSWER:"""),
])


# ── Chain Construction ───────────────────────────────────────────────

def get_qa_chain(llm: Optional[ChatGroq] = None):
    """Build a LangChain QA chain for contract questions."""
    _llm = llm or get_llm()
    return QA_PROMPT | _llm | StrOutputParser()


# ── High-Level API ──────────────────────────────────────────────────

def answer_question(
    question: str,
    context: str,
    llm: Optional[ChatGroq] = None,
) -> str:
    """Answer a question about a legal document using retrieved context.

    Args:
        question: The user's natural-language question.
        context:  RAG-retrieved document chunks relevant to the question.
        llm:      Optional pre-configured LLM instance.

    Returns:
        LLM-generated answer grounded in the provided context.
    """
    if not question or not question.strip():
        return "Please provide a question."

    if not context or not context.strip():
        return ("No relevant context was found in the document to answer this question. "
                "Please try rephrasing or ask a different question.")

    try:
        chain = get_qa_chain(llm)
        result = chain.invoke({"question": question, "context": context})
        return result.strip()

    except Exception as e:
        logger.error(f"QA chain failed: {e}")
        raise
