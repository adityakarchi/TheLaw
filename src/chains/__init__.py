"""LangChain chains for legal document processing."""

from src.chains.simplify_chain import get_simplify_chain, simplify_with_context
from src.chains.risk_chain import get_risk_chain, analyze_risks
from src.chains.qa_chain import get_qa_chain, answer_question

__all__ = [
    "get_simplify_chain", "simplify_with_context",
    "get_risk_chain", "analyze_risks",
    "get_qa_chain", "answer_question",
]
