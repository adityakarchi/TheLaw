"""LangGraph State Definition — typed state passed between workflow nodes.

Supports TWO autonomous pipelines:
  1. Legal Document Simplifier — contract analysis
  2. Legal Case Research Assistant — criminal law lookup
"""

from typing import TypedDict, Optional, List, Dict, Any


class GraphState(TypedDict, total=False):
    """Shared state that flows through every LangGraph node.

    Fields are populated progressively as nodes execute.
    Using total=False so nodes can write only the fields they own.
    """

    # Routing
    pipeline_mode: str            # "simplifier" or "case_research"

    # Input
    input_data: Any               # Raw input (str or file path)
    input_type: str               # "text" or "pdf"
    user_question: Optional[str]  # Optional QA question

    # Document (Simplifier)
    raw_text: str                 # Cleaned full document text
    chunk_count: int              # Number of chunks created

    # Legal Detection (Simplifier)
    is_legal: bool
    legal_confidence: float
    legal_classification: str     # "definitely_legal", "likely_legal", etc.
    detected_terms: List[str]
    category_scores: Dict[str, float]
    legal_explanation: str        # Human-readable explanation

    # RAG / Retrieval (Simplifier)
    retriever_ready: bool

    # Simplification (Simplifier)
    simplified_text: str

    # Risk Analysis (Simplifier)
    risk_analysis: str

    # QA (Simplifier)
    qa_answer: str

    # Case Research (Lawyer Assistant)
    case_description: str         # User's case description
    retrieved_laws: List[Dict[str, Any]]  # Structured law results from FAISS
    law_context: str              # Formatted law context for LLM
    case_analysis: str            # LLM-generated legal analysis
    case_sections_found: int      # Number of relevant sections found

    # Meta
    error: Optional[str]
    processing_time_ms: int
