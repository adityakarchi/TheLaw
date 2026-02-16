"""LangGraph State Definition — typed state passed between workflow nodes."""

from typing import TypedDict, Optional, List, Dict, Any


class GraphState(TypedDict, total=False):
    """Shared state that flows through every LangGraph node.

    Fields are populated progressively as nodes execute.
    Using total=False so nodes can write only the fields they own.
    """

    # Input 
    input_data: Any               # Raw input (str or file path)
    input_type: str               # "text" or "pdf"
    user_question: Optional[str]  # Optional QA question

    #  Document 
    raw_text: str                 # Cleaned full document text
    chunk_count: int              # Number of chunks created

    #  Legal Detection 
    is_legal: bool
    legal_confidence: float
    legal_classification: str     # "definitely_legal", "likely_legal", etc.
    detected_terms: List[str]
    category_scores: Dict[str, float]
    legal_explanation: str        # Human-readable explanation

    #  RAG / Retrieval 
    retriever_ready: bool

    #  Simplification 
    simplified_text: str

    #  Risk Analysis 
    risk_analysis: str

    #  QA 
    qa_answer: str

    #  Meta 
    error: Optional[str]
    processing_time_ms: int
