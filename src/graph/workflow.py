"""LangGraph Workflow — orchestrates the full Legal AI pipeline.

Nodes:
  1. ingest_node        — load document, chunk, build FAISS index
  2. legal_detect_node  — determine if input is a legal document
  3. retriever_node     — retrieve context for downstream chains
  4. simplify_node      — simplify the document to plain English
  5. risk_node          — identify and explain risky clauses
  6. qa_node            — answer user questions via RAG
  7. output_node        — assemble final structured output

Conditional edges route around LLM calls when the document is not legal.
"""

import logging
import time
from typing import Optional, Dict, Any

from langgraph.graph import StateGraph, END

from src.graph.state import GraphState
from src.rag.retriever import LegalRetriever
from src.legal_detector import detect_legal_document
from src.chains.simplify_chain import simplify_with_context
from src.chains.risk_chain import analyze_risks
from src.chains.qa_chain import answer_question
from src.utils.config import get_llm

logger = logging.getLogger(__name__)

# ── Module-level retriever (shared across a session) ─────────────────

_retriever: Optional[LegalRetriever] = None


def _get_retriever() -> LegalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = LegalRetriever()
    return _retriever


def reset_retriever() -> None:
    """Reset the shared retriever (e.g., when processing a new document)."""
    global _retriever
    _retriever = None


# =====================================================================
# NODE IMPLEMENTATIONS
# =====================================================================


def ingest_node(state: GraphState) -> dict:
    """Load document, clean, chunk, and build FAISS index."""
    logger.info("▶ ingest_node")
    start = time.time()
    try:
        retriever = _get_retriever()
        raw_text = retriever.ingest(
            state["input_data"],
            state.get("input_type", "text"),
        )
        stats = retriever.stats()
        elapsed = int((time.time() - start) * 1000)
        return {
            "raw_text": raw_text,
            "chunk_count": stats["chunk_count"],
            "retriever_ready": True,
            "processing_time_ms": state.get("processing_time_ms", 0) + elapsed,
        }
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        return {"error": f"Document loading failed: {e}", "raw_text": "", "retriever_ready": False}


def legal_detect_node(state: GraphState) -> dict:
    """Detect whether the document is a legal document."""
    logger.info("▶ legal_detect_node")

    raw_text = state.get("raw_text", "")
    if not raw_text:
        return {
            "is_legal": False,
            "legal_confidence": 0.0,
            "legal_classification": "not_legal",
            "detected_terms": [],
            "category_scores": {},
            "legal_explanation": "No text available for analysis.",
        }

    detection = detect_legal_document(raw_text)

    # Build human-readable explanation
    if detection.is_legal:
        top_categories = sorted(
            detection.category_scores.items(), key=lambda x: x[1], reverse=True
        )[:3]
        cat_str = ", ".join(f"{c} ({s:.1f})" for c, s in top_categories if s > 0)
        explanation = (
            f"This document is classified as {detection.classification.replace('_', ' ')} "
            f"(confidence: {detection.confidence:.0%}). "
            f"Top legal categories: {cat_str}. "
            f"Found {len(detection.detected_terms)} legal terms."
        )
    else:
        explanation = (
            "This does not appear to be a legal document. "
            f"Confidence: {detection.confidence:.0%}. "
            f"Terms found: {len(detection.detected_terms)}."
        )

    return {
        "is_legal": detection.is_legal,
        "legal_confidence": detection.confidence,
        "legal_classification": detection.classification,
        "detected_terms": detection.detected_terms,
        "category_scores": detection.category_scores,
        "legal_explanation": explanation,
    }


def retriever_node(state: GraphState) -> dict:
    """Ensure the FAISS index is ready (already built in ingest_node).

    This node exists as a checkpoint — in future iterations it could
    fetch external knowledge bases or enrich the index.
    """
    logger.info("▶ retriever_node")
    retriever = _get_retriever()
    if not retriever.is_ready:
        return {"error": "Retriever not initialised. Ingest a document first."}
    return {"retriever_ready": True}


def simplify_node(state: GraphState) -> dict:
    """Simplify the legal document using the LLM chain + RAG context."""
    logger.info("▶ simplify_node")
    start = time.time()

    try:
        retriever = _get_retriever()
        llm = get_llm()

        # Get broad context for summarisation
        context = retriever.get_summary_context(max_chunks=6)

        # Use truncated raw text
        raw_text = state.get("raw_text", "")
        # Limit to ~12k chars to stay within Groq context window
        text_for_llm = raw_text[:12000]

        simplified = simplify_with_context(text_for_llm, context, llm)
        elapsed = int((time.time() - start) * 1000)

        return {
            "simplified_text": simplified,
            "processing_time_ms": state.get("processing_time_ms", 0) + elapsed,
        }

    except Exception as e:
        logger.error(f"Simplification failed: {e}")
        return {"simplified_text": f"[Simplification failed: {e}]"}


def risk_node(state: GraphState) -> dict:
    """Analyze the document for risky clauses."""
    logger.info("▶ risk_node")
    start = time.time()

    try:
        retriever = _get_retriever()
        llm = get_llm()

        # Retrieve risk-relevant context
        risk_queries = [
            "liability limitation damages penalty",
            "termination breach default remedy",
            "indemnification warranty disclaimer",
            "non-compete restriction confidentiality",
        ]

        contexts = []
        for q in risk_queries:
            ctx = retriever.get_context_string(q, k=2)
            if ctx:
                contexts.append(ctx)

        combined_context = "\n\n---\n\n".join(contexts) if contexts else ""

        raw_text = state.get("raw_text", "")
        text_for_llm = raw_text[:12000]

        risk_result = analyze_risks(text_for_llm, combined_context, llm)
        elapsed = int((time.time() - start) * 1000)

        return {
            "risk_analysis": risk_result,
            "processing_time_ms": state.get("processing_time_ms", 0) + elapsed,
        }

    except Exception as e:
        logger.error(f"Risk analysis failed: {e}")
        return {"risk_analysis": f"[Risk analysis failed: {e}]"}


def qa_node(state: GraphState) -> dict:
    """Answer a user question about the document using RAG."""
    logger.info("▶ qa_node")

    question = state.get("user_question", "")
    if not question or not question.strip():
        return {"qa_answer": ""}

    try:
        retriever = _get_retriever()
        llm = get_llm()
        context = retriever.get_context_string(question, k=4)
        answer = answer_question(question, context, llm)
        return {"qa_answer": answer}

    except Exception as e:
        logger.error(f"QA failed: {e}")
        return {"qa_answer": f"[Failed to answer: {e}]"}


def output_node(state: GraphState) -> dict:
    """Final node — passthrough that signals completion."""
    logger.info("▶ output_node (done)")
    return {}


# =====================================================================
# ROUTING FUNCTIONS
# =====================================================================


def route_after_detection(state: GraphState) -> str:
    """Route after legal detection: skip LLM calls if not legal."""
    if state.get("error"):
        return "output_node"
    if state.get("is_legal", False):
        return "retriever_node"
    return "output_node"


def route_after_retriever(state: GraphState) -> str:
    """After retriever, decide whether to run QA or full analysis."""
    if state.get("user_question"):
        return "qa_node"
    return "simplify_node"


def route_after_qa(state: GraphState) -> str:
    """After QA, go to output."""
    return "output_node"


def route_after_simplify(state: GraphState) -> str:
    """After simplification, run risk analysis."""
    return "risk_node"


def route_after_risk(state: GraphState) -> str:
    """After risk analysis, check if there's also a QA question."""
    if state.get("user_question") and not state.get("qa_answer"):
        return "qa_node"
    return "output_node"


# =====================================================================
# GRAPH CONSTRUCTION
# =====================================================================


def build_workflow() -> StateGraph:
    """Construct and compile the LangGraph workflow.

    Graph topology:
      ingest → legal_detect ─┬─ (legal)     → retriever ─┬─ (has question) → qa → output
                              │                            └─ (no question)  → simplify → risk ─┬─ (has Q) → qa → output
                              │                                                                  └─ (no Q) → output
                              └─ (not legal) → output
    """
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("ingest_node", ingest_node)
    graph.add_node("legal_detect_node", legal_detect_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("simplify_node", simplify_node)
    graph.add_node("risk_node", risk_node)
    graph.add_node("qa_node", qa_node)
    graph.add_node("output_node", output_node)

    # Entry point
    graph.set_entry_point("ingest_node")

    # Edges
    graph.add_edge("ingest_node", "legal_detect_node")

    graph.add_conditional_edges(
        "legal_detect_node",
        route_after_detection,
        {
            "retriever_node": "retriever_node",
            "output_node": "output_node",
        },
    )

    graph.add_conditional_edges(
        "retriever_node",
        route_after_retriever,
        {
            "qa_node": "qa_node",
            "simplify_node": "simplify_node",
        },
    )

    graph.add_edge("simplify_node", "risk_node")

    graph.add_conditional_edges(
        "risk_node",
        route_after_risk,
        {
            "qa_node": "qa_node",
            "output_node": "output_node",
        },
    )

    graph.add_edge("qa_node", "output_node")
    graph.add_edge("output_node", END)

    return graph.compile()


# =====================================================================
# HIGH-LEVEL API
# =====================================================================


def run_full_analysis(
    input_data,
    input_type: str = "text",
    user_question: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the complete analysis pipeline and return the final state.

    Args:
        input_data:    Raw text string or PDF file path / buffer.
        input_type:    "text" or "pdf".
        user_question: Optional question to answer about the document.

    Returns:
        Final GraphState dict with all analysis results.
    """
    reset_retriever()
    workflow = build_workflow()

    initial_state: GraphState = {
        "input_data": input_data,
        "input_type": input_type,
        "user_question": user_question or "",
        "raw_text": "",
        "chunk_count": 0,
        "is_legal": False,
        "legal_confidence": 0.0,
        "legal_classification": "not_legal",
        "detected_terms": [],
        "category_scores": {},
        "legal_explanation": "",
        "retriever_ready": False,
        "simplified_text": "",
        "risk_analysis": "",
        "qa_answer": "",
        "error": None,
        "processing_time_ms": 0,
    }

    result = workflow.invoke(initial_state)
    return dict(result)


def run_qa(question: str) -> str:
    """Run a QA query against the already-ingested document.

    The document must have been ingested by a prior run_full_analysis call.
    """
    retriever = _get_retriever()
    if not retriever.is_ready:
        return "No document loaded. Please upload and analyze a document first."

    try:
        llm = get_llm()
        context = retriever.get_context_string(question, k=4)
        return answer_question(question, context, llm)
    except Exception as e:
        logger.error(f"QA query failed: {e}")
        return f"Failed to answer: {e}"
