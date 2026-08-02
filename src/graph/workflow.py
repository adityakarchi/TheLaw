"""LangGraph Workflow — orchestrates the full Legal AI Platform.

TWO autonomous pipelines:

Pipeline 1: Legal Document Simplifier
  ingest → detect → [if legal] → retrieve → simplify → risk → output

Pipeline 2: Legal Case Research Assistant
  case_input → law_search (FAISS) → case_explain (LLM) → output

LLM (Groq) is ONLY used for explanation, NOT for retrieval/classification/risk scoring.
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
from src.chains.case_chain import analyze_case
from src.utils.config import get_llm

logger = logging.getLogger(__name__)

# Module-level singletons

_retriever: Optional[LegalRetriever] = None
_law_index = None
_pol_index = None   # Pile of Law FAISS index (Feature 2)
_classifier = None  # Clause classifier singleton (Feature 1)


def _get_retriever() -> LegalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = LegalRetriever()
    return _retriever


def _get_law_index():
    """Lazy-load law index singleton."""
    global _law_index
    if _law_index is None:
        from src.rag.law_index import LawIndexBuilder
        _law_index = LawIndexBuilder()
        _law_index.build_index()
    return _law_index


def _get_pol_index():
    """Lazy-load Pile of Law FAISS index (Feature 2)."""
    global _pol_index
    if _pol_index is None:
        try:
            from src.rag.pile_of_law import load_pol_documents
            from src.rag.embedder import EmbeddingPipeline
            from src.rag.vectordb import VectorStore
            import faiss
            from pathlib import Path

            index_dir = Path(__file__).resolve().parent.parent.parent / "data" / "pile_of_law_index"

            if index_dir.exists() and (index_dir / "index.faiss").exists():
                logger.info("Loading cached Pile of Law index")
                embedder = EmbeddingPipeline()
                vs = VectorStore(embedder)
                vs.load(str(index_dir))
                _pol_index = vs
            else:
                logger.info("Building Pile of Law index (first run)...")
                docs = load_pol_documents(max_per_source=50)
                if docs:
                    embedder = EmbeddingPipeline()
                    vs = VectorStore(embedder)
                    vs.add_documents(docs)
                    index_dir.mkdir(parents=True, exist_ok=True)
                    vs.save(str(index_dir))
                    _pol_index = vs
                    logger.info(f"Built Pile of Law index with {len(docs)} docs")
                else:
                    logger.warning("No Pile of Law data found")
                    return None
        except Exception as e:
            logger.warning(f"Pile of Law index unavailable: {e}")
            return None
    return _pol_index


def _get_classifier():
    """Lazy-load clause classifier singleton (Feature 1)."""
    global _classifier
    if _classifier is None:
        try:
            from src.classification import get_classifier
            _classifier = get_classifier()
        except Exception as e:
            logger.warning(f"Clause classifier unavailable: {e}")
            return None
    return _classifier


def reset_retriever() -> None:
    """Reset the shared retriever (new document)."""
    global _retriever
    _retriever = None


def reset_law_index() -> None:
    """Reset the law index (force rebuild)."""
    global _law_index
    _law_index = None


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
        language = state.get("output_language", "English")

        # Get broad context for summarisation
        context = retriever.get_summary_context(max_chunks=6)

        # Use truncated raw text
        raw_text = state.get("raw_text", "")
        # Limit to ~12k chars to stay within Groq context window
        text_for_llm = raw_text[:12000]

        simplified = simplify_with_context(text_for_llm, context, llm, language=language)
        elapsed = int((time.time() - start) * 1000)

        return {
            "simplified_text": simplified,
            "processing_time_ms": state.get("processing_time_ms", 0) + elapsed,
        }

    except Exception as e:
        logger.error(f"Simplification failed: {e}")
        return {"simplified_text": f"[Simplification failed: {e}]"}


def risk_node(state: GraphState) -> dict:
    """Analyze the document for risky clauses.

    Feature 1: Pre-classifies clauses with BERT before sending to LLM.
    """
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

        # Feature 1: Pre-classify clauses with BERT
        classifier = _get_classifier()
        raw_text = state.get("raw_text", "")
        classifier_context = ""
        if classifier:
            try:
                from src.segmentation import segment_into_clauses
                clauses = segment_into_clauses(raw_text)
                clause_texts = [c["text"] for c in clauses]
                if clause_texts:
                    classifier_context = classifier.format_for_risk_prompt(clause_texts)
                    combined_context = (
                        f"PRE-CLASSIFIED CLAUSE ANALYSIS:\n{classifier_context}"
                        f"\n\n---\n\n{combined_context}"
                    )
            except Exception as e:
                logger.warning(f"Clause pre-classification failed: {e}")

        language = state.get("output_language", "English")
        text_for_llm = raw_text[:12000]

        risk_result = analyze_risks(text_for_llm, combined_context, llm, language=language)
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
# CASE RESEARCH PIPELINE NODES
# =====================================================================


def case_input_node(state: GraphState) -> dict:
    """Validate and prepare case description for search."""
    logger.info("▶ case_input_node")
    desc = state.get("case_description", "").strip()
    if not desc:
        return {"error": "No case description provided.", "case_description": ""}
    return {"case_description": desc, "pipeline_mode": "case_research"}


def law_search_node(state: GraphState) -> dict:
    """Search Indian law corpus + Pile of Law using FAISS (NO LLM).

    Feature 2: Searches both law_index and pile_of_law_index.
    """
    logger.info("▶ law_search_node")
    start = time.time()

    desc = state.get("case_description", "")
    if not desc:
        return {"error": "No case description for law search."}

    try:
        # Primary: Indian law statutes
        law_idx = _get_law_index()
        results = law_idx.search_laws(desc, k=8)
        context = law_idx.get_context_for_llm(desc, k=6)

        # Label primary results as "Statute"
        for r in results:
            r.setdefault("source_type", "Statute")

        # Feature 2: Secondary search — Pile of Law (case law, contracts, etc.)
        pol_context = ""
        try:
            pol_idx = _get_pol_index()
            if pol_idx is not None:
                pol_docs = pol_idx.search(desc, k=4)
                if pol_docs:
                    pol_entries = []
                    for doc in pol_docs:
                        source_label = doc.metadata.get("label", "Case Law")
                        category = doc.metadata.get("category", "case_law")

                        # Map category to display type
                        if category == "case_law":
                            stype = "Case Law"
                        elif category == "contract":
                            stype = "Contract Precedent"
                        elif category == "legal_advice":
                            stype = "Legal Q&A"
                        else:
                            stype = "Legal Reference"

                        pol_entries.append({
                            "section": source_label,
                            "title": doc.page_content[:80] + "...",
                            "act_name": source_label,
                            "crime": "",
                            "punishment": "",
                            "jail_term": "",
                            "fine": "",
                            "bailable": "",
                            "cognizable": "",
                            "confidence": doc.metadata.get("score", 0.5),
                            "source_type": stype,
                        })
                        pol_context += f"\n[{stype}] {doc.page_content[:500]}\n"

                    results.extend(pol_entries)
                    logger.info(f"Added {len(pol_entries)} Pile of Law results")
        except Exception as e:
            logger.warning(f"Pile of Law search failed (non-fatal): {e}")

        # Merge contexts
        if pol_context:
            context += "\n\n--- ADDITIONAL LEGAL REFERENCES ---\n" + pol_context

        elapsed = int((time.time() - start) * 1000)

        return {
            "retrieved_laws": results,
            "law_context": context,
            "case_sections_found": len(results),
            "processing_time_ms": state.get("processing_time_ms", 0) + elapsed,
        }
    except Exception as e:
        logger.error(f"Law search failed: {e}")
        return {"error": f"Law search failed: {e}", "retrieved_laws": [], "law_context": ""}


def case_explain_node(state: GraphState) -> dict:
    """Use LLM (Groq) ONLY for explanation of retrieved sections."""
    logger.info("▶ case_explain_node")
    start = time.time()

    desc = state.get("case_description", "")
    context = state.get("law_context", "")
    if not context:
        return {"case_analysis": "No relevant law sections found for this case."}

    try:
        llm = get_llm()
        analysis = analyze_case(desc, context, llm)
        elapsed = int((time.time() - start) * 1000)

        return {
            "case_analysis": analysis,
            "processing_time_ms": state.get("processing_time_ms", 0) + elapsed,
        }
    except Exception as e:
        logger.error(f"Case explanation failed: {e}")
        return {"case_analysis": f"[Case analysis failed: {e}]"}


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
    output_language: str = "English",
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
        "output_language": output_language,
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


# =====================================================================
# CASE RESEARCH WORKFLOW (Pipeline 2)
# =====================================================================


def build_case_research_workflow() -> StateGraph:
    """Build the Case Research Assistant workflow.

    Graph topology:
      case_input → law_search → case_explain → output
    """
    graph = StateGraph(GraphState)

    graph.add_node("case_input_node", case_input_node)
    graph.add_node("law_search_node", law_search_node)
    graph.add_node("case_explain_node", case_explain_node)
    graph.add_node("output_node", output_node)

    graph.set_entry_point("case_input_node")

    graph.add_edge("case_input_node", "law_search_node")
    graph.add_edge("law_search_node", "case_explain_node")
    graph.add_edge("case_explain_node", "output_node")
    graph.add_edge("output_node", END)

    return graph.compile()


def run_case_research(case_description: str) -> Dict[str, Any]:
    """Run the case research pipeline.

    Args:
        case_description: Natural language description of the legal case/incident.

    Returns:
        Final state with retrieved_laws, case_analysis, etc.
    """
    workflow = build_case_research_workflow()

    initial_state: GraphState = {
        "input_data": "",
        "input_type": "text",
        "user_question": "",
        "raw_text": "",
        "chunk_count": 0,
        "is_legal": False,
        "legal_confidence": 0.0,
        "legal_classification": "",
        "detected_terms": [],
        "category_scores": {},
        "legal_explanation": "",
        "retriever_ready": False,
        "simplified_text": "",
        "risk_analysis": "",
        "qa_answer": "",
        "error": None,
        "processing_time_ms": 0,
        "output_language": "English",
        # Case research fields
        "pipeline_mode": "case_research",
        "case_description": case_description,
        "retrieved_laws": [],
        "law_context": "",
        "case_analysis": "",
        "case_sections_found": 0,
    }

    result = workflow.invoke(initial_state)
    return dict(result)
