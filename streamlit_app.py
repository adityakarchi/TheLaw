"""Legal AI Assistant — Production Streamlit Interface.

Full-featured UI:
  - Upload PDF or paste text
  - Detect legal documents with confidence scoring
  - Simplify to plain English via RAG + LLM
  - Risk analysis with severity ratings
  - Interactive contract Q&A
  - Download results
"""

import streamlit as st
import sys
import os
import tempfile
import time
import logging
from pathlib import Path

# ── Path Setup ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.graph.workflow import run_full_analysis, run_qa, reset_retriever
from src.utils.config import check_api_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional Dark Theme CSS ──────────────────────────────────────

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #FAFAFA; }
.main .block-container { padding-top: 1.5rem; max-width: 1200px; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    border-left: 5px solid #667eea;
    border: 1px solid #333;
}
.hero h1 { color: #FAFAFA; font-size: 2.4rem; margin: 0; font-weight: 700; }
.hero p { color: #A0A0A0; margin-top: 0.5rem; font-size: 1.05rem; }

.card {
    background: #1E1E1E;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid #333;
}

.badge-legal {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: #000; padding: 6px 18px; border-radius: 20px;
    font-weight: 600; font-size: 13px; display: inline-block;
}
.badge-not-legal {
    background: linear-gradient(135deg, #eb3349, #f45c43);
    color: #fff; padding: 6px 18px; border-radius: 20px;
    font-weight: 600; font-size: 13px; display: inline-block;
}

.term-pill {
    background: #667eea; color: white;
    padding: 3px 10px; border-radius: 12px;
    font-size: 12px; margin: 2px; display: inline-block;
}

.result-block {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid #667eea;
    margin: 0.75rem 0;
    color: #e0e0e0;
    line-height: 1.7;
}

.simplified-block {
    background: rgba(17, 153, 142, 0.15);
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid #11998e;
    margin: 0.75rem 0;
    color: #e0e0e0;
    line-height: 1.7;
}

.risk-block {
    background: rgba(235, 51, 73, 0.1);
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid #eb3349;
    margin: 0.75rem 0;
    color: #e0e0e0;
    line-height: 1.7;
}

.qa-block {
    background: rgba(102, 126, 234, 0.12);
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid #667eea;
    margin: 0.75rem 0;
    color: #e0e0e0;
    line-height: 1.7;
}

.conf-bar { background: #333; border-radius: 8px; height: 8px; overflow: hidden; margin-top: 4px; }
.conf-fill { height: 100%; border-radius: 8px; transition: width .4s ease; }
.conf-high { background: linear-gradient(90deg, #11998e, #38ef7d); }
.conf-med  { background: linear-gradient(90deg, #f7971e, #ffd200); }
.conf-low  { background: linear-gradient(90deg, #eb3349, #f45c43); }

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; border: none; border-radius: 10px;
    padding: 0.7rem 1.5rem; font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102,126,234,0.6);
}

.stTextArea textarea {
    border-radius: 12px; border: 2px solid #333;
    background-color: #262730; color: #FAFAFA;
}
.stTextArea textarea:focus { border-color: #667eea; }

[data-testid="stFileUploader"] {
    border: 2px dashed #667eea; border-radius: 12px;
    padding: 1.5rem; background: rgba(102,126,234,0.08);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}

[data-testid="metric-container"] {
    background: #262730; padding: 0.8rem; border-radius: 12px;
}

.footer { text-align: center; padding: 2rem; color: #888; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ──────────────────────────────────────

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    st.markdown("### 🔌 API Status")
    if st.button("Check Connection", use_container_width=True):
        with st.spinner("Checking…"):
            ok, msg = check_api_connection()
            if ok:
                st.success("✅ Connected to Groq API")
            else:
                st.error(f"❌ {msg}")

    st.markdown("---")
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    **RAG Pipeline**
    - Embeddings: `sentence-transformers`
    - Vector DB: `FAISS`
    - LLM: `Groq (Llama 3.1)`

    **Workflow**: `LangGraph`
    **Chains**: `LangChain`
    """)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **Legal AI Assistant** uses RAG + LangGraph to:
    1. 🔍 Detect legal documents
    2. ✨ Simplify to plain English
    3. ⚠️ Analyze risky clauses
    4. 💬 Answer contract questions

    ---
    ⚠️ *For informational purposes only.
    Always consult a legal professional.*
    """)

    if st.session_state.document_loaded:
        st.markdown("---")
        st.success("📄 Document loaded")
        if st.button("🗑️ Clear Document", use_container_width=True):
            reset_retriever()
            st.session_state.analysis_result = None
            st.session_state.chat_history = []
            st.session_state.document_loaded = False
            st.rerun()


# ── Hero Header ───────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>⚖️ Legal AI Assistant</h1>
    <p>RAG-powered legal document analysis — simplify, assess risk, and ask questions</p>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def render_confidence_bar(confidence: float):
    pct = int(confidence * 100)
    cls = "conf-high" if confidence >= 0.7 else "conf-med" if confidence >= 0.4 else "conf-low"
    st.markdown(f"""
    <div class="conf-bar"><div class="conf-fill {cls}" style="width:{pct}%"></div></div>
    """, unsafe_allow_html=True)


def render_terms(terms: list):
    if not terms:
        return
    pills = " ".join(f'<span class="term-pill">{t}</span>' for t in terms[:20])
    st.markdown(f'<div style="margin:0.5rem 0">{pills}</div>', unsafe_allow_html=True)


def execute_analysis(input_data, input_type: str):
    """Execute the LangGraph workflow and cache the result."""
    with st.spinner("🔄 Running full analysis pipeline (detect → embed → simplify → risk)…"):
        start = time.time()
        result = run_full_analysis(input_data, input_type)
        total_ms = int((time.time() - start) * 1000)
        result["processing_time_ms"] = total_ms

    st.session_state.analysis_result = result
    st.session_state.document_loaded = result.get("is_legal", False)
    st.session_state.chat_history = []
    return result


# =====================================================================
# INPUT SECTION
# =====================================================================

st.markdown("### 📄 Upload Your Document")
tab_text, tab_pdf = st.tabs(["📝 Paste Text", "📁 Upload PDF"])

with tab_text:
    user_text = st.text_area(
        "Enter legal document text:",
        height=250,
        placeholder=(
            "Paste your contract, agreement, terms of service, or any legal document here…\n\n"
            "Example: This Agreement is entered into as of the Effective Date between "
            "Company Inc. (\"Company\") and the Client…"
        ),
    )
    analyze_text_btn = st.button("🔍 Analyze Document", type="primary",
                                  use_container_width=True, key="analyze_text")

with tab_pdf:
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"],
                                      help="PDF up to 10 MB")
    if uploaded_file:
        st.info(f"📄 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
    analyze_pdf_btn = st.button("🔍 Analyze PDF", type="primary",
                                 use_container_width=True, key="analyze_pdf",
                                 disabled=not uploaded_file)


# ── Trigger Analysis ──────────────────────────────────────────────────

if analyze_text_btn:
    if not user_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        execute_analysis(user_text, "text")

if analyze_pdf_btn and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        execute_analysis(tmp_path, "pdf")
    finally:
        os.unlink(tmp_path)


# =====================================================================
# RESULTS DISPLAY
# =====================================================================

result = st.session_state.analysis_result

if result:
    st.markdown("---")

    # ── Error Check ───────────────────────────────────────────────────
    if result.get("error"):
        st.error(f"⚠️ {result['error']}")

    # ── Detection Results ─────────────────────────────────────────────
    st.markdown("### 📊 Analysis Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if result.get("is_legal"):
            st.markdown('<span class="badge-legal">✅ LEGAL DOCUMENT</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-not-legal">❌ NOT LEGAL</span>',
                        unsafe_allow_html=True)

    with col2:
        conf = result.get("legal_confidence", 0)
        st.metric("Confidence", f"{int(conf * 100)}%")
        render_confidence_bar(conf)

    with col3:
        st.metric("Legal Terms", len(result.get("detected_terms", [])))

    with col4:
        st.metric("Chunks Indexed", result.get("chunk_count", 0))

    # ── Explanation ───────────────────────────────────────────────────
    explanation = result.get("legal_explanation", "")
    if explanation:
        st.info(explanation)

    # ── Detected Terms ────────────────────────────────────────────────
    terms = result.get("detected_terms", [])
    if terms:
        with st.expander("🏷️ Detected Legal Terms", expanded=False):
            render_terms(terms)

    # ── NOT LEGAL → stop here ─────────────────────────────────────────
    if not result.get("is_legal"):
        st.warning("This does not appear to be a legal document. "
                    "Upload a contract, agreement, or terms of service to get full analysis.")

        with st.expander("📄 Input Text Preview"):
            raw = result.get("raw_text", "")
            st.text(raw[:2000] + ("…" if len(raw) > 2000 else ""))

    else:
        # ── Simplified Version ────────────────────────────────────────
        simplified = result.get("simplified_text", "")
        if simplified and not simplified.startswith("["):
            st.markdown("### ✨ Simplified Version")
            st.markdown(simplified)

            st.download_button(
                "📥 Download Simplified Text",
                data=simplified,
                file_name="simplified_legal_document.txt",
                mime="text/plain",
                use_container_width=True,
            )
        elif simplified:
            st.warning(simplified)

        # ── Risk Analysis ─────────────────────────────────────────────
        risk = result.get("risk_analysis", "")
        if risk and not risk.startswith("["):
            st.markdown("### ⚠️ Risk Analysis")
            st.markdown(risk)
        elif risk:
            st.warning(risk)

        # ── Original Text ─────────────────────────────────────────────
        with st.expander("📜 Original Legal Text", expanded=False):
            raw = result.get("raw_text", "")
            st.text(raw[:8000] + ("…" if len(raw) > 8000 else ""))

        # ── Processing Time ───────────────────────────────────────────
        proc_ms = result.get("processing_time_ms", 0)
        if proc_ms:
            st.caption(f"⏱️ Total processing time: {proc_ms / 1000:.1f}s")


    # ==================================================================
    # CONTRACT Q&A SECTION
    # ==================================================================

    if result.get("is_legal") and st.session_state.document_loaded:
        st.markdown("---")
        st.markdown("### 💬 Ask Questions About This Contract")

        # Show chat history
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.markdown(entry["answer"])

        # Chat input
        question = st.chat_input("Ask a question about the contract…")

        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching document…"):
                    answer = run_qa(question)
                st.markdown(answer)

            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
            })

        # Suggested questions
        st.markdown("**💡 Try asking:**")
        suggested = [
            "What are the main obligations?",
            "Can they terminate anytime?",
            "What are my risks?",
            "Any penalty clauses?",
            "What if there's a breach?",
        ]
        cols = st.columns(len(suggested))
        for i, sq in enumerate(suggested):
            with cols[i]:
                if st.button(sq, key=f"sq_{i}", use_container_width=True):
                    with st.spinner("Searching document…"):
                        answer = run_qa(sq)
                    st.session_state.chat_history.append({
                        "question": sq,
                        "answer": answer,
                    })
                    st.rerun()


# ── Footer ────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    <p>⚖️ <strong>Legal AI Assistant</strong> | RAG + LangGraph + LangChain + Groq</p>
    <p style="opacity:0.6; font-size:0.85rem">
        Built by Aditya & AI ❤️ — For informational purposes only. Consult a legal professional.
    </p>
</div>
""", unsafe_allow_html=True)
