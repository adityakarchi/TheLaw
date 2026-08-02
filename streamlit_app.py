"""LEGAL AUTONOMOUS AI PLATFORM — Streamlit Interface.

TWO autonomous products:
  1. Legal Document Simplifier — Upload PDF / paste → detect → simplify → risk → Q&A
  2. Legal Case Research Assistant (Lawyer AI) — Describe case → FAISS law search → LLM explanation
"""

import streamlit as st
import sys
import os
import tempfile
import time
import logging
from pathlib import Path

# Path setup
sys.path.insert(0, str(Path(__file__).parent))

from src.graph.workflow import run_full_analysis, run_qa, reset_retriever, run_case_research
from src.utils.config import check_api_connection
from src.utils.s3_storage import (
    upload_pdf_to_s3,
    upload_text_to_s3,
    list_uploaded_documents,
    is_s3_configured,
    get_s3_console_url,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config

st.set_page_config(
    page_title="Legal AI Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional dark theme CSS

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

/* Clean analysis section styling */
[data-testid="stExpander"] details {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 10px;
}
[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #FAFAFA;
}
[data-testid="stMarkdownContainer"] h4 {
    color: #667eea;
    margin-top: 1rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #333;
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

# Session state initialization

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "case_result" not in st.session_state:
    st.session_state.case_result = None
if "case_history" not in st.session_state:
    st.session_state.case_history = []
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "📄 Document Simplifier"

# Sidebar

with st.sidebar:
    st.markdown("## ⚖️ Legal AI Platform")
    st.markdown("---")

    st.markdown("### 🔀 Select Product")
    mode = st.radio(
        "Choose your tool:",
        ["📄 Document Simplifier", "🔍 Case Research Assistant"],
        index=0 if st.session_state.app_mode == "📄 Document Simplifier" else 1,
        key="mode_selector",
        label_visibility="collapsed",
    )
    st.session_state.app_mode = mode

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

    if mode == "📄 Document Simplifier":
        st.markdown("""
        **Pipeline 1: Document Simplifier**
        - Upload PDF / paste text
        - Local legal detection (no LLM)
        - FAISS vector search
        - Groq LLM for simplification
        - Local risk scoring
        - Interactive Q&A
        """)
    else:
        st.markdown("""
        **Pipeline 2: Case Research**
        - Describe case in natural language
        - FAISS search over Indian laws
        - IPC, CrPC, Evidence Act corpus
        - Groq LLM for explanation ONLY
        - Applicable sections + punishment
        - Winning probability estimate
        """)

    st.markdown("---")

    st.markdown("### ☁️ S3 Storage")
    if is_s3_configured():
        st.success("✅ S3 Connected")
        if st.button("📋 View Uploads", use_container_width=True):
            docs = list_uploaded_documents()
            if docs:
                for d in docs[:5]:
                    st.caption(f"📄 {d['filename']} ({d['size_kb']} KB) — {d['uploaded_at']}")
            else:
                st.info("No documents uploaded yet.")
    else:
        st.warning("⚠️ S3 not configured\nAdd AWS keys to .env")

    st.markdown("---")
    st.markdown("### 📖 Tech Stack")
    st.markdown("""
    - **Embeddings**: `sentence-transformers`
    - **Vector DB**: `FAISS`
    - **LLM**: `Groq (Llama 3.1)`
    - **Workflow**: `LangGraph`
    - **Chains**: `LangChain`
    - **Storage**: `AWS S3`
    """)

    st.markdown("---")

    st.markdown("""
    ⚠️ *For informational purposes only.
    Always consult a legal professional.*
    """)

    # Clear buttons
    if mode == "📄 Document Simplifier" and st.session_state.document_loaded:
        if st.button("🗑️ Clear Document", use_container_width=True):
            reset_retriever()
            st.session_state.analysis_result = None
            st.session_state.chat_history = []
            st.session_state.document_loaded = False
            st.rerun()

    if mode == "🔍 Case Research Assistant" and st.session_state.case_result:
        if st.button("🗑️ Clear Case Results", use_container_width=True):
            st.session_state.case_result = None
            st.session_state.case_history = []
            st.rerun()


# Hero header

if st.session_state.app_mode == "📄 Document Simplifier":
    st.markdown("""
    <div class="hero">
        <h1>📄 Legal Document Simplifier</h1>
        <p>RAG-powered legal document analysis — simplify, assess risk, and ask questions</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero">
        <h1>🔍 Legal Case Research Assistant</h1>
        <p>Describe your legal case — AI searches Indian laws (IPC, CrPC, Evidence Act) and explains applicable sections</p>
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
# MODE: DOCUMENT SIMPLIFIER
# =====================================================================

if st.session_state.app_mode == "📄 Document Simplifier":

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

    # Trigger analysis

    if analyze_text_btn:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            execute_analysis(user_text, "text")
            # Upload text to S3 after successful analysis
            if is_s3_configured():
                s3_key = upload_text_to_s3(user_text, "pasted_contract.txt")
                if s3_key:
                    st.toast("☁️ Saved to S3: pasted_contract.txt", icon="✅")

    if analyze_pdf_btn and uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file_bytes = uploaded_file.read()
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            execute_analysis(tmp_path, "pdf")
            # Upload to S3 after successful analysis
            if is_s3_configured():
                s3_key = upload_pdf_to_s3(file_bytes, uploaded_file.name)
                if s3_key:
                    st.toast(f"☁️ Saved to S3: {uploaded_file.name}", icon="✅")
        finally:
            os.unlink(tmp_path)

    # Results display

    result = st.session_state.analysis_result

    if result:
        st.markdown("---")

        if result.get("error"):
            st.error(f"⚠️ {result['error']}")

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

        explanation = result.get("legal_explanation", "")
        if explanation:
            st.info(explanation)

        terms = result.get("detected_terms", [])
        if terms:
            with st.expander("🏷️ Detected Legal Terms", expanded=False):
                render_terms(terms)

        if not result.get("is_legal"):
            st.warning("This does not appear to be a legal document. "
                        "Upload a contract, agreement, or terms of service to get full analysis.")

            with st.expander("📄 Input Text Preview"):
                raw = result.get("raw_text", "")
                st.text(raw[:2000] + ("…" if len(raw) > 2000 else ""))

        else:
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

            risk = result.get("risk_analysis", "")
            if risk and not risk.startswith("["):
                st.markdown("### ⚠️ Risk Analysis")
                st.markdown(risk)
            elif risk:
                st.warning(risk)

            with st.expander("📜 Original Legal Text", expanded=False):
                raw = result.get("raw_text", "")
                st.text(raw[:8000] + ("…" if len(raw) > 8000 else ""))

            proc_ms = result.get("processing_time_ms", 0)
            if proc_ms:
                st.caption(f"⏱️ Total processing time: {proc_ms / 1000:.1f}s")

        # Contract Q&A

        if result.get("is_legal") and st.session_state.document_loaded:
            st.markdown("---")
            st.markdown("### 💬 Ask Questions About This Contract")

            for entry in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(entry["question"])
                with st.chat_message("assistant"):
                    st.markdown(entry["answer"])

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


# =====================================================================
# MODE: CASE RESEARCH ASSISTANT
# =====================================================================

else:

    st.markdown("### 📝 Describe Your Legal Case")
    st.markdown("""
    <div class="card">
        <p style="color:#A0A0A0; margin:0;">
        Describe the incident or legal situation in natural language. The AI will search
        through <strong>IPC</strong>, <strong>CrPC</strong>, and <strong>Indian Evidence Act</strong>
        to find applicable sections, punishments, and provide legal guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    case_description = st.text_area(
        "Case Description:",
        height=200,
        placeholder=(
            "Example: A person was stabbed during a robbery attempt at night. "
            "The victim survived but suffered serious injuries. The attacker also "
            "threatened the victim's family. What sections apply?\n\n"
            "Or: Someone posted defamatory content about a public figure on social media..."
        ),
        key="case_input",
    )

    # Suggested case examples
    st.markdown("**💡 Quick Examples:**")
    example_cols = st.columns(3)
    examples = [
        ("🔪 Robbery & Assault", "A gang of 3 people robbed a jewellery shop at night using weapons. They assaulted the shop owner causing grievous injuries. One of the accused was a minor."),
        ("💻 Cyber Crime", "Someone hacked into another person's bank account and transferred money. They also used fake identity documents for KYC verification."),
        ("🏠 Domestic Violence", "A woman is facing continuous physical and mental abuse from her husband and in-laws over dowry demands. She wants to file a case."),
    ]

    for i, (label, desc) in enumerate(examples):
        with example_cols[i]:
            if st.button(label, key=f"case_ex_{i}", use_container_width=True):
                st.session_state["case_prefill"] = desc
                st.rerun()

    # Apply prefill
    if "case_prefill" in st.session_state:
        case_description = st.session_state.pop("case_prefill")

    analyze_case_btn = st.button(
        "🔍 Research Applicable Laws",
        type="primary",
        use_container_width=True,
        key="analyze_case",
    )

    # Trigger case research

    if analyze_case_btn:
        if not case_description or not case_description.strip():
            st.warning("⚠️ Please describe the legal case or incident.")
        else:
            with st.spinner("🔄 Searching Indian law corpus (FAISS) → Analyzing with LLM…"):
                start = time.time()
                case_result = run_case_research(case_description.strip())
                total_ms = int((time.time() - start) * 1000)
                case_result["processing_time_ms"] = total_ms

            st.session_state.case_result = case_result
            st.session_state.case_history.append({
                "description": case_description.strip(),
                "result": case_result,
            })

    # Case results display

    case_result = st.session_state.case_result

    if case_result:
        st.markdown("---")

        if case_result.get("error"):
            st.error(f"⚠️ {case_result['error']}")

        # Metrics row
        st.markdown("### 📊 Research Results")

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Sections Found", case_result.get("case_sections_found", 0))
        with mcol2:
            proc_ms = case_result.get("processing_time_ms", 0)
            st.metric("Processing Time", f"{proc_ms / 1000:.1f}s")
        with mcol3:
            n_laws = len(case_result.get("retrieved_laws", []))
            st.metric("Laws Retrieved", n_laws)

        # Retrieved law sections
        retrieved_laws = case_result.get("retrieved_laws", [])
        if retrieved_laws:
            st.markdown("### 📜 Applicable Law Sections")

            for i, law in enumerate(retrieved_laws):
                section = law.get("section", "N/A")
                title = law.get("title", "N/A")
                act = law.get("act_name", "N/A")
                score = law.get("confidence", 0)
                crime = law.get("crime", "")
                punishment = law.get("punishment", "")
                jail = law.get("jail_term", "")
                fine = law.get("fine", "")
                bailable = law.get("bailable", "")
                cognizable = law.get("cognizable", "")

                pct = int(score * 100) if score <= 1 else int(score)

                with st.expander(
                    f"**Section {section}** — {title} ({act}) | Relevance: {pct}%",
                    expanded=(i < 3),
                ):
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown(f"**🏛️ Act:** {act}")
                        st.markdown(f"**📌 Section:** {section}")
                        st.markdown(f"**⚖️ Crime:** {crime}")
                        st.markdown(f"**🔒 Cognizable:** {cognizable}")
                    with cols[1]:
                        st.markdown(f"**⚠️ Punishment:** {punishment}")
                        st.markdown(f"**🏢 Jail Term:** {jail}")
                        st.markdown(f"**💰 Fine:** {fine}")
                        st.markdown(f"**🔓 Bailable:** {bailable}")

                    render_confidence_bar(score if score <= 1 else score / 100)

        # LLM Analysis
        analysis = case_result.get("case_analysis", "")
        if analysis and not analysis.startswith("["):
            st.markdown("### 🧠 AI Legal Analysis")

            # Parse and render each section cleanly
            sections = analysis.split("\n## ")

            for idx_s, section in enumerate(sections):
                if idx_s == 0 and not section.startswith("##"):
                    # First block (may be intro or "## Case Analysis" without leading \n)
                    section_text = section.lstrip("# ").strip()
                    if section_text:
                        # Check if it starts with a heading
                        if section.strip().startswith("##"):
                            heading = section.strip().split("\n", 1)
                            title = heading[0].lstrip("# ").strip()
                            body = heading[1].strip() if len(heading) > 1 else ""
                            st.markdown(f"#### {title}")
                            if body:
                                st.markdown(body)
                        else:
                            st.markdown(section.strip())
                else:
                    # Subsequent sections: split heading from body
                    parts = section.split("\n", 1)
                    title = parts[0].lstrip("# ").strip()
                    body = parts[1].strip() if len(parts) > 1 else ""

                    # Color-coded containers per section type
                    if "applicable" in title.lower():
                        icon = "📜"
                    elif "confidence" in title.lower():
                        icon = "📊"
                    elif "recommended" in title.lower() or "action" in title.lower():
                        icon = "✅"
                    elif "winning" in title.lower() or "probability" in title.lower():
                        icon = "🎯"
                    elif "disclaimer" in title.lower():
                        icon = "⚠️"
                    elif "case" in title.lower() and "analysis" in title.lower():
                        icon = "🔍"
                    else:
                        icon = "📌"

                    with st.container():
                        st.markdown(f"#### {icon} {title}")

                        if body:
                            # Handle sub-sections (### headings for individual laws)
                            sub_sections = body.split("\n### ")

                            if len(sub_sections) > 1:
                                # First part before any ### 
                                intro = sub_sections[0].strip()
                                if intro:
                                    st.markdown(intro)

                                # Each law section as an expander
                                for sub in sub_sections[1:]:
                                    sub_parts = sub.split("\n", 1)
                                    sub_title = sub_parts[0].strip()
                                    sub_body = sub_parts[1].strip() if len(sub_parts) > 1 else ""

                                    with st.expander(f"⚖️ {sub_title}", expanded=True):
                                        if sub_body:
                                            st.markdown(sub_body)
                            else:
                                st.markdown(body)

                        st.markdown("---")

            st.download_button(
                "📥 Download Case Analysis",
                data=analysis,
                file_name="case_analysis_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        elif analysis:
            st.warning(analysis)

    # Case history

    history = st.session_state.case_history
    if len(history) > 1:
        st.markdown("---")
        with st.expander("📋 Previous Case Searches", expanded=False):
            for i, entry in enumerate(reversed(history[:-1])):
                st.markdown(f"**Case {len(history) - 1 - i}:** {entry['description'][:100]}…")
                n = entry['result'].get('case_sections_found', 0)
                st.caption(f"Found {n} applicable sections")


# Footer

st.markdown("""
<div class="footer">
    <p>⚖️ <strong>Legal Autonomous AI Platform</strong> | RAG + LangGraph + LangChain + FAISS + Groq</p>
    <p style="opacity:0.6; font-size:0.85rem">
        Built by Aditya & AI ❤️ — For informational purposes only. Consult a legal professional.
    </p>
</div>
""", unsafe_allow_html=True)
