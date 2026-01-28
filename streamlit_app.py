"""Legal Document Simplifier - Web interface."""

import streamlit as st
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simple_pipeline import analyze_input, quick_check
from src.preprocessing import read_pdf
from src.simplification import check_api_connection

# Page configuration

st.set_page_config(
    page_title="Legal Document Simplifier",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling

st.markdown("""
<style>
/* Modern Professional Theme */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Card styling */
.card {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
}

/* Header styling */
.main-header {
    background: white;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    border-left: 5px solid #667eea;
}

.main-header h1 {
    color: #1a1a2e;
    font-size: 2.5rem;
    margin: 0;
    font-weight: 700;
}

.main-header p {
    color: #4a5568;
    margin-top: 0.5rem;
    font-size: 1.1rem;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

/* Status badges */
.status-legal {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    padding: 8px 20px;
    border-radius: 25px;
    font-weight: 600;
    font-size: 14px;
    display: inline-block;
}

.status-not-legal {
    background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    color: white;
    padding: 8px 20px;
    border-radius: 25px;
    font-weight: 600;
    font-size: 14px;
    display: inline-block;
}

/* Confidence meter */
.confidence-bar {
    background: #e0e0e0;
    border-radius: 10px;
    height: 10px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.5s ease;
}

.confidence-high { background: linear-gradient(90deg, #11998e, #38ef7d); }
.confidence-medium { background: linear-gradient(90deg, #f7971e, #ffd200); }
.confidence-low { background: linear-gradient(90deg, #eb3349, #f45c43); }

/* Text areas */
.stTextArea textarea {
    border-radius: 12px;
    border: 2px solid #e0e0e0;
    font-size: 1rem;
    transition: border-color 0.3s ease;
}

.stTextArea textarea:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #667eea;
    border-radius: 12px;
    padding: 2rem;
    background: rgba(102, 126, 234, 0.05);
}

/* Result boxes */
.result-box {
    background: #f8fafc;
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
}

.simplified-box {
    background: linear-gradient(135deg, rgba(17, 153, 142, 0.1) 0%, rgba(56, 239, 125, 0.1) 100%);
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid #11998e;
    margin: 1rem 0;
}

/* Terms pill */
.term-pill {
    background: #667eea;
    color: white;
    padding: 4px 12px;
    border-radius: 15px;
    font-size: 12px;
    margin: 2px;
    display: inline-block;
}

/* Metric styling */
[data-testid="metric-container"] {
    background: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

/* Expander styling */
.streamlit-expanderHeader {
    background: white;
    border-radius: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}

[data-testid="stSidebar"] .stMarkdown {
    color: white;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem;
    color: white;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# Sidebar

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    
    # API Status Check
    st.markdown("### 🔌 API Status")
    if st.button("Check Connection", use_container_width=True):
        with st.spinner("Checking..."):
            connected, message = check_api_connection()
            if connected:
                st.success("✅ Connected")
            else:
                st.error(f"❌ {message}")
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **Legal Document Simplifier** uses AI to:
    
    1. 🔍 Detect legal documents
    2. 📝 Extract key terms
    3. ✨ Simplify complex language
    
    ---
    
    **Supported Inputs:**
    - Plain text
    - PDF files
    
    ---
    
    ⚠️ *This tool is for informational purposes. 
    Always consult a legal professional.*
    """)

# Main header

st.markdown("""
<div class="main-header">
    <h1>⚖️ Legal Document Simplifier</h1>
    <p>Transform complex legal text into plain English with AI-powered analysis</p>
</div>
""", unsafe_allow_html=True)

# INPUT SECTION

st.markdown("### 📄 Upload Your Document")

tab1, tab2 = st.tabs(["📝 Paste Text", "📁 Upload PDF"])

with tab1:
    st.markdown("""
    <div class="card">
    """, unsafe_allow_html=True)
    
    user_text = st.text_area(
        "Enter your legal document text:",
        height=250,
        placeholder="Paste your contract, agreement, terms of service, or any legal document here...\n\nExample: This Agreement is entered into as of the date last signed below (the \"Effective Date\") between Company Inc., a Delaware corporation (\"Company\"), and the individual or entity identified below (\"Client\")..."
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_text_btn = st.button(
            "🔍 Analyze Document",
            type="primary",
            use_container_width=True,
            key="analyze_text"
        )
    with col2:
        quick_check_btn = st.button(
            "⚡ Quick Check",
            use_container_width=True,
            key="quick_check",
            help="Check if document is legal without simplification"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div class="card">
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Supported format: PDF (up to 10MB)"
    )
    
    if uploaded_file:
        st.info(f"📄 **File:** {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
    
    analyze_pdf_btn = st.button(
        "🔍 Analyze PDF",
        type="primary",
        use_container_width=True,
        key="analyze_pdf",
        disabled=not uploaded_file
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Processing logic

def display_results(df):
    """Display analysis results in a professional format."""
    
    result = df.iloc[0]
    status = result.get("status", "error")
    
    st.markdown("---")
    st.markdown("### 📊 Analysis Results")
    
    # Status and confidence row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if status == "legal":
            st.markdown('<span class="status-legal">✅ LEGAL DOCUMENT</span>', unsafe_allow_html=True)
        elif status == "non_legal":
            st.markdown('<span class="status-not-legal">❌ NOT A LEGAL DOCUMENT</span>', unsafe_allow_html=True)
        else:
            st.error("⚠️ Error occurred")
    
    with col2:
        confidence = result.get("confidence", 0)
        confidence_pct = int(confidence * 100)
        st.metric("Confidence Score", f"{confidence_pct}%")
        
        # Confidence bar
        if confidence >= 0.7:
            bar_class = "confidence-high"
        elif confidence >= 0.4:
            bar_class = "confidence-medium"
        else:
            bar_class = "confidence-low"
        
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill {bar_class}" style="width: {confidence_pct}%"></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        terms = result.get("detected_terms", "")
        term_count = len(terms.split(", ")) if terms else 0
        st.metric("Legal Terms Found", term_count)
    
    # Detected terms
    if terms:
        st.markdown("#### 🏷️ Detected Legal Terms")
        terms_html = " ".join([f'<span class="term-pill">{term.strip()}</span>' for term in terms.split(", ")])
        st.markdown(f'<div style="margin: 1rem 0;">{terms_html}</div>', unsafe_allow_html=True)
    
    # Handle different statuses
    if status == "non_legal":
        st.warning(result.get("message", "This does not appear to be a legal document."))
        
        # Show preview of text
        original = result.get("original_text", "")
        if original:
            with st.expander("📄 Input Text Preview"):
                st.text(original[:1000] + ("..." if len(original) > 1000 else ""))
    
    elif status == "error":
        st.error(result.get("message", "An error occurred during analysis."))
    
    else:  # Legal document
        # Original text
        with st.expander("📜 Original Legal Text", expanded=False):
            original = result.get("original_text", "")
            st.markdown(f"""
            <div class="result-box">
                {original[:5000]}{'...' if len(original) > 5000 else ''}
            </div>
            """, unsafe_allow_html=True)
        
        # Simplified text
        st.markdown("#### ✨ Simplified Version")
        simplified = result.get("simplified_text", "")
        
        if simplified and not simplified.startswith("[Simplification unavailable"):
            st.markdown(f"""
            <div class="simplified-box">
                {simplified}
            </div>
            """, unsafe_allow_html=True)
            
            # Download button
            st.download_button(
                label="📥 Download Simplified Text",
                data=simplified,
                file_name="simplified_legal_document.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.warning(simplified or "Simplification not available.")


def display_quick_check(result):
    """Display quick check results."""
    st.markdown("---")
    st.markdown("### ⚡ Quick Check Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if result["is_legal"]:
            st.success("✅ This appears to be a **legal document**")
        else:
            st.warning("❌ This does **not** appear to be a legal document")
    
    with col2:
        confidence = int(result["confidence"] * 100)
        st.metric("Confidence", f"{confidence}%")
    
    # Show detected terms
    if result["detected_terms"]:
        st.markdown("**Detected terms:** " + ", ".join(result["detected_terms"][:10]))
    
    st.info(f"**Classification:** {result['classification'].replace('_', ' ').title()}")


# Process text input
if analyze_text_btn and user_text.strip():
    with st.spinner("🔍 Analyzing document..."):
        try:
            df = analyze_input(user_text, input_type="text")
            display_results(df)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

elif analyze_text_btn and not user_text.strip():
    st.warning("⚠️ Please enter some text to analyze.")

# Quick check
if quick_check_btn and user_text.strip():
    with st.spinner("⚡ Running quick check..."):
        try:
            result = quick_check(user_text)
            display_quick_check(result)
        except Exception as e:
            st.error(f"Quick check failed: {str(e)}")

# Process PDF input
if analyze_pdf_btn and uploaded_file:
    with st.spinner("🔍 Processing PDF..."):
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            df = analyze_input(tmp_path, input_type="pdf")
            display_results(df)
            
            # Cleanup
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"PDF analysis failed: {str(e)}")

# Footer

st.markdown("""
<div class="footer">
    <p>💡 <strong>Legal Document Simplifier</strong> | Built with AI & ❤️</p>
    <p style="opacity: 0.7; font-size: 0.9rem;">
        This tool uses advanced AI to help understand legal language. 
        Always consult a qualified legal professional for official advice.
    </p>
</div>
""", unsafe_allow_html=True)
