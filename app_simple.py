"""
Streamlit App - Simplified Legal Document Analysis
Uses document-level analysis instead of clause segmentation
"""

import streamlit as st
from src.simple_pipeline import analyze_input
from src.preprocessing import read_pdf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Legal Document Simplifier | Enterprise Solution",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- PROFESSIONAL STYLES ----------------
st.markdown("""
<style>
/* Space 3D Background */
.stApp {
    background: linear-gradient(135deg, #0c0e27 0%, #1a1c3a 25%, #2d1b4e 50%, #1a1c3a 75%, #0c0e27 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    position: relative;
    overflow: hidden;
}

.stApp::before {
    content: '';
    position: fixed;
    width: 200%;
    height: 200%;
    top: -50%;
    left: -50%;
    background-image: 
        radial-gradient(2px 2px at 20% 30%, white, transparent),
        radial-gradient(2px 2px at 60% 70%, white, transparent),
        radial-gradient(1px 1px at 50% 50%, white, transparent),
        radial-gradient(1px 1px at 80% 10%, white, transparent),
        radial-gradient(2px 2px at 90% 60%, white, transparent),
        radial-gradient(1px 1px at 33% 80%, white, transparent),
        radial-gradient(1px 1px at 15% 90%, white, transparent);
    background-size: 200px 200px, 300px 300px, 150px 150px, 250px 250px, 180px 180px, 220px 220px, 350px 350px;
    background-repeat: repeat;
    animation: space-drift 200s linear infinite;
    z-index: 0;
    opacity: 0.6;
}

@keyframes space-drift {
    from {
        transform: translate(0, 0);
    }
    to {
        transform: translate(-50px, -50px);
    }
}

.stApp > * {
    position: relative;
    z-index: 1;
}

/* Professional Title Styling */
h1 {
    color: #1e3a8a !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem !important;
}

/* Professional Button Styling */
.stButton > button {
    background: #1e40af !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    transition: all 0.2s ease !important;
    cursor: pointer;
    letter-spacing: 0.3px;
}

.stButton > button:hover {
    background: #1e3a8a !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15) !important;
    transform: translateY(-1px);
}

/* Professional Metric Cards */
[data-testid="metric-container"] {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    border: 1px solid #e5e7eb;
    transition: all 0.2s ease;
}

[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

[data-testid="metric-container"] label,
[data-testid="metric-container"] div,
[data-testid="metric-container"] p {
    color: #000000 !important;
}

/* Metric values - ensure black color */
[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #000000 !important;
    font-weight: 700 !important;
}

/* Professional Expander Styling */
.streamlit-expanderHeader {
    background: white !important;
    border-radius: 8px !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.2s ease !important;
    font-weight: 600 !important;
    color: #000000 !important;
}

.streamlit-expanderHeader:hover {
    background: #f9fafb !important;
    border-color: #d1d5db !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08) !important;
}

.streamlit-expanderHeader p,
.streamlit-expanderHeader div,
.streamlit-expanderHeader span {
    color: #000000 !important;
}

/* Professional Text Area */
textarea {
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
    background: white !important;
    transition: all 0.2s ease !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    color: #000000 !important;
}

textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    outline: none !important;
}

/* Professional File Uploader */
[data-testid="stFileUploader"] {
    border-radius: 8px;
    border: 2px dashed #d1d5db !important;
    background: white;
    padding: 20px;
    transition: all 0.2s ease;
    color: #000000 !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6 !important;
    background: #f9fafb;
}

[data-testid="stFileUploader"] label {
    color: #000000 !important;
}

[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color: #ffc0cb !important;
}

/* Professional Messages */
.stSuccess, .stInfo, .stWarning {
    border-radius: 8px !important;
    border-left: 4px solid !important;
}

.stSuccess {
    color: #000000 !important;
}

.stSuccess div, .stSuccess p {
    color: #000000 !important;
}

.stInfo {
    color: #000000 !important;
}

.stInfo div, .stInfo p {
    color: #000000 !important;
}

/* Clean Divider */
hr {
    border: none;
    height: 1px;
    background: #e5e7eb;
    margin: 2rem 0;
}

/* Professional Radio Buttons */
[data-testid="stRadio"] > div {
    background: white;
    border-radius: 8px;
    padding: 10px;
    border: 1px solid #e5e7eb;
}

[data-testid="stRadio"] label {
    color: #000000 !important;
}

/* Professional Subheaders */
h2, h3 {
    color: #1f2937 !important;
    font-weight: 700 !important;
}

strong, b {
    color: #000000 !important;
}

p {
    color: #000000 !important;
}

/* Professional Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-radius: 8px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    font-weight: 600;
}

/* Black text for all labels */
label {
    color: #000000 !important;
}

/* Section Headers */
.section-header {
    background: white;
    border-left: 4px solid #1e40af;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    margin: 1.5rem 0;
}

.section-header h2 {
    margin: 0;
    color: #1f2937 !important;
    font-size: 1.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PROFESSIONAL HEADER ----------------
st.markdown("""
<div style="
    background: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    border-left: 5px solid #1e40af;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
">
    <h1 style="margin: 0; color: #111827;">⚖️ Legal Document Simplifier</h1>
    <p style="color: #4b5563; margin-top: 0.5rem; font-size: 1rem;">
        Transform complex legal text into plain English with AI-powered analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar options
st.sidebar.header("⚙️ Options")
mode = st.sidebar.radio(
    "Analysis Mode",
    ["Document-level (recommended)", "Clause-level (experimental)"]
)

st.sidebar.info(
    "**Document-level** analyzes the entire document as one piece. "
    "**Clause-level** breaks it into individual clauses (legacy feature)."
)


# =====================================================
# INPUT SECTION
# =====================================================
st.markdown("""
<div class="section-header">
    <h2>Document Input</h2>
    <p style="color: #6b7280; margin: 0.5rem 0 0 0; font-size: 0.95rem;">Upload a PDF or paste your legal document text for simplified analysis</p>
</div>
""", unsafe_allow_html=True)

# Main input area
tab1, tab2 = st.tabs(["📝 Text Input", "📄 PDF Upload"])

with tab1:
    user_text = st.text_area(
        "Legal Document Text",
        height=220,
        placeholder="Paste contract, agreement, or legal clause here..."
    )
    
    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if user_text.strip():
            with st.spinner("Analyzing document..."):
                df = analyze_input(user_text, input_type="text")
                st.session_state["results"] = df
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    uploaded_file = st.file_uploader(
        "Upload PDF Document", 
        type=["pdf"],
        help="Select a PDF file to analyze"
    )
    
    if uploaded_file and st.button("🔍 Analyze PDF", type="primary", use_container_width=True):
        with st.spinner("Processing PDF..."):
            # Save temporarily and process
            with open("temp_upload.pdf", "wb") as f:
                f.write(uploaded_file.read())
            
            df = analyze_input("temp_upload.pdf", input_type="pdf")
            st.session_state["results"] = df



# =====================================================
# RESULTS SECTION
# =====================================================
if "results" in st.session_state:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <h2>Analysis Results</h2>
        <p style="color: #6b7280; margin: 0.5rem 0 0 0; font-size: 0.95rem;">Simplified analysis of your legal document</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state["results"]
    result = df.iloc[0]
    
    if result["status"] == "non_legal":
        # Non-legal document
        st.warning(result["message"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Legal Confidence", f"{result['confidence']*100:.0f}%")
        with col2:
            st.metric("Status", "Not Legal")
            
        if result["detected_terms"]:
            st.info(f"**Detected terms:** {result['detected_terms']}")
            
    else:
        # Legal document - show results
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "✅ Legal")
        with col2:
            st.metric("Confidence", f"{result['confidence']*100:.0f}%")
        with col3:
            st.metric("Analysis", "Complete")
        
        if result["detected_terms"]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"**Detected legal terms:** {result['detected_terms']}")
        
        # Original text
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📜 Original Legal Text", expanded=False):
            st.markdown(f"""
            <div style="background: #f9fafb; padding: 1.25rem; border-radius: 8px; 
                        border-left: 3px solid #3b82f6; color: #1f2937; line-height: 1.6;">
                {result["original_text"]}
            </div>
            """, unsafe_allow_html=True)
        
        # Simplified version
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### ✨ Simplified Explanation")
        st.success(result["simplified_text"])
        
        # Download option
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Simplified Text",
            data=result["simplified_text"],
            file_name="simplified_legal_text.txt",
            mime="text/plain",
            use_container_width=True
        )

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 1.5rem; background: white; 
            border-radius: 8px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); border-top: 2px solid #e5e7eb;">
    <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">
        💡 This tool uses AI to simplify legal language. Always consult a legal professional for official advice.
    </p>
    <p style="color: #9ca3af; font-size: 0.85rem; margin-top: 0.5rem;">
        Powered by Advanced AI & Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

