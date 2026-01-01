import streamlit as st
import sys

sys.path.append("..")

from src.pipeline import analyze_input

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Legal Document Analyzer | Enterprise Solution",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- PROFESSIONAL STYLES ----------------
st.markdown("""
<style>
/* Clean Professional Background */
.stApp {
    background: #f5f7fa;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Professional Title Styling */
h1 {
    color: #1e3a8a !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem !important;
}

/* Professional Risk Badges */
.risk-high {
    background: #dc2626;
    color: white;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    border: 1px solid #b91c1c;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.risk-medium {
    background: #f59e0b;
    color: white;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    border: 1px solid #d97706;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.risk-low {
    background: #059669;
    color: white;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    border: 1px solid #047857;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Professional Clause Tag */
.clause-tag {
    background: #3b82f6;
    color: white;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    border: 1px solid #2563eb;
    text-transform: uppercase;
    letter-spacing: 0.5px;
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

/* Light pink color for drag and drop text and file limit */
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color: #ffc0cb !important;
}

/* Professional Messages */
.stSuccess, .stInfo {
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

[data-testid="stRadio"] div[role="radiogroup"] label {
    color: #000000 !important;
}

[data-testid="stRadio"] p {
    color: #000000 !important;
}

/* Professional Caption */
.stCaption {
    color: #6b7280 !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
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

/* Professional Spinner */
.stSpinner > div {
    border-color: #3b82f6 transparent transparent transparent !important;
}

/* Professional Multi-select */
[data-baseweb="select"] {
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
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

/* Black text for all input labels */
label {
    color: #000000 !important;
}

/* Black text for radio button labels */
[data-testid="stRadio"] label,
[data-testid="stRadio"] span {
    color: #000000 !important;
}

/* Black text for text input and text area labels */
[data-testid="stTextArea"] label,
[data-testid="stTextInput"] label {
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PROFESSIONAL HEADER ----------------
st.markdown("""
<div style="background: white; padding: 2rem; border-radius: 12px;  rgba(0, 0, 0, 0.1); margin-middle: 2rem; border-middle: 4px solid #1e40af;">
    <h1 style="margin: 0; color: #1f2937;">⚖️Legal Document Analyze</h1>
    <p style="color: #6b7280; margin: 0.5rem 0 0 0; font-size: 1rem;">Enterprise-grade contract analysis powered by advanced AI</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT SECTION
# =====================================================
st.markdown("""
<div class="section-header">
    <h2>Document Input</h2>
    <p style="color: #6b7280; margin: 0.5rem 0 0 0; font-size: 0.95rem;">Upload a PDF or paste your legal document text for analysis</p>
</div>
""", unsafe_allow_html=True)

input_mode = st.radio(
    "Input Method",
    ["Paste Text", "Upload PDF"],
    horizontal=True
)

raw_input = None
input_type = None

if input_mode == "Paste Text":
    raw_input = st.text_area(
        "Document Text",
        height=220,
        placeholder="Paste your legal document text here..."
    )
    input_type = "text"

else:
    raw_input = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Select a PDF file to analyze"
    )
    input_type = "pdf"

st.markdown("<br>", unsafe_allow_html=True)
analyze_clicked = st.button("Analyze Document", use_container_width=True)

# =====================================================
# 🔹 ANALYSIS WITH ENHANCED UI
# =====================================================
if analyze_clicked and raw_input:
    with st.spinner("Analyzing document..."):
        results = analyze_input(raw_input, input_type=input_type)

    st.session_state["results"] = results
    st.success("Analysis completed successfully. View results below.")

# =====================================================
# 🔹 RESULTS SECTION WITH 3D DESIGN
# =====================================================
if "results" in st.session_state:

    df = st.session_state["results"]

    # Normalize column names (SAFE)
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <h2>Analysis Results</h2>
        <p style="color: #6b7280; margin: 0.5rem 0 0 0; font-size: 0.95rem;">Detailed clause-by-clause analysis with risk assessments</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- SUMMARY METRICS ----------
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Clauses", len(df), delta=None, delta_color="normal")
    c2.metric("High Risk", len(df[df["risk_level"] == "High Risk"]), delta=None, delta_color="inverse")
    c3.metric("Low Risk", len(df[df["risk_level"] == "Low Risk"]), delta=None, delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- FILTERS ----------
    with st.expander("Filters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            risk_filter = st.multiselect(
                "Risk Level",
                df["risk_level"].unique().tolist(),
                default=df["risk_level"].unique().tolist()
            )
        
        with col2:
            clause_filter = st.multiselect(
                "Clause Type",
                df["clause_type"].unique().tolist(),
                default=df["clause_type"].unique().tolist()
            )

    filtered_df = df[
        (df["risk_level"].isin(risk_filter)) &
        (df["clause_type"].isin(clause_filter))
    ]

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display filtered count
    if len(filtered_df) < len(df):
        st.info(f"Showing {len(filtered_df)} of {len(df)} clauses")

    # ---------- CLAUSE ANALYSIS ----------
    for i, row in filtered_df.iterrows():

        risk_class = (
            "risk-high" if row["risk_level"] == "High Risk"
            else "risk-medium" if row["risk_level"] == "Medium Risk"
            else "risk-low"
        )

        with st.expander(f"Clause {i + 1}: {row['clause_type']}", expanded=False):
            st.markdown(
                f"""
<div style="margin-bottom: 1.5rem;">
    <span class="clause-tag">{row['clause_type']}</span>
    &nbsp;&nbsp;
    <span class="{risk_class}">{row['risk_level']}</span>
</div>
""",
                unsafe_allow_html=True
            )

            st.markdown("**Original Text**")
            st.markdown(f"""
            <div style="background: #f9fafb; padding: 1.25rem; border-radius: 8px; 
                        border-left: 3px solid #3b82f6; color: #1f2937; line-height: 1.6;">
                {row["clause"]}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Simplified Explanation**")
            st.info(f"{row['simplified_explanation']}")
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: white; 
                border-radius: 8px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); border-top: 2px solid #e5e7eb;">
        <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">
            Powered by Advanced AI & Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)
