import streamlit as st
import sys

sys.path.append("..")

from src.pipeline import analyze_input

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Legal Document Simplifier",
    layout="wide"
)

# ---------------- STYLES ----------------
st.markdown("""
<style>
.risk-high {
    background-color: #ff4d4f;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    display: inline-block;
}
.risk-medium {
    background-color: #faad14;
    color: black;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    display: inline-block;
}
.risk-low {
    background-color: #52c41a;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    display: inline-block;
}
.clause-tag {
    background-color: #e6f4ff;
    color: #0958d9;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("📄 Legal Document Simplifier")
st.caption("AI-powered contract understanding using ML + Groq LLM")

st.divider()

# =====================================================
# 🔹 INPUT SECTION
# =====================================================
st.subheader("📤 Upload or Paste Legal Document")

input_mode = st.radio(
    "Choose input type:",
    ["Paste Text", "Upload PDF"],
    horizontal=True
)

raw_input = None
input_type = None

if input_mode == "Paste Text":
    raw_input = st.text_area(
        "Paste legal text here:",
        height=220
    )
    input_type = "text"

else:
    raw_input = st.file_uploader(
        "Upload a legal PDF",
        type=["pdf"]
    )
    input_type = "pdf"

analyze_clicked = st.button("🚀 Analyze Document", use_container_width=True)

# =====================================================
# 🔹 ANALYSIS
# =====================================================
if analyze_clicked and raw_input:
    with st.spinner("Analyzing document..."):
        results = analyze_input(raw_input, input_type=input_type)

    st.session_state["results"] = results
    st.success("Analysis complete! Scroll down to view results.")

# =====================================================
# 🔹 RESULTS SECTION
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

    st.divider()
    st.subheader("🔍 Clause Analysis Results")

    # ---------- SUMMARY ----------
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Clauses", len(df))
    c2.metric("High Risk", len(df[df["risk_level"] == "High Risk"]))
    c3.metric("Low Risk", len(df[df["risk_level"] == "Low Risk"]))

    st.divider()

    # ---------- FILTERS ----------
    with st.expander("🎛️ Filters"):
        risk_filter = st.multiselect(
            "Risk Level",
            df["risk_level"].unique().tolist(),
            default=df["risk_level"].unique().tolist()
        )

        clause_filter = st.multiselect(
            "Clause Type",
            df["clause_type"].unique().tolist(),
            default=df["clause_type"].unique().tolist()
        )

    filtered_df = df[
        (df["risk_level"].isin(risk_filter)) &
        (df["clause_type"].isin(clause_filter))
    ]

    # ---------- CLAUSE CARDS ----------
    for i, row in filtered_df.iterrows():

        risk_class = (
            "risk-high" if row["risk_level"] == "High Risk"
            else "risk-medium" if row["risk_level"] == "Medium Risk"
            else "risk-low"
        )

        with st.expander(f"Clause {i + 1}"):
            st.markdown(
                f"""
<span class="clause-tag">{row['clause_type']}</span>
&nbsp;
<span class="{risk_class}">{row['risk_level']}</span>
""",
                unsafe_allow_html=True
            )

            st.markdown("### 📜 Original Clause")
            st.write(row["clause"])

            st.markdown("### ✨ Simplified Explanation")
            st.success(row["simplified_explanation"])
