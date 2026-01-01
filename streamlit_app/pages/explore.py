import streamlit as st

st.set_page_config(
    page_title="Legal Document Simplifier",
    layout="wide"
)
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

st.title("📄 Legal Document Simplifier")
st.caption("AI-powered contract understanding using ML + Groq LLM")

st.sidebar.success("Select a page")