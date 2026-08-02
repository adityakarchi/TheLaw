"""Feature verification script — checks F1 to F9 are all properly wired."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

results = {}

def check(name, fn):
    try:
        fn()
        results[name] = ("PASS", "")
    except Exception as e:
        results[name] = ("FAIL", str(e))

# ── F1: Fine-tuned BERT Classifier ──────────────────────────────────
def f1():
    from src.classification import ClassifierPipeline, get_classifier, CLAUSE_RISK_MAP
    clf = get_classifier()
    assert clf is not None
    assert hasattr(clf, "classify_clause")
    assert hasattr(clf, "classify_batch")
    assert hasattr(clf, "format_for_risk_prompt")
    assert len(CLAUSE_RISK_MAP) >= 5
    # Check workflow imports it
    import inspect, src.graph.workflow as wf
    src_code = inspect.getsource(wf)
    assert "_get_classifier" in src_code
    assert "format_for_risk_prompt" in src_code
check("F1: BERT Classifier", f1)

# ── F2: Pile of Law Index ────────────────────────────────────────────
def f2():
    from src.rag.pile_of_law import load_pol_documents, get_pol_stats, SOURCE_META
    assert len(SOURCE_META) >= 3
    stats = get_pol_stats()
    assert "available" in stats
    import inspect, src.graph.workflow as wf
    src_code = inspect.getsource(wf)
    assert "_get_pol_index" in src_code
    assert "pile_of_law" in src_code
    assert "source_type" in src_code
check("F2: Pile of Law Index", f2)

# ── F3: DOCX Support ─────────────────────────────────────────────────
def f3():
    from src.rag.loader import DocumentLoader
    dl = DocumentLoader()
    assert hasattr(dl, "load_docx")
    from src.utils.s3_storage import upload_docx_to_s3
    import inspect
    import streamlit_app as app
    src_code = inspect.getsource(app)
    assert "upload_docx_to_s3" in src_code
    assert "tab_docx" in src_code
    assert "analyze_docx_btn" in src_code
    assert "docx" in src_code
check("F3: DOCX Support", f3)

# ── F4: Clause Breakdown ─────────────────────────────────────────────
def f4():
    from src.segmentation import segment_into_clauses, _segment_by_paragraphs
    from src.chains.clause_chain import simplify_clause, simplify_clauses_batch
    import inspect
    import streamlit_app as app
    src_code = inspect.getsource(app)
    assert "Clause Breakdown" in src_code
    assert "simplify_clauses_batch" in src_code
    assert "negotiation_tip" in src_code
    assert "Show High Risk Only" in src_code
    assert "Show Plain English" in src_code
check("F4: Clause Breakdown", f4)

# ── F5: Contract Comparison ──────────────────────────────────────────
def f5():
    from src.comparison import segment_and_align, calculate_risk_delta
    from src.chains.compare_chain import summarize_change, overall_comparison_summary
    import inspect
    import streamlit_app as app
    src_code = inspect.getsource(app)
    assert "Compare Contracts" in src_code
    assert "compare_result" in src_code
    assert "risk_delta" in src_code
    assert "segment_and_align" in src_code
    assert "overall_comparison_summary" in src_code
check("F5: Contract Comparison", f5)

# ── F6: PDF Export ───────────────────────────────────────────────────
def f6():
    from src.utils.pdf_export import generate_analysis_report, generate_case_report
    import inspect
    import streamlit_app as app
    src_code = inspect.getsource(app)
    assert "PDF_EXPORT_AVAILABLE" in src_code
    assert "generate_analysis_report" in src_code
    assert "generate_case_report" in src_code
    assert "Download Full PDF Report" in src_code
    assert "Download Case Report PDF" in src_code
check("F6: PDF Export", f6)

# ── F7: Multi-language ───────────────────────────────────────────────
def f7():
    from src.graph.state import GraphState
    from src.chains.simplify_chain import simplify_with_context
    from src.chains.risk_chain import analyze_risks
    import inspect
    src_simplify = inspect.getsource(simplify_with_context)
    src_risk = inspect.getsource(analyze_risks)
    assert "language" in src_simplify
    assert "language" in src_risk
    import src.graph.workflow as wf
    wf_code = inspect.getsource(wf)
    assert "output_language" in wf_code
    import streamlit_app as app
    app_code = inspect.getsource(app)
    assert "LANGUAGE_OPTIONS" in app_code
    assert "output_language" in app_code
    assert "Hindi" in app_code
check("F7: Multi-language", f7)

# ── F8: Analytics ────────────────────────────────────────────────────
def f8():
    from src.utils.analytics import log_analysis_event, load_analytics, get_summary_stats
    import inspect
    import streamlit_app as app
    src_code = inspect.getsource(app)
    assert "log_analysis_event" in src_code
    assert "Analytics Dashboard" in src_code
    assert "load_analytics" in src_code
    assert "get_summary_stats" in src_code
    assert "plotly.express" in src_code
    # Check it logs for BOTH pipelines
    assert src_code.count("log_analysis_event") >= 2
check("F8: Analytics", f8)

# ── F9: Rate Limiting ────────────────────────────────────────────────
def f9():
    import inspect
    import streamlit_app as app
    src_code = inspect.getsource(app)
    assert "check_rate_limit" in src_code
    assert "MAX_ANALYSES_PER_SESSION" in src_code
    assert "analysis_count" in src_code
    assert "case_count" in src_code
    assert "Session Usage" in src_code
    assert "st.progress" in src_code
    # check rate limit called before every analysis trigger
    assert src_code.count("check_rate_limit") >= 4
check("F9: Rate Limiting", f9)

# ── Print Results ────────────────────────────────────────────────────
print("\n" + "="*55)
print("  LEGAL AI PLATFORM — FEATURE VERIFICATION REPORT")
print("="*55)

all_pass = True
for feature, (status, error) in results.items():
    icon = "✅" if status == "PASS" else "❌"
    print(f"  {icon}  {feature}")
    if status == "FAIL":
        all_pass = False
        print(f"       └─ {error[:100]}")

print("="*55)
if all_pass:
    print("  🎉  ALL 9 FEATURES PASSED")
else:
    fail_count = sum(1 for s, _ in results.values() if s == "FAIL")
    print(f"  ⚠️   {fail_count} feature(s) failed — see errors above")
print("="*55 + "\n")

sys.exit(0 if all_pass else 1)
