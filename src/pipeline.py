import re
import pickle
import pandas as pd
import os
from src.preprocessing import read_pdf

from .simplification import simplify_clause
from .classification import classify_clause

# Get the base directory (legal/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models once
with open(os.path.join(BASE_DIR, "models/clause_classifier/tfidf.pkl"), "rb") as f:
    tfidf = pickle.load(f)

with open(os.path.join(BASE_DIR, "models/clause_classifier/model.pkl"), "rb") as f:
    classifier = pickle.load(f)

RISK_KEYWORDS = {
    "high": ["penalty", "terminate immediately", "unlimited liability", "sole discretion"],
    "medium": ["terminate", "liability", "damages", "breach"],
    "low": ["notice", "agreement", "confidential"]
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def segment_clauses(text):
    clauses = re.split(r"\.\s+|;\s+|\n+|\:\s+", text)
    return [c.strip() for c in clauses if len(c.strip()) > 20]

def calculate_risk_score(text):
    score = 0
    for w in RISK_KEYWORDS["high"]:
        if w in text:
            score += 3
    for w in RISK_KEYWORDS["medium"]:
        if w in text:
            score += 2
    for w in RISK_KEYWORDS["low"]:
        if w in text:
            score += 1
    return score

def risk_level(score):
    if score >= 6:
        return "High Risk"
    elif score >= 3:
        return "Medium Risk"
    return "Low Risk"

def analyze_document(text: str, max_clauses: int = 8):
    cleaned_text = clean_text(text)
    clauses = segment_clauses(cleaned_text)

    results = []

    for clause in clauses[:max_clauses]:
        clause_type = classify_clause(clause)

        score = calculate_risk_score(clause)
        level = risk_level(score)

        simplified = simplify_clause(clause)

        results.append({
            "clause": clause,
            "clause_type": clause_type,
            "risk_score": score,
            "risk_level": level,
            "simplified_explanation": simplified
        })

    # Ensure we always return a DataFrame with the correct columns
    if not results:
        return pd.DataFrame(columns=["clause", "clause_type", "risk_score", "risk_level", "simplified_explanation"])
    
    return pd.DataFrame(results)



def analyze_input(input_data, input_type="text"):
    """
    input_type: 'text' or 'pdf'
    """
    if input_type == "pdf":
        raw_text = read_pdf(input_data)
    else:
        raw_text = input_data

    return analyze_document(raw_text)
