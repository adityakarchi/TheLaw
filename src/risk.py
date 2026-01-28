"""Risk scoring based on keyword matching."""

RISK_KEYWORDS = {
    "high": [
        "penalty", "terminate immediately",
        "unlimited liability", "sole discretion"
    ],
    "medium": [
        "terminate", "liability",
        "damages", "breach"
    ],
    "low": [
        "notice", "agreement", "confidential"
    ]
}

def calculate_risk_score(text: str) -> int:
    score = 0
    text = text.lower()

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

def risk_level(score: int) -> str:
    if score >= 6:
        return "High Risk"
    elif score >= 3:
        return "Medium Risk"
    return "Low Risk"
