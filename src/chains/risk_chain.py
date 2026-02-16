"""Risk Analysis Chain — identifies and explains risky clauses.

Scans legal text for problematic provisions such as unlimited liability,
unilateral termination rights, penalty clauses, non-compete overreach, etc.
Returns structured risk output.
"""

import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.utils.config import get_llm

logger = logging.getLogger(__name__)

# ── Prompt Templates ────────────────────────────────────────────────

RISK_SYSTEM_PROMPT = """You are a senior contract risk analyst with expertise in identifying
potentially harmful clauses in legal agreements. You protect your client's interests.

Your analysis must be:
- Thorough: cover ALL types of risk (financial, operational, legal, reputational)
- Specific: cite the exact language that creates the risk
- Actionable: recommend concrete changes or negotiation points
- Prioritized: rank risks by severity (Critical / High / Medium / Low)"""

RISK_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RISK_SYSTEM_PROMPT),
    ("human", """Analyze the following legal text for risky clauses and potential issues.

RELEVANT CONTEXT FROM THE DOCUMENT:
---
{context}
---

LEGAL TEXT TO ANALYZE:
\"\"\"
{text}
\"\"\"

Provide a comprehensive risk analysis with the following structure:

## Risk Summary
(1-2 sentence overview of overall risk level)

## Identified Risks

For EACH risk found:
### [Risk Title]
- **Severity**: Critical / High / Medium / Low
- **Clause**: Quote the problematic language
- **Risk**: Explain why this is dangerous
- **Recommendation**: What to negotiate or change

## Overall Risk Score
Rate the document: Low Risk / Moderate Risk / High Risk / Critical Risk

## Key Recommendations
(Top 3-5 actionable recommendations)

RISK ANALYSIS:"""),
])

RISK_NO_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RISK_SYSTEM_PROMPT),
    ("human", """Analyze the following legal text for risky clauses and potential issues.

LEGAL TEXT TO ANALYZE:
\"\"\"
{text}
\"\"\"

Provide a comprehensive risk analysis with the following structure:

## Risk Summary
(1-2 sentence overview of overall risk level)

## Identified Risks

For EACH risk found:
### [Risk Title]
- **Severity**: Critical / High / Medium / Low
- **Clause**: Quote the problematic language
- **Risk**: Explain why this is dangerous
- **Recommendation**: What to negotiate or change

## Overall Risk Score
Rate the document: Low Risk / Moderate Risk / High Risk / Critical Risk

## Key Recommendations
(Top 3-5 actionable recommendations)

RISK ANALYSIS:"""),
])


# ── Chain Construction ───────────────────────────────────────────────

def get_risk_chain(llm: Optional[ChatGroq] = None):
    """Build a LangChain risk analysis chain."""
    _llm = llm or get_llm()
    return RISK_ANALYSIS_PROMPT | _llm | StrOutputParser()


def get_risk_chain_no_context(llm: Optional[ChatGroq] = None):
    """Risk chain without retrieved context (fallback)."""
    _llm = llm or get_llm()
    return RISK_NO_CONTEXT_PROMPT | _llm | StrOutputParser()


# ── High-Level API ──────────────────────────────────────────────────

def analyze_risks(
    text: str,
    context: str = "",
    llm: Optional[ChatGroq] = None,
) -> str:
    """Analyze a legal document for risky clauses.

    Args:
        text:    The legal text to analyze.
        context: RAG-retrieved context focused on risk-related clauses.
        llm:     Optional pre-configured LLM instance.

    Returns:
        Structured risk analysis as a markdown string.
    """
    if not text or len(text.strip()) < 20:
        return "Text too short for risk analysis."

    try:
        if context.strip():
            chain = get_risk_chain(llm)
            result = chain.invoke({"text": text, "context": context})
        else:
            chain = get_risk_chain_no_context(llm)
            result = chain.invoke({"text": text})

        return result.strip()

    except Exception as e:
        logger.error(f"Risk analysis chain failed: {e}")
        raise
