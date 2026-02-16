"""Case Research Chain — LLM explanation of retrieved law sections.

This chain takes the FAISS-retrieved law sections and uses Groq LLM
ONLY to explain them in context of the user's case description.
The LLM does NOT do retrieval or classification — only explanation.
"""

import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.utils.config import get_llm

logger = logging.getLogger(__name__)

# Prompt templates

CASE_SYSTEM_PROMPT = """You are a senior Indian criminal lawyer with 25 years of experience.
You analyze legal situations and explain applicable laws clearly.

CRITICAL RULES:
1. ONLY use the law sections provided in the context — do NOT invent or hallucinate laws.
2. Explain each applicable section in simple language anyone can understand.
3. Provide practical legal advice based on the retrieved sections.
4. Be honest about uncertainties — if a section might not apply perfectly, say so.
5. Always recommend consulting a qualified lawyer for actual legal proceedings.
6. Focus on the specific case scenario described by the user.
7. Consider both the victim's and accused's perspectives where relevant."""

CASE_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CASE_SYSTEM_PROMPT),
    ("human", """A person has described the following legal situation:

CASE DESCRIPTION:
\"\"\"{case_description}\"\"\"

Based on my legal database search, the following Indian law sections are relevant:

RETRIEVED LAW SECTIONS:
---
{law_context}
---

Provide a comprehensive legal analysis with the following structure:

## Case Analysis
(Brief analysis of the legal situation described)

## Applicable Laws

For EACH relevant law section retrieved:
### [Act Abbreviation] Section [Number] - [Crime Name]
- **Act**: Full act name
- **Section**: Section number
- **Crime**: What offence this covers
- **Punishment**: What penalty applies
- **Jail Term**: Duration of imprisonment
- **Fine**: Fine amount if applicable
- **Bailable/Non-bailable**: Whether bail is available
- **Relevance to Case**: Why this section applies to this specific situation

## Confidence Assessment
Rate how well the retrieved laws match the described situation (High/Medium/Low) and explain.

## Recommended Actions
1. Immediate steps the person should take
2. Legal process overview (FIR, investigation, trial)
3. Important precautions

## Winning Probability Estimate
Based on the described situation and applicable laws, provide a rough estimate of success probability with explanation.

## Important Disclaimer
(Always include a disclaimer about consulting a qualified lawyer)

LEGAL ANALYSIS:"""),
])

CASE_BRIEF_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CASE_SYSTEM_PROMPT),
    ("human", """Case: \"{case_description}\"

Relevant law sections found:
{law_context}

Provide a concise analysis: applicable sections, punishments, recommended actions, and winning probability.

ANALYSIS:"""),
])


# ── Chain Construction ───────────────────────────────────────────────

def get_case_analysis_chain(llm: Optional[ChatGroq] = None):
    """Build a LangChain chain for case analysis with full explanation."""
    _llm = llm or get_llm()
    return CASE_ANALYSIS_PROMPT | _llm | StrOutputParser()


def get_case_brief_chain(llm: Optional[ChatGroq] = None):
    """Build a shorter analysis chain for quick responses."""
    _llm = llm or get_llm()
    return CASE_BRIEF_PROMPT | _llm | StrOutputParser()


# High-level API

def analyze_case(
    case_description: str,
    law_context: str,
    llm: Optional[ChatGroq] = None,
    brief: bool = False,
) -> str:
    """Analyze a case description using retrieved law context.

    Args:
        case_description: User's description of the legal situation.
        law_context: Formatted string of retrieved law sections from FAISS.
        llm: Optional ChatGroq instance.
        brief: If True, use the shorter prompt for faster response.

    Returns:
        Structured legal analysis string.
    """
    if not case_description or not case_description.strip():
        return "Please describe your legal situation."

    if not law_context or not law_context.strip():
        return ("No relevant law sections were found for your case description. "
                "Please try describing the situation in more detail.")

    try:
        if brief:
            chain = get_case_brief_chain(llm)
        else:
            chain = get_case_analysis_chain(llm)

        result = chain.invoke({
            "case_description": case_description,
            "law_context": law_context,
        })
        return result.strip()

    except Exception as e:
        logger.error(f"Case analysis chain failed: {e}")
        raise


def explain_single_section(
    section_data: dict,
    case_description: str,
    llm: Optional[ChatGroq] = None,
) -> str:
    """Explain a single law section in context of a case."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior Indian criminal lawyer. Explain the following law section in simple terms, relating it to the described case."),
        ("human", """Case: \"{case_description}\"

Law Section:
- Act: {act_name}
- Section: {section}
- Crime: {crime}
- Punishment: {punishment}
- Jail Term: {jail_term}

Explain how this section applies to the case in 3-4 sentences:"""),
    ])

    _llm = llm or get_llm()
    chain = prompt | _llm | StrOutputParser()

    try:
        return chain.invoke({
            "case_description": case_description,
            "act_name": section_data.get("act_name", ""),
            "section": section_data.get("section", ""),
            "crime": section_data.get("crime", ""),
            "punishment": section_data.get("punishment", ""),
            "jail_term": section_data.get("jail_term", ""),
        }).strip()
    except Exception as e:
        logger.error(f"Section explanation failed: {e}")
        return f"[Explanation unavailable: {e}]"
