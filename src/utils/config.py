"""Centralised configuration — single source of truth for all settings.

Loads environment variables from .env, defines model parameters, paths,
and provides factory functions for shared resources (LLM, embeddings).
"""

import os
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ── Load .env ────────────────────────────────────────────────────────

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# ── LLM Settings ────────────────────────────────────────────────────

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ── Embedding Settings ──────────────────────────────────────────────

EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

# ── RAG Settings ─────────────────────────────────────────────────────

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "4"))

# ── Detection Settings ──────────────────────────────────────────────

MIN_LEGAL_SCORE: float = float(os.getenv("MIN_LEGAL_SCORE", "5.0"))
CONFIDENCE_THRESHOLD_LEGAL: float = float(os.getenv("CONFIDENCE_THRESHOLD_LEGAL", "0.50"))

# ── Factories ────────────────────────────────────────────────────────


def _validate_api_key() -> str:
    """Validate and return the Groq API key."""
    key = GROQ_API_KEY
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Set it in .env or as an environment variable.\n"
            "Get your free key at: https://console.groq.com/"
        )
    return key


@lru_cache(maxsize=1)
def get_llm():
    """Return a cached ChatGroq instance configured from environment."""
    from langchain_groq import ChatGroq

    api_key = _validate_api_key()
    logger.info(f"Initializing ChatGroq: model={LLM_MODEL}, temp={LLM_TEMPERATURE}")
    return ChatGroq(
        api_key=api_key,
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        max_retries=3,
    )


def check_api_connection() -> tuple[bool, str]:
    """Quick health check for the Groq API."""
    try:
        llm = get_llm()
        resp = llm.invoke("Say OK")
        if resp.content:
            return True, "API connection successful"
        return False, "API returned empty response"
    except Exception as e:
        return False, f"Connection failed: {e}"
