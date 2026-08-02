"""Centralised configuration — single source of truth for all settings.

Loads environment variables from .env (local) or Streamlit Secrets (cloud).
"""

import os
import logging
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


def _get_env(key: str, default: str = "") -> str:
    """Read from os.environ first, then Streamlit secrets (cloud deployment fallback)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
MODELS_DIR      = PROJECT_ROOT / "models"
DATA_DIR        = PROJECT_ROOT / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# ─────────────────────────────────────────────
# LLM (Groq)
# ─────────────────────────────────────────────
GROQ_API_KEY:    str   = _get_env("GROQ_API_KEY")
LLM_MODEL:       str   = _get_env("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE: float = float(_get_env("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS:  int   = int(_get_env("LLM_MAX_TOKENS", "2048"))

# ─────────────────────────────────────────────
# AWS / S3
# ─────────────────────────────────────────────
AWS_ACCESS_KEY_ID:     str = _get_env("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY: str = _get_env("AWS_SECRET_ACCESS_KEY")
AWS_REGION:            str = _get_env("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME:        str = _get_env("S3_BUCKET_NAME")

# ─────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────
EMBEDDING_MODEL:  str = _get_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE: str = _get_env("EMBEDDING_DEVICE", "cpu")

# ─────────────────────────────────────────────
# RAG / Retrieval
# ─────────────────────────────────────────────
CHUNK_SIZE:      int = int(_get_env("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP:   int = int(_get_env("CHUNK_OVERLAP", "200"))
RETRIEVER_TOP_K: int = int(_get_env("RETRIEVER_TOP_K", "4"))

# ─────────────────────────────────────────────
# Legal Detection
# ─────────────────────────────────────────────
MIN_LEGAL_SCORE:            float = float(_get_env("MIN_LEGAL_SCORE", "5.0"))
CONFIDENCE_THRESHOLD_LEGAL: float = float(_get_env("CONFIDENCE_THRESHOLD_LEGAL", "0.50"))


def _validate_api_key() -> str:
    """Validate and return the Groq API key."""
    key = _get_env("GROQ_API_KEY")
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Set it in .env (local) or Streamlit Secrets (cloud).\n"
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


def get_aws_status() -> dict:
    """Return a dict with AWS/S3 configuration status."""
    return {
        "configured": all([
            _get_env("AWS_ACCESS_KEY_ID"),
            _get_env("AWS_SECRET_ACCESS_KEY"),
            _get_env("S3_BUCKET_NAME"),
        ]),
        "bucket": _get_env("S3_BUCKET_NAME"),
        "region": _get_env("AWS_REGION", "ap-south-1"),
    }
