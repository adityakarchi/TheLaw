"""Embedding Pipeline — sentence-transformers via LangChain.

Wraps HuggingFaceEmbeddings for compatibility with FAISS and LangChain
retrieval chains.  Embeddings are cached per-model to avoid reloading.
"""

import logging
from typing import List, Optional
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 64


@lru_cache(maxsize=4)
def _get_embeddings_model(
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
) -> HuggingFaceEmbeddings:
    """Singleton factory — loads model once and caches it."""
    logger.info(f"Loading embedding model: {model_name} on {device}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": DEFAULT_BATCH_SIZE,
        },
    )


class EmbeddingPipeline:
    """Produces dense vector embeddings for text using sentence-transformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
    ):
        self.model_name = model_name
        self.device = device
        self._model: Optional[HuggingFaceEmbeddings] = None

    @property
    def model(self) -> HuggingFaceEmbeddings:
        """Lazy-loaded, cached embedding model."""
        if self._model is None:
            self._model = _get_embeddings_model(self.model_name, self.device)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings → list of float vectors."""
        if not texts:
            return []
        return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string (optimised for retrieval queries)."""
        return self.model.embed_query(query)

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimensionality."""
        sample = self.embed_query("dimension probe")
        return len(sample)

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Return the underlying LangChain-compatible embeddings object.

        Use this when constructing a FAISS vector store directly.
        """
        return self.model
