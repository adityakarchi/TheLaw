"""FAISS Vector Store — create, persist, and search.

Thin wrapper around langchain_community FAISS that adds caching,
persistence helpers, and batch-insert utilities.
"""

import logging
import os
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.rag.embedder import EmbeddingPipeline

logger = logging.getLogger(__name__)

# Default persistence directory
_DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "faiss_index"


class VectorStore:
    """FAISS-backed vector store for legal document chunks."""

    def __init__(
        self,
        embedding_pipeline: Optional[EmbeddingPipeline] = None,
        index_dir: Optional[str] = None,
    ):
        self.embedder = embedding_pipeline or EmbeddingPipeline()
        self.index_dir = Path(index_dir) if index_dir else _DEFAULT_INDEX_DIR
        self._store: Optional[FAISS] = None

    @property
    def store(self) -> Optional[FAISS]:
        return self._store

    @property
    def is_ready(self) -> bool:
        return self._store is not None

    # Build / Load

    def build_from_documents(self, documents: List[Document]) -> "VectorStore":
        """Create a FAISS index from a list of LangChain Documents."""
        if not documents:
            raise ValueError("Cannot build vector store from empty document list")

        logger.info(f"Building FAISS index from {len(documents)} chunks …")
        embeddings = self.embedder.get_langchain_embeddings()
        self._store = FAISS.from_documents(documents, embeddings)
        logger.info("FAISS index built successfully")
        return self

    def add_documents(self, documents: List[Document]) -> None:
        """Insert additional documents into the existing store."""
        if self._store is None:
            self.build_from_documents(documents)
            return
        self._store.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to existing index")

    # Persistence

    def save(self, directory: Optional[str] = None) -> None:
        """Persist FAISS index to disk."""
        if self._store is None:
            raise RuntimeError("No vector store to save. Build or load first.")

        save_dir = Path(directory) if directory else self.index_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(save_dir))
        logger.info(f"FAISS index saved to {save_dir}")

    def load(self, directory: Optional[str] = None) -> "VectorStore":
        """Load FAISS index from disk."""
        load_dir = Path(directory) if directory else self.index_dir

        if not (load_dir / "index.faiss").exists():
            raise FileNotFoundError(f"No FAISS index at {load_dir}")

        embeddings = self.embedder.get_langchain_embeddings()
        self._store = FAISS.load_local(
            str(load_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Loaded FAISS index from {load_dir}")
        return self

    # Search

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """Retrieve the top-k most similar documents.

        Optionally filter by minimum similarity score.
        """
        if self._store is None:
            raise RuntimeError("Vector store not initialized. Build or load first.")

        if score_threshold is not None:
            results = self._store.similarity_search_with_score(query, k=k)
            # FAISS returns L2 distance — lower is better
            return [doc for doc, score in results if score <= score_threshold]

        return self._store.similarity_search(query, k=k)

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 4,
    ) -> List[tuple]:
        """Return (Document, score) tuples."""
        if self._store is None:
            raise RuntimeError("Vector store not initialized. Build or load first.")
        return self._store.similarity_search_with_score(query, k=k)

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """Return a LangChain-compatible retriever interface."""
        if self._store is None:
            raise RuntimeError("Vector store not initialized. Build or load first.")
        kwargs = search_kwargs or {"k": 4}
        return self._store.as_retriever(search_kwargs=kwargs)
