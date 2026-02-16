"""Law Index Builder — creates and manages the FAISS index for Indian laws.

Loads IPC, CrPC, Evidence Act sections from the corpus, embeds them,
and stores in a persistent FAISS index. The index is built once and
loaded from disk on subsequent runs for fast startup.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.rag.embedder import EmbeddingPipeline
from src.rag.law_corpus import load_all_documents, load_all_laws, LawSection

logger = logging.getLogger(__name__)

# Paths

LAW_INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "law_index"


class LawIndexBuilder:
    """Builds, persists, and searches the FAISS index of Indian law sections."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_dir: Optional[str] = None,
    ):
        self.embedder = EmbeddingPipeline(model_name=embedding_model)
        self.index_dir = Path(index_dir) if index_dir else LAW_INDEX_DIR
        self._store: Optional[FAISS] = None
        self._laws: Optional[List[LawSection]] = None

    @property
    def is_ready(self) -> bool:
        return self._store is not None

    def build_index(self, force_rebuild: bool = False) -> "LawIndexBuilder":
        """Build or load the FAISS index for all law sections.

        If a saved index exists on disk and force_rebuild is False,
        loads from disk. Otherwise, builds from the JSON corpus.
        """
        index_file = self.index_dir / "index.faiss"

        if index_file.exists() and not force_rebuild:
            try:
                return self.load_index()
            except Exception as e:
                logger.warning(f"Failed to load saved index, rebuilding: {e}")

        logger.info("Building law FAISS index from corpus...")
        documents = load_all_documents()
        self._laws = load_all_laws()

        if not documents:
            raise RuntimeError("No law documents loaded. Check corpus files in data/legal_corpus/.")

        embeddings = self.embedder.get_langchain_embeddings()
        self._store = FAISS.from_documents(documents, embeddings)

        # Persist to disk
        self.save_index()
        logger.info(f"Law index built with {len(documents)} sections")
        return self

    def save_index(self) -> None:
        """Persist index to disk."""
        if self._store is None:
            raise RuntimeError("No index to save. Build first.")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.index_dir))
        logger.info(f"Law index saved to {self.index_dir}")

    def load_index(self) -> "LawIndexBuilder":
        """Load index from disk."""
        load_path = self.index_dir
        if not (load_path / "index.faiss").exists():
            raise FileNotFoundError(f"No law index at {load_path}. Run build_index() first.")

        embeddings = self.embedder.get_langchain_embeddings()
        self._store = FAISS.load_local(
            str(load_path), embeddings, allow_dangerous_deserialization=True
        )
        self._laws = load_all_laws()
        logger.info(f"Law index loaded from {load_path}")
        return self

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """Search the law index for relevant sections.

        Returns list of (Document, score) tuples sorted by relevance.
        Lower score = more relevant for FAISS L2 distance.
        """
        if not self.is_ready:
            self.build_index()

        results = self._store.similarity_search_with_score(query, k=k)
        return results

    def search_laws(
        self,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search and return structured law results with confidence scores.

        Returns a list of dicts with all law metadata plus a confidence score.
        """
        results = self.search(query, k=k)

        if not results:
            return []

        # Normalize scores to 0-1 confidence (FAISS L2: lower = better)
        max_score = max(score for _, score in results) if results else 1.0
        min_score = min(score for _, score in results) if results else 0.0
        score_range = max_score - min_score if max_score != min_score else 1.0

        structured = []
        for doc, score in results:
            # Convert L2 distance to confidence (invert and normalize)
            confidence = max(0.0, 1.0 - (score / (max_score + 0.1)))
            confidence = round(confidence, 3)

            meta = doc.metadata
            structured.append({
                "act_name": meta.get("act_name", ""),
                "abbreviation": meta.get("abbreviation", ""),
                "section": meta.get("section", ""),
                "title": meta.get("title", ""),
                "crime": meta.get("crime", ""),
                "punishment": meta.get("punishment", ""),
                "jail_term": meta.get("jail_term", ""),
                "fine": meta.get("fine", ""),
                "bailable": meta.get("bailable", False),
                "cognizable": meta.get("cognizable", True),
                "category": meta.get("category", ""),
                "confidence": confidence,
                "relevance_score": round(score, 4),
                "full_text": doc.page_content,
            })

        return structured

    def get_context_for_llm(self, query: str, k: int = 5) -> str:
        """Get a formatted context string for the LLM explanation chain."""
        results = self.search(query, k=k)
        if not results:
            return "No relevant law sections found."

        parts = []
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            parts.append(
                f"--- Law {i} ---\n"
                f"Act: {meta.get('act_name', '')} ({meta.get('abbreviation', '')})\n"
                f"Section: {meta.get('section', '')}\n"
                f"Crime: {meta.get('crime', '')}\n"
                f"Punishment: {meta.get('punishment', '')}\n"
                f"Jail Term: {meta.get('jail_term', '')}\n"
                f"Fine: {meta.get('fine', '')}\n"
                f"Bailable: {'Yes' if meta.get('bailable') else 'No'}\n"
                f"Cognizable: {'Yes' if meta.get('cognizable') else 'No'}\n"
                f"Description: {doc.page_content[:500]}\n"
            )
        return "\n".join(parts)

    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        return {
            "index_ready": self.is_ready,
            "index_path": str(self.index_dir),
            "law_count": len(self._laws) if self._laws else 0,
            "index_exists_on_disk": (self.index_dir / "index.faiss").exists(),
        }


# Singleton

_law_index: Optional[LawIndexBuilder] = None


def get_law_index() -> LawIndexBuilder:
    """Get or create the singleton law index."""
    global _law_index
    if _law_index is None:
        _law_index = LawIndexBuilder()
        _law_index.build_index()
    return _law_index


def reset_law_index() -> None:
    """Reset the singleton (forces rebuild on next access)."""
    global _law_index
    _law_index = None
