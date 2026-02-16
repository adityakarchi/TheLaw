"""Legal Retriever — end-to-end retrieval interface.

Orchestrates loader → embedder → vector store → search in one object.
Designed to be the single entry-point consumed by LangGraph nodes and
LangChain chains.
"""

import logging
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from src.rag.loader import DocumentLoader, LoaderConfig
from src.rag.embedder import EmbeddingPipeline
from src.rag.vectordb import VectorStore

logger = logging.getLogger(__name__)


class LegalRetriever:
    """High-level retrieval facade: ingest a document, then answer queries."""

    def __init__(
        self,
        loader_config: Optional[LoaderConfig] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 4,
    ):
        self.loader = DocumentLoader(loader_config)
        self.embedder = EmbeddingPipeline(model_name=embedding_model)
        self.vector_store = VectorStore(embedding_pipeline=self.embedder)
        self.top_k = top_k

        # Cache of last-ingested document text
        self._raw_text: Optional[str] = None
        self._chunks: Optional[List[Document]] = None

    @property
    def is_ready(self) -> bool:
        return self.vector_store.is_ready

    @property
    def raw_text(self) -> Optional[str]:
        return self._raw_text

    @property
    def chunks(self) -> Optional[List[Document]]:
        return self._chunks

    # Ingest

    def ingest(self, input_data, input_type: str = "text") -> str:
        """Load, chunk, embed, and index a document. Returns cleaned text."""
        logger.info(f"Ingesting document (type={input_type}) …")

        # Load & clean
        self._raw_text = self.loader.load(input_data, input_type)

        # Split
        self._chunks = self.loader.split_into_chunks(
            self._raw_text,
            metadata={"source_type": input_type},
        )

        # Build FAISS index
        self.vector_store.build_from_documents(self._chunks)
        logger.info(f"Ingestion complete: {len(self._chunks)} chunks indexed")
        return self._raw_text

    # Retrieve

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve top-k relevant chunks for a query."""
        if not self.vector_store.is_ready:
            raise RuntimeError("No document ingested yet. Call ingest() first.")
        return self.vector_store.similarity_search(query, k=k or self.top_k)

    def retrieve_with_scores(self, query: str, k: Optional[int] = None) -> List[tuple]:
        """Retrieve with similarity scores."""
        if not self.vector_store.is_ready:
            raise RuntimeError("No document ingested yet. Call ingest() first.")
        return self.vector_store.similarity_search_with_scores(query, k=k or self.top_k)

    def get_context_string(self, query: str, k: Optional[int] = None) -> str:
        """Retrieve and concatenate chunks into a single context string."""
        docs = self.retrieve(query, k)
        if not docs:
            return ""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """Return LangChain Retriever interface for chain integration."""
        kwargs = search_kwargs or {"k": self.top_k}
        return self.vector_store.as_retriever(search_kwargs=kwargs)

    # Utilities

    def get_full_text(self) -> str:
        """Return the full raw text of the ingested document."""
        return self._raw_text or ""

    def get_summary_context(self, max_chunks: int = 6) -> str:
        """Return a broad context sample for summarisation tasks."""
        if not self._chunks:
            return self._raw_text or ""

        # Sample evenly across the document
        step = max(1, len(self._chunks) // max_chunks)
        sampled = self._chunks[::step][:max_chunks]
        return "\n\n---\n\n".join(c.page_content for c in sampled)

    def stats(self) -> Dict[str, Any]:
        """Return ingestion stats."""
        return {
            "text_length": len(self._raw_text) if self._raw_text else 0,
            "chunk_count": len(self._chunks) if self._chunks else 0,
            "index_ready": self.vector_store.is_ready,
        }
