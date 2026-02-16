"""RAG components: document loading, embedding, vector storage, retrieval."""

from src.rag.loader import DocumentLoader
from src.rag.embedder import EmbeddingPipeline
from src.rag.vectordb import VectorStore
from src.rag.retriever import LegalRetriever
from src.rag.law_corpus import LawSection, load_all_laws, load_all_documents
from src.rag.law_index import LawIndexBuilder, get_law_index
from src.rag.pile_of_law import load_pol_documents, load_pol_for_classification, get_pol_stats

__all__ = [
    "DocumentLoader",
    "EmbeddingPipeline",
    "VectorStore",
    "LegalRetriever",
    "LawSection",
    "load_all_laws",
    "load_all_documents",
    "LawIndexBuilder",
    "get_law_index",
    "load_pol_documents",
    "load_pol_for_classification",
    "get_pol_stats",
]
