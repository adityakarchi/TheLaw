"""Pile of Law Loader — loads downloaded JSONL samples for RAG testing.

Reads court opinions, contracts, legal advice, and constitutions from
the Pile of Law dataset. Chunks them and builds a FAISS index for
embedding quality testing and retrieval benchmarking.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

POL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "pile_of_law"

# Source metadata for categorization
SOURCE_META = {
    "courtlistener_opinions": {
        "category": "case_law",
        "label": "Court Opinion",
        "is_legal": True,
    },
    "atticus_contracts": {
        "category": "contract",
        "label": "Legal Contract",
        "is_legal": True,
    },
    "r_legaladvice": {
        "category": "legal_advice",
        "label": "Legal Advice Q&A",
        "is_legal": True,   # mostly legal, some casual
    },
    "constitutions": {
        "category": "constitutional",
        "label": "Constitution",
        "is_legal": True,
    },
}


def load_pol_documents(
    max_per_source: Optional[int] = None,
    chunk: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Load Pile of Law JSONL files as LangChain Documents.

    Args:
        max_per_source: Max docs per JSONL file. None = all.
        chunk: Whether to split into smaller chunks.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of LangChain Documents ready for FAISS indexing.
    """
    if not POL_DIR.exists():
        logger.warning(f"Pile of Law directory not found: {POL_DIR}")
        return []

    all_docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    ) if chunk else None

    for jsonl_file in sorted(POL_DIR.glob("*.jsonl")):
        source_name = jsonl_file.stem
        meta_info = SOURCE_META.get(source_name, {
            "category": "unknown",
            "label": source_name,
            "is_legal": True,
        })

        count = 0
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if max_per_source and count >= max_per_source:
                        break

                    record = json.loads(line.strip())
                    text = record.get("text", "").strip()
                    if not text or len(text) < 50:
                        continue

                    base_metadata = {
                        "source": source_name,
                        "category": meta_info["category"],
                        "label": meta_info["label"],
                        "is_legal": meta_info["is_legal"],
                        "url": record.get("url", ""),
                        "timestamp": record.get("timestamp", ""),
                    }

                    if splitter:
                        # Chunk long documents
                        chunks = splitter.create_documents(
                            [text],
                            metadatas=[base_metadata],
                        )
                        all_docs.extend(chunks)
                    else:
                        doc = Document(
                            page_content=text,
                            metadata=base_metadata,
                        )
                        all_docs.append(doc)

                    count += 1

        except Exception as e:
            logger.error(f"Failed to load {jsonl_file}: {e}")
            continue

        logger.info(f"Loaded {count} docs from {source_name}")

    logger.info(f"Total Pile of Law documents: {len(all_docs)}")
    return all_docs


def load_pol_for_classification() -> List[Dict[str, Any]]:
    """Load samples for legal vs non-legal classification testing.

    Returns list of dicts with 'text', 'is_legal', 'category', 'label'.
    """
    if not POL_DIR.exists():
        return []

    samples = []
    for jsonl_file in sorted(POL_DIR.glob("*.jsonl")):
        source_name = jsonl_file.stem
        meta_info = SOURCE_META.get(source_name, {})

        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line.strip())
                    text = record.get("text", "").strip()
                    if not text or len(text) < 100:
                        continue

                    samples.append({
                        "text": text[:2000],  # Truncate for classification
                        "is_legal": meta_info.get("is_legal", True),
                        "category": meta_info.get("category", "unknown"),
                        "label": meta_info.get("label", source_name),
                        "source": source_name,
                    })
        except Exception as e:
            logger.error(f"Failed to load {jsonl_file}: {e}")

    logger.info(f"Loaded {len(samples)} classification samples")
    return samples


def get_pol_stats() -> Dict[str, Any]:
    """Get statistics about the downloaded Pile of Law data."""
    if not POL_DIR.exists():
        return {"available": False}

    stats = {"available": True, "sources": {}, "total_docs": 0, "total_size_kb": 0}

    for jsonl_file in sorted(POL_DIR.glob("*.jsonl")):
        source_name = jsonl_file.stem
        size_kb = jsonl_file.stat().st_size / 1024
        doc_count = sum(1 for _ in open(jsonl_file, encoding="utf-8"))

        stats["sources"][source_name] = {
            "docs": doc_count,
            "size_kb": round(size_kb, 1),
            "label": SOURCE_META.get(source_name, {}).get("label", source_name),
        }
        stats["total_docs"] += doc_count
        stats["total_size_kb"] += size_kb

    stats["total_size_kb"] = round(stats["total_size_kb"], 1)
    return stats
