"""Document Loader — supports PDF and plain text with semantic chunking.

Uses LangChain text splitters for intelligent chunking that preserves
legal clause boundaries where possible.
"""

import logging
import tempfile
import os
from typing import List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig:
    """Configuration for the document loader."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_pdf_pages: int = 100
    min_chunk_length: int = 50
    # Legal-specific separators for better clause boundary detection
    separators: List[str] = field(default_factory=lambda: [
        "\n\n\n",       # Major section breaks
        "\n\n",          # Paragraph breaks
        "\nARTICLE ",    # Article headers
        "\nSECTION ",   # Section headers
        "\nCLAUSE ",    # Clause headers
        "\n",            # Line breaks
        ". ",            # Sentence boundaries
        " ",             # Word boundaries
    ])


class DocumentLoader:
    """Production document loader with PDF/text support and LangChain chunking."""

    def __init__(self, config: Optional[LoaderConfig] = None):
        self.config = config or LoaderConfig()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def load_pdf(self, file_path_or_buffer) -> str:
        """Extract raw text from a PDF file or file-like buffer.

        Tries PyMuPDF (fitz) first for quality, falls back to pypdf.
        """
        try:
            return self._load_with_pymupdf(file_path_or_buffer)
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed ({e}), falling back to pypdf")
            return self._load_with_pypdf(file_path_or_buffer)

    def load_text(self, text: str) -> str:
        """Clean and validate raw text input."""
        if not text or not text.strip():
            raise ValueError("Input text is empty")
        return self._clean_text(text)

    def load(self, input_data, input_type: str = "text") -> str:
        """Unified loader: accepts PDF path/buffer or raw text string."""
        if input_type == "pdf":
            raw = self.load_pdf(input_data)
        else:
            raw = self.load_text(str(input_data))
        return self._clean_text(raw)

    def split_into_chunks(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """Split text into LangChain Documents with metadata."""
        base_meta = metadata or {}
        docs = self._splitter.create_documents(
            texts=[text],
            metadatas=[base_meta],
        )
        # Filter out tiny chunks
        docs = [d for d in docs if len(d.page_content.strip()) >= self.config.min_chunk_length]

        # Enrich each chunk with positional metadata
        for idx, doc in enumerate(docs):
            doc.metadata["chunk_index"] = idx
            doc.metadata["chunk_count"] = len(docs)
            doc.metadata["char_start"] = text.find(doc.page_content[:80])

        logger.info(f"Split document into {len(docs)} chunks (avg {sum(len(d.page_content) for d in docs) // max(len(docs), 1)} chars)")
        return docs

    def load_and_split(self, input_data, input_type: str = "text") -> List[Document]:
        """End-to-end: load → clean → split into chunks."""
        text = self.load(input_data, input_type)
        return self.split_into_chunks(text, metadata={"source_type": input_type})

    # Private helpers

    def _load_with_pymupdf(self, file_path_or_buffer) -> str:
        """Extract text using PyMuPDF (fitz)."""
        import fitz  # PyMuPDF

        if hasattr(file_path_or_buffer, "read"):
            data = file_path_or_buffer.read()
            if hasattr(file_path_or_buffer, "seek"):
                file_path_or_buffer.seek(0)
            doc = fitz.open(stream=data, filetype="pdf")
        else:
            doc = fitz.open(str(file_path_or_buffer))

        if doc.page_count == 0:
            raise ValueError("PDF has no pages")
        if doc.page_count > self.config.max_pdf_pages:
            logger.warning(f"PDF has {doc.page_count} pages, truncating to {self.config.max_pdf_pages}")

        pages = []
        for i in range(min(doc.page_count, self.config.max_pdf_pages)):
            page_text = doc[i].get_text()
            if page_text.strip():
                pages.append(page_text)
        doc.close()

        if not pages:
            raise ValueError("No text could be extracted from PDF")
        return "\n\n".join(pages)

    def _load_with_pypdf(self, file_path_or_buffer) -> str:
        """Fallback extraction using pypdf."""
        from pypdf import PdfReader

        reader = PdfReader(file_path_or_buffer)
        if len(reader.pages) == 0:
            raise ValueError("PDF has no pages")

        pages = []
        for i, page in enumerate(reader.pages[:self.config.max_pdf_pages]):
            try:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(text)
            except Exception as e:
                logger.warning(f"Failed to extract page {i + 1}: {e}")

        if not pages:
            raise ValueError("No text could be extracted from PDF")
        return "\n\n".join(pages)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace while preserving document structure."""
        import re
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' *\n *', '\n', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()
