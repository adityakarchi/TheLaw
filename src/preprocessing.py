"""Preprocessing: PDF extraction, text cleaning, validation."""

import re
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass


def read_pdf(file_path_or_buffer) -> str:
    """Extract text from PDF with error handling."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise PreprocessingError("pypdf library not installed. Run: pip install pypdf")
    
    try:
        reader = PdfReader(file_path_or_buffer)
        
        if len(reader.pages) == 0:
            raise PreprocessingError("PDF has no pages")
        
        text_parts = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {i+1}: {e}")
                continue
        
        if not text_parts:
            raise PreprocessingError("No text could be extracted from PDF")
        
        return "\n".join(text_parts).strip()
        
    except PreprocessingError:
        raise
    except Exception as e:
        raise PreprocessingError(f"Failed to read PDF: {str(e)}")


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    
    # Remove excessive newlines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Clean up line breaks
    text = re.sub(r' *\n *', '\n', text)
    
    # Remove control characters except newlines
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()


def validate_input(text: str, min_length: int = 50, max_length: int = 100000) -> Tuple[bool, str]:
    """Validate input text length and content."""
    if not text:
        return False, "Input text is empty"
    
    if len(text.strip()) < min_length:
        return False, f"Text too short. Minimum {min_length} characters required."
    
    if len(text) > max_length:
        return False, f"Text too long. Maximum {max_length} characters allowed."
    
    # Check if it's mostly non-text (binary data)
    printable_ratio = sum(c.isprintable() or c.isspace() for c in text) / len(text)
    if printable_ratio < 0.8:
        return False, "Input appears to contain binary or corrupted data"
    
    return True, ""


def extract_text_from_input(
    input_data,
    input_type: str = "text"
) -> Tuple[str, Optional[str]]:
    """Extract and clean text from text or PDF input."""
    try:
        if input_type == "pdf":
            raw_text = read_pdf(input_data)
        else:
            raw_text = str(input_data)
        
        cleaned_text = clean_text(raw_text)
        
        is_valid, error = validate_input(cleaned_text)
        if not is_valid:
            return "", error
        
        return cleaned_text, None
        
    except PreprocessingError as e:
        return "", str(e)
    except Exception as e:
        logger.exception("Unexpected error during text extraction")
        return "", f"Unexpected error: {str(e)}"


def truncate_text(text: str, max_tokens: int = 4000, chars_per_token: float = 4.0) -> str:
    """Truncate text to approximate token limit for LLM."""
    max_chars = int(max_tokens * chars_per_token)
    if len(text) <= max_chars:
        return text
    
    # Try to truncate at sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1]
    
    return truncated + "..."
