"""Legal Document Simplifier - Core module exports."""

from src.simple_pipeline import analyze_input, quick_check, LegalDocumentSimplifier
from src.legal_detector import detect_legal_document, is_legal_document
from src.simplification import simplify_text
from src.preprocessing import read_pdf, clean_text

__version__ = "1.0.0"
__author__ = "Legal AI Team"

__all__ = [
    "analyze_input",
    "quick_check", 
    "LegalDocumentSimplifier",
    "detect_legal_document",
    "is_legal_document",
    "simplify_text",
    "read_pdf",
    "clean_text",
]