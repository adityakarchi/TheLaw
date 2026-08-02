"""Clause Classification Pipeline — uses fine-tuned BERT model.

Loads the BERT model from models/finetuned_clause_classifier/ to classify
each clause into a type and assign a local risk score BEFORE sending to LLM.

Feature 1: Connect the Fine-tuned Clause Classifier.
"""

import logging
from pathlib import Path
from typing import Optional
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "finetuned_clause_classifier"

# Risk mapping: clause type → risk level + score
# Higher score = higher risk (0.0–1.0)
CLAUSE_RISK_MAP = {
    "Cap On Liability":   {"risk_level": "High",     "risk_score": 0.85},
    "Anti-Assignment":    {"risk_level": "High",     "risk_score": 0.75},
    "Insurance":          {"risk_level": "Medium",   "risk_score": 0.60},
    "Audit Rights":       {"risk_level": "Medium",   "risk_score": 0.55},
    "License Grant":      {"risk_level": "Medium",   "risk_score": 0.50},
    "Agreement Date":     {"risk_level": "Low",      "risk_score": 0.20},
    "Document Name":      {"risk_level": "Low",      "risk_score": 0.10},
    "Parties":            {"risk_level": "Low",       "risk_score": 0.15},
}

DEFAULT_RISK = {"risk_level": "Medium", "risk_score": 0.50}


class ClassifierPipeline:
    """BERT-based clause classifier with risk scoring."""

    def __init__(self, model_dir: Optional[str] = None):
        self._model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self._tokenizer = None
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self, model_dir: Optional[str] = None) -> None:
        """Load tokenizer + model once. Idempotent."""
        if self._loaded:
            return

        path = Path(model_dir) if model_dir else self._model_dir
        if not path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")

        logger.info(f"Loading clause classifier from {path}")
        self._tokenizer = AutoTokenizer.from_pretrained(str(path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(path))
        self._model.eval()
        self._loaded = True
        logger.info(f"Clause classifier loaded — {len(self._model.config.id2label)} classes")

    def _ensure_loaded(self) -> None:
        """Lazy-load model if not already loaded."""
        if not self._loaded:
            self.load_model()

    def classify_clause(self, text: str) -> dict:
        """Classify a single clause.

        Returns:
            {"type": str, "confidence": float, "risk_score": float, "risk_level": str}
        """
        self._ensure_loaded()

        if not text or len(text.strip()) < 10:
            return {
                "type": "Unknown",
                "confidence": 0.0,
                "risk_score": 0.0,
                "risk_level": "Low",
            }

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_id].item()

        clause_type = self._model.config.id2label.get(str(pred_id), f"class_{pred_id}")
        risk_info = CLAUSE_RISK_MAP.get(clause_type, DEFAULT_RISK)

        return {
            "type": clause_type,
            "confidence": round(confidence, 4),
            "risk_score": risk_info["risk_score"],
            "risk_level": risk_info["risk_level"],
        }

    def classify_batch(self, clauses: list[str]) -> list[dict]:
        """Classify multiple clauses.

        Args:
            clauses: List of clause text strings.

        Returns:
            List of classification dicts (same format as classify_clause).
        """
        self._ensure_loaded()

        if not clauses:
            return []

        # Filter out empty clauses
        results = []
        valid_indices = []
        valid_texts = []

        for i, text in enumerate(clauses):
            if text and len(text.strip()) >= 10:
                valid_indices.append(i)
                valid_texts.append(text)
            else:
                results.append((i, {
                    "type": "Unknown",
                    "confidence": 0.0,
                    "risk_score": 0.0,
                    "risk_level": "Low",
                }))

        if valid_texts:
            inputs = self._tokenizer(
                valid_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )

            with torch.no_grad():
                outputs = self._model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)
            pred_ids = torch.argmax(probs, dim=1)

            for j, idx in enumerate(valid_indices):
                pred_id = pred_ids[j].item()
                confidence = probs[j, pred_id].item()
                clause_type = self._model.config.id2label.get(str(pred_id), f"class_{pred_id}")
                risk_info = CLAUSE_RISK_MAP.get(clause_type, DEFAULT_RISK)

                results.append((idx, {
                    "type": clause_type,
                    "confidence": round(confidence, 4),
                    "risk_score": risk_info["risk_score"],
                    "risk_level": risk_info["risk_level"],
                }))

        # Sort by original index and return just the dicts
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def format_for_risk_prompt(self, clauses: list[str]) -> str:
        """Classify clauses and format as structured input for the risk LLM prompt.

        Returns a string like:
          HIGH RISK CLAUSES:
          - [Cap On Liability] (confidence: 92%): "The total liability..."
          MEDIUM RISK CLAUSES:
          - [Insurance] (confidence: 78%): "The party shall maintain..."
        """
        classifications = self.classify_batch(clauses)

        high_risk = []
        medium_risk = []
        low_risk = []

        for clause_text, clf in zip(clauses, classifications):
            snippet = clause_text[:200].strip()
            entry = f'- [{clf["type"]}] (confidence: {int(clf["confidence"]*100)}%): "{snippet}"'

            if clf["risk_level"] == "High":
                high_risk.append(entry)
            elif clf["risk_level"] == "Medium":
                medium_risk.append(entry)
            else:
                low_risk.append(entry)

        parts = []
        if high_risk:
            parts.append("HIGH RISK CLAUSES (review carefully):\n" + "\n".join(high_risk))
        if medium_risk:
            parts.append("MEDIUM RISK CLAUSES:\n" + "\n".join(medium_risk))
        if low_risk:
            parts.append("LOW RISK CLAUSES:\n" + "\n".join(low_risk))

        return "\n\n".join(parts) if parts else "No clauses classified."


# Module-level cached singleton
_classifier: Optional[ClassifierPipeline] = None


def get_classifier() -> ClassifierPipeline:
    """Return the cached classifier singleton."""
    global _classifier
    if _classifier is None:
        _classifier = ClassifierPipeline()
    return _classifier
