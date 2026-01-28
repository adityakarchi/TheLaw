"""Clause classification using fine-tuned Legal-BERT."""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/finetuned_clause_classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def classify_clause(text: str) -> str:
    """Classify legal clause type using Legal-BERT."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return model.config.id2label[pred_id]
