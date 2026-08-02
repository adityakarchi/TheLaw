# ⚖️ Legal AI Assistant

> **Production-grade legal document analysis powered by RAG + LangGraph + LangChain + Groq + AWS S3**

Upload a contract (PDF or text) and get:
- **Legal detection** with confidence scoring
- **Plain-English simplification** via RAG-augmented LLM
- **Risk analysis** — risky clauses flagged with severity ratings
- **Interactive Q&A** — ask natural-language questions about the contract
- **☁️ AWS S3 backup** — every document automatically saved with AES-256 encryption

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │────▶│  LangGraph   │────▶│   Output     │
│  Frontend    │     │  Workflow    │     │   Display    │
└─────────────┘     └──────┬───────┘     └──────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Legal   │ │   RAG    │ │ LangChain│
        │ Detector │ │ Pipeline │ │  Chains  │
        └──────────┘ └────┬─────┘ └──────────┘
                          │
                ┌─────────┼─────────┐
                ▼         ▼         ▼
          ┌────────┐ ┌────────┐ ┌────────┐
          │ Loader │ │Embedder│ │ FAISS  │
          │(PyMuPDF)│ │(MiniLM)│ │VectorDB│
          └────────┘ └────────┘ └────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   AWS S3    │
                   │  (Mumbai)   │
                   │ AES-256 enc │
                   └─────────────┘
```

## 📁 Project Structure

```
legal/
├── streamlit_app.py          # Streamlit UI + S3 integration
├── run.py                    # Entry point
├── requirements.txt
├── .env.example              # Copy to .env and fill in keys
├── src/
│   ├── rag/                  # RAG pipeline
│   │   ├── loader.py
│   │   ├── embedder.py
│   │   ├── vectordb.py
│   │   └── retriever.py
│   ├── graph/                # LangGraph workflow
│   │   ├── state.py
│   │   └── workflow.py
│   ├── chains/               # LangChain chains
│   │   ├── simplify_chain.py
│   │   ├── risk_chain.py
│   │   └── qa_chain.py
│   ├── utils/
│   │   ├── config.py         # Centralised config + AWS status
│   │   └── s3_storage.py     # S3 upload / list helpers
│   ├── legal_detector.py
│   └── preprocessing.py
└── models/
```

## 🚀 Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set up environment**
```bash
cp .env.example .env
# Edit .env — add GROQ_API_KEY at minimum
# Add AWS keys to enable S3 document storage
```

**3. Run the app**
```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501**

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (Llama 3.1 8B Instant) |
| Framework | LangChain + LangGraph |
| Vector DB | FAISS (local, per-session) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Frontend | Streamlit |
| Storage | AWS S3 (ap-south-1, AES-256) |
| Language | Python 3.10+ |

## ☁️ AWS S3 Setup

**1. Create S3 bucket** (ap-south-1, block all public access)

**2. Create IAM user** with this policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket", "s3:DeleteObject"],
    "Resource": [
      "arn:aws:s3:::YOUR-BUCKET-NAME",
      "arn:aws:s3:::YOUR-BUCKET-NAME/*"
    ]
  }]
}
```

**3. Add to `.env`:**
```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=ap-south-1
S3_BUCKET_NAME=your-bucket-name
```

S3 is **optional** — the app works without it.

## 🎯 Features

- **📄 PDF & Text Support** — PyMuPDF + pypdf fallback
- **🔍 Legal Detection** — Weighted keyword scoring with confidence
- **✨ RAG Simplification** — Context-aware plain-English translation
- **⚠️ Risk Analysis** — Critical/High/Medium/Low severity ratings
- **💬 Contract Q&A** — Ask questions grounded in the document
- **📊 Confidence Scores** — Transparent classification reasoning
- **☁️ S3 Storage** — Auto-backup with encryption and upload history

---

> ⚠️ For educational purposes only. Does not constitute legal advice. Always consult a qualified professional.
