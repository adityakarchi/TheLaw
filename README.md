# ⚖️ Legal AI Assistant

> **Production-grade legal document analysis powered by RAG + LangGraph + LangChain + Groq**

Upload a contract (PDF or text) and get:
- **Legal detection** with confidence scoring
- **Plain-English simplification** via RAG-augmented LLM
- **Risk analysis** — risky clauses flagged with severity ratings
- **Interactive Q&A** — ask natural-language questions about the contract

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │────▶│  LangGraph   │────▶│   Output     │
│  Frontend    │     │  Workflow     │     │   Display    │
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
```

**LangGraph Workflow Nodes:**
1. `ingest_node` — Load document, chunk, build FAISS index
2. `legal_detect_node` — Weighted keyword detection + confidence scoring  
3. `retriever_node` — FAISS similarity search
4. `simplify_node` — RAG-augmented simplification chain
5. `risk_node` — Multi-query risk analysis chain
6. `qa_node` — Contract Q&A via RAG
7. `output_node` — Assemble final structured output

## 📁 Project Structure

```
legal/
├── streamlit_app.py          # Streamlit UI
├── run.py                    # Entry point
├── requirements.txt
├── .env.example
├── src/
│   ├── rag/                  # RAG pipeline
│   │   ├── loader.py         # PDF/text loader + LangChain chunking
│   │   ├── embedder.py       # sentence-transformers embeddings
│   │   ├── vectordb.py       # FAISS vector store
│   │   └── retriever.py      # High-level retrieval facade
│   ├── graph/                # LangGraph workflow
│   │   ├── state.py          # Typed state definition
│   │   └── workflow.py       # Graph nodes + routing
│   ├── chains/               # LangChain chains
│   │   ├── simplify_chain.py # Simplification chain
│   │   ├── risk_chain.py     # Risk analysis chain
│   │   └── qa_chain.py       # Contract Q&A chain
│   ├── utils/
│   │   └── config.py         # Centralised configuration
│   ├── legal_detector.py     # Legal document detection
│   └── preprocessing.py      # Text cleaning utilities
└── models/                   # Pre-trained classifiers
```

## 🚀 Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

3. **Run the app**
   ```bash
   python run.py
   # or: streamlit run streamlit_app.py
   ```

4. Open **http://localhost:8501**

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (Llama 3.1 8B) |
| Framework | LangChain + LangGraph |
| Vector DB | FAISS |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Frontend | Streamlit |
| Language | Python |

## 🎯 Features

- **📄 PDF & Text Support** — PyMuPDF + pypdf fallback
- **🔍 Legal Detection** — Weighted keyword scoring with confidence
- **✨ RAG Simplification** — Context-aware plain-English translation
- **⚠️ Risk Analysis** — Critical/High/Medium/Low severity ratings
- **💬 Contract Q&A** — Ask questions grounded in the document
- **📊 Confidence Scores** — Transparent classification reasoning
- **🏷️ Legal Term Detection** — Categorized by legal domain



This tool is for educational purposes only and does not constitute legal advice. Always consult a qualified professional for legal matters.
