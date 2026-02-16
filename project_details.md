# Legal AI Assistant - Comprehensive Project Documentation

## 1. Executive Summary
**Project Name:** Legal AI Assistant  
**Status:** Production Ready  
**Core Function:** Automated legal document analysis, risk assessment, and simplification.  
**Target Audience:** Non-lawyers needing to understand contracts, or legal professionals needing quick first-pass reviews.

This application essentially functions as a "AI Junior Associate." It takes a PDF or text contract, verifies it is actually a legal document, and then performs three distinct parallel tasks:
1.  **Simplification:** Translates legalese into plain English.
2.  **Risk Analysis:** Identifies dangerous clauses like unlimited liability.
3.  **Interactive Q&A:** Allows the user to chat with the document.

---

## 2. Directory Structure & File Map

```
legal/
├── streamlit_app.py          # [ENTRY POINT] The main user interface.
├── requirements.txt          # Python dependencies (Streamlit, LangChain, FAISS, etc).
├── .env.example              # Template for environment variables (API keys).
├── README.md                 # Quick start guide.
├── src/                      # Source Code
│   ├── graph/                # [BRAIN] Orchestrates the workflow.
│   │   ├── workflow.py       # Defines nodes (steps) and edges (logic) of the pipeline.
│   │   ├── state.py          # Defines the data structure passed between nodes.
│   ├── rag/                  # [MEMORY] Retrieval Augmented Generation components.
│   │   ├── loader.py         # Handles PDF parsing and text chunking.
│   │   ├── embedder.py       # Converts text to numbers (vectors).
│   │   ├── vectordb.py       # Manages the FAISS database operations.
│   │   ├── retriever.py      # Facade that combines Loader+Embedder+VectorDB.
│   ├── chains/               # [REASONING] Specific LLM prompts and chains.
│   │   ├── simplify_chain.py # Prompts for simplifying text.
│   │   ├── risk_chain.py     # Prompts for identifying risks.
│   │   ├── qa_chain.py       # Prompts for Q&A chat.
│   ├── utils/                # [UTILITIES] Helper functions.
│   │   ├── config.py         # Central configuration and API key validation.
│   ├── legal_detector.py     # [GATEKEEPER] Algorithmic check for legal document validity.
```

---

## 3. Detailed Architecture

The system is built on a **Graph-based Architecture** (using LangGraph) rather than a linear script. This allows for complex, conditional logic.

### A. The RAG Pipeline (Retrieval Augmented Generation)
This is the engine that allows the AI to "read" the document.
1.  **Ingestion:** The document is loaded using `PyMuPDF` (for PDFs) to ensure high-fidelity text extraction.
2.  **Chunking:** The text is split into smaller pieces ("chunks") of ~1000 characters. We use `RecursiveCharacterTextSplitter` which respects paragraph and sentence boundaries, ensuring we don't cut a legal clause in half.
3.  **Embedding:** Each chunk is converted into a list of 384 numbers (a vector) using the `sentence-transformers/all-MiniLM-L6-v2` model. This model represents the *meaning* of the text.
4.  **Indexing:** These vectors are stored in a `FAISS` index (Facebook AI Similarity Search). This allows us to find relevant chunks in milliseconds.
5.  **Retrieval:** When the AI needs to answer a question or find a risk, it converts the query into vectors, searches the FAISS index for the most similar chunks, and feeds *only those chunks* to the LLM.

### B. The Workflow Graph
The application flow is defined as a directed graph:
1.  **`ingest_node`**: Loads file -> Chunks -> FAISS Index.
2.  **`legal_detect_node`**: Runs the keyword algorithm.
    *   *Conditional Edge:* If `is_legal == False` → Stop and show error.
    *   *Conditional Edge:* If `is_legal == True` → Proceed to Analysis.
3.  **Parallel Execution**:
    *   **`simplify_node`**: Retrieves general context and summarizes.
    *   **`risk_node`**: Retrieves specific risk context ("liability", "termination") and analyzes.
4.  **`qa_node`**: (Interactive) Runs only when user types a message.

### C. The Intelligence (LLM)
*   **Model:** Llama 3 (via Groq API).
*   **Why Groq?** Groq provides specific hardware (LPUs) that makes inference incredibly fast, essential for a responsive UI.
*   **Prompt Engineering:** We use "persona-based" prompting (e.g., "You are a senior risk analyst").

---

## 4. Key Features & Capabilities

### 1. Legal Document Detection
*   **Goal:** Prevent users from uploading non-legal text (like a cooking recipe) and wasting API credits.
*   **Method:** A weighted keyword density algorithm. It looks for words like "whereas", "indemnify", "jurisdiction".
*   **Threshold:** If the density of these words is too low, the document is rejected.

### 2. Intelligent Simplification
*   **Goal:** Make contracts readable.
*   **Method:** It doesn't just shorten the text. It uses RAG to pull the full context and then rewrites it in plain English, ensuring no obligations or rights are lost in translation.

### 3. Risk Analysis
*   **Goal:** Find "gotchas".
*   **Method:** It performs targeted searches for 4 specific categories:
    *   Liability & Damages
    *   Termination Rights
    *   Indemnification
    *   Non-compete & Restrictions
*   It then rates the risk level (Critical/High/Medium/Low) and provides negotiation advice.

---

## 5. Deployment Information

### Environment Variables (.env)
*   `GROQ_API_KEY`: Required for the LLM.
*   `LLM_MODEL`: Defaults to `llama-3.1-8b-instant`.
*   `CHUNK_SIZE`: Controls RAG granularity (default 1000).

### Requirements
*   **Python:** 3.10+
*   **Streamlit:** For the UI.
*   **LangChain Components:** `langchain`, `langchain-groq`, `langchain-community`.
*   **Vector Store:** `faiss-cpu`.
*   **PDF:** `pymupdf`.

### Deployment Target
*   **Streamlit Cloud:** Compatible. Requires `packages.txt` for system-level dependencies if any (none currently required for this pure-python stack).
