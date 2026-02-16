# Codebase Explanation: Line-by-Line Breakdown

This document provides a deep-dive technical explanation of the core files in the project.

---

## 1. `streamlit_app.py` (The Interface)
**Role:** The Frontend & Controller.

*   **Page Config (`st.set_page_config`):** Sets up the browser tab and wide layout.
*   **State Management (`st.session_state`):** Crucial for Streamlit.
    *   `analysis_result`: Stores the heavy computation output (risk analysis, summary) so it doesn't vanish when you click a button.
    *   `messages`: Stores the chat history for the Q&A feature.
*   **Inputs:**
    *   Accepts PDF uploads or raw text paste.
    *   The "Analyze" button triggers the `run_full_analysis` function.
*   **Rendering:**
    *   **Tabs:** Splits view into Summary, Risk, and Chat.
    *   **Risk Renderer:** loops through the structured risk data and uses `st.error` for critical risks and `st.warning` for others.
    *   **Chat Interface:** Uses `st.chat_message` to create a WhatsApp-style conversation view.

---

## 2. `src/graph/workflow.py` (The Orchestrator)
**Role:** The logic flow controller.

*   **`GraphState` (TypedDict):** The "packet" of data that travels through the system. It accumulates data: begins with just `raw_text`, adds `is_legal`, adds `risk_analysis`, etc.
*   **`_get_retriever()` Singleton:**
    *   **Logic:** `if _retriever is None: _retriever = LegalRetriever()`.
    *   **Why:** Loading the embedding model takes 2-3 seconds. We don't want to do this every time a function is called. We do it once per session.
*   **`ingest_node(state)`:**
    *   Calls `retriever.ingest(text)`.
    *   Returns updated state with `retriever_ready=True`.
*   **`legal_detect_node(state)`:**
    *   Runs the heuristic check.
    *   Returns `state` with `is_legal` (bool) and `legal_explanation` (string).
*   **`risk_node` & `simplify_node`:**
    *   These run independently. They both access the *same* `retriever` singleton to get data, but ask different questions.

---

## 3. `src/rag/retriever.py` (The Retrieval Facade)
**Role:** A unified interface for all RAG operations.

*   **`__init__`**: Initializes three components:
    1.  `DocumentLoader`: To parse files.
    2.  `EmbeddingPipeline`: To calculate vectors.
    3.  `VectorStore`: To store vectors.
*   **`ingest(input_data)`**:
    1.  `loader.load()`: Gets text.
    2.  `loader.split_into_chunks()`: Gets chunks.
    3.  `vector_store.build_from_documents()`: Embeds and Indexes.
*   **`get_context_string(query)`**:
    *   A helper function for the chains.
    *   Takes a query ("liability"), searches FAISS, gets top 4 chunks, and joins them into a single string to paste into the LLM prompt.

---

## 4. `src/rag/loader.py` (The Data Processor)
**Role:** PDF Parsing and Text Chunking.

*   **`DocumentLoader` Class**:
    *   **`load_pdf`**: Uses `fitz` (PyMuPDF). It iterates through pages and extracts text. It's wrapped in a try/except to fall back to `pypdf` if `fitz` fails.
    *   **`split_into_chunks`**: Uses `RecursiveCharacterTextSplitter`.
        *   `separators=["\n\n\n", "\n\n", "ARTICLE ", ...]`: This custom list is vital. It tells the splitter to try breaking at "ARTICLE" or paragraph breaks *before* breaking mid-sentence.
        *   `chunk_size=1000`: Large enough to contain a full legal clause.
        *   `chunk_overlap=200`: Ensures context isn't lost at the edges of chunks.

---

## 5. `src/chains/*` (The LLM Prompts)
**Role:** The specific instructions for Llama 3.

*   **`simplify_chain.py`**:
    *   **Prompt:** "You are a senior legal translator... preserve ALL substantive legal meaning."
    *   **Logic:** It explicitly warns the LLM *not* to hallucinate (invent) facts.
*   **`risk_chain.py`**:
    *   **Prompt:** "You are a senior contract risk analyst... protect your client's interests."
    *   **Structure:** It forces the output into a specific Markdown format (`## Risk Summary`, `### [Risk Title]`) so the Streamlit app can easily parse/display it.
*   **`qa_chain.py`**:
    *   **Prompt:** "Yields answers based ONLY on the provided context."
    *   **Safety:** "If the answer is not in the context, say 'I cannot find this information'." This prevents the AI from making up laws or contract terms.

---

## 6. `src/legal_detector.py` (The Gatekeeper)
**Role:** A fast, cheap filter for non-legal documents.

*   **`LEGAL_TERMS_WEIGHTED`**: A massive dictionary of words mapped to importance weights.
    *   High Value (2.0): "witnesseth", "force majeure".
    *   Medium Value (1.0): "party", "agreement".
    *   Low Value (0.5): "term", "section".
*   **`detect_legal_document`**:
    *   Counts hits for these words.
    *   Calculates a `confidence` score based on density (hits per total words).
    *   If score > 0.5, it passes.

---

## 7. `src/utils/config.py` (Configuration)
**Role:** Central logic for settings and keys.

*   **`get_llm()` with `@lru_cache`**:
    *   This is a performance optimization. It ensures we create the `ChatGroq` object (connection pool) only once, not every time we send a message.
*   **Environment loading**: Uses `python-dotenv` to safely load API keys from `.env` without hardcoding them in the source.
