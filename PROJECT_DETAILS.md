# Legal Document Simplifier - Project Documentation

## 1. Project Overview
The **Legal Document Simplifier** is an AI-powered application designed to make complex legal documents (contracts, agreements, terms of service) understandable for non-lawyers. It uses a combination of **keyword-based detection algorithms** and **Large Language Models (LLMs)** to identify if a text is legal in nature and then simplify it into plain English.

---

## 2. Directory Structure & Key Files

The project follows a modular architecture where the core logic is separated from the User Interface (UI).

```
legal/
├── streamlit_app.py        # PRIMARY ENTRY POINT: The main Web GUI application.
├── src/                    # CORE SOURCE CODE MODULES
│   ├── simple_pipeline.py  # MAIN ORCHESTRATOR: Connects all other modules together.
│   ├── legal_detector.py   # IDENTIFICATION LOGIC: Determines if text is a legal doc.
│   ├── simplification.py   # AI LOGIC: Connects to Groq/Llama3 for rewriting text.
│   ├── preprocessing.py    # FILE HANDLING: Extracts text from PDFs and cleans inputs.
│   ├── config.py           # SETTINGS: Manages API keys and environment variables.
│   ├── classification.py   # (Utility) Fine-grained clause classification (BERT).
│   ├── segmentation.py     # (Utility) Splits text into individual clauses.
│   └── risk.py             # (Utility) Calculates risk scores based on keywords.
├── requirements.txt        # DEPENDENCIES: List of Python libraries needed.
└── README.md               # QUICK START GUIDE.
```

---

## 3. detailed Module Breakdown

### A. The User Interface (`streamlit_app.py`)
*   **What it does:** This is the face of the application. It creates the website where users can upload files or paste text.
*   **Key Terms:**
    *   **Streamlit:** The Python library used to build the web page.
    *   **Session State:** Keeps track of your data (the uploaded file, the analysis results) so it doesn't disappear when you click buttons.
*   **Workflow:**
    1.  User enters API Key (or checks environment).
    2.  User uploads PDF or pastes text.
    3.  App calls `src.simple_pipeline.analyze_input`.
    4.  App displays results: "Likelihood of being legal", "Key Terms Found", and the "Simplified Version".

### B. The Orchestrator (`src/simple_pipeline.py`)
*   **What it does:** The "Brain" of the operation. It decides the order of operations.
*   **Key Logic (`LegalDocumentSimplifier` class):**
    1.  **Lazy Loading:** It only imports heavy modules (like the detector or simplifier) when they are actually needed, making the app start faster.
    2.  **Step-by-Step Analysis:**
        *   *Step 1:* Call `preprocessing` to get clean text.
        *   *Step 2:* Call `detector` to check if it's worth analyzing. If it's not a legal doc, it stops here to save money/time.
        *   *Step 3:* Call `simplification` to rewrite the text using AI.
    3.  **AnalysisResult:** A data structure that packages the answer (Status, Original Text, Simplified Text, Confidence Score) into one object.

### C. Legal Detection (`src/legal_detector.py`)
*   **What it does:** Determines if the input text is actually a legal document or just random text (like a baking recipe).
*   **How it works:**
    *   **Weighted Keywords:** It doesn't just count words; it gives them "points".
        *   *High Value (2.0):* "witnesseth", "indemnification".
        *   *Medium Value (1.5):* "agreement", "contract", "shall".
    *   **Scoring:** It calculates a density score (legal words per total words). If the score crosses a threshold, it is marked as "Legal".

### D. AI Simplification (`src/simplification.py`)
*   **What it does:** The "Translator". It takes complex legalese and turns it into simple English.
*   **Key Terms:**
    *   **Groq API:** The service provider for the AI model.
    *   **Llama-3:** The specific Artificial Intelligence model used to read and rewrite the text.
    *   **System Prompt:** The instruction given to the AI (e.g., *"You are a legal expert... rewrite this for a 5th grader..."*).
*   **Process:**
    1.  It constructs a prompt with your legal text.
    2.  Sends it to the Groq API.
    3.  Receives the simplified text back.

### E. Preprocessing (`src/preprocessing.py`)
*   **What it does:** The "Janitor". It prepares files for analysis.
*   **Key Functions:**
    *   `read_pdf`: Uses the `pypdf` library to open PDF files and pull out readable text.
    *   `clean_text`: Removes weird characters, extra spaces, or formatting headers that might confuse the AI.
    *   `truncate_text`: Cuts off text if it is too long for the AI to handle in one go (Token limit management).

### F. Utility Modules (Legacy/Advanced)
These files provide specific NLP capabilities that can be integrated if more granular analysis is needed:
*   `src/classification.py`: Uses a Transformer model (BERT) to look at a specific sentence and say "This is a Termination Clause" or "This is a Liability Clause".
*   `src/segmentation.py`: Uses Regex (Regular Expressions) to chop a long paragraph into individual sentences or clauses.
*   `src/risk.py`: Scans for "scary" words (like "unlimited liability") to give the document a Risk Score (High/Medium/Low).

---

## 4. End-to-End Data Flow

1.  **Input:** User uploads `Contract.pdf` on the Streamlit page.
2.  **Extraction:** `streamlit_app.py` sends the file to `src.preprocessing.read_pdf` -> returns raw text string.
3.  **Pipeline Start:** Text is sent to `src.simple_pipeline.analyze`.
4.  **Verification:** `src.legal_detector` reads the text.
    *   *If score < threshold:* Returns "Not a legal document". Stop.
    *   *If score > threshold:* Continue.
5.  **Simplification:** `src.simple_pipeline` sends the text to `src.simplification`.
    *   The module formats the query for Groq API.
    *   Groq returns the simplified version.
6.  **Display:** `streamlit_app.py` receives the final package and prints it on the screen.

---

## 5. Deployment & Configuration

*   **API Keys:** The project requires a `GROQ_API_KEY`. This is managed in `src/config.py`. It looks for this key in your system's environment variables or `.env` file.
*   **Dependencies:** All external libraries (like `streamlit`, `groq`, `pypdf`) are listed in `requirements.txt`.
