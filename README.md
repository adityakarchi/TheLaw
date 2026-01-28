# ⚖️ Legal Document Simplifier

> **Transform complex legal documents into plain English with AI**

This tool analyzes legal documents (Text or PDF), detects legal content, and simplifies complex jargon into easy-to-understand explanations using Generative AI.

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/legal-document-simplifier.git
   cd legal-document-simplifier
   ```

2. **Install dependencies**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Setup API Key**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
   *(Get a free key at [console.groq.com](https://console.groq.com/))*

4. **Run the App**
   ```bash
   streamlit run streamlit_app.py
   ```

## 🎯 Features

- **📄 PDF & Text Support**
- **🔍 Auto-Detection**
- **✨ AI Simplification**
- **📊 Confidence Score**

## 💻 Usage

1. Open http://localhost:8501
2. Paste text or upload PDF
3. Click **Analyze Document**

## ⚠️ Disclaimer

This tool is for educational purposes only and does not constitute legal advice. Always consult a qualified professional for legal matters.
