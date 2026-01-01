Legal Document Simplifier

An AI-powered application that analyzes legal documents and converts complex legal clauses into clear, plain English.
The system combines machine learning, transformer-based NLP, and large language models to classify clauses, assess risk, and simplify legal language through an interactive web interface.

Features

Upload legal documents as PDF or paste raw text

Automatic clause segmentation

Clause classification using a fine-tuned Legal-BERT model

Risk assessment with explainable scoring (High / Medium / Low)

Plain-English clause simplification using Groq LLM

Interactive, single-page UI built with Streamlit

System Overview
Input (PDF / Text)
        ↓
Preprocessing & Cleaning
        ↓
Clause Segmentation
        ↓
Clause Classification (Legal-BERT)
        ↓
Risk Analysis
        ↓
LLM-based Simplification
        ↓
Streamlit Interface


The backend is modular and designed for easy extension or migration to an API-based service.

Project Structure
legal-doc-simplifier/
├── notebooks/
│   ├── 00_eda.ipynb
│   └── 07_finetune_clause_classifier.ipynb
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── classification.py
│   ├── risk.py
│   ├── simplification.py
│   └── pipeline.py
├── models/
│   └── finetuned_clause_classifier/
├── data/
│   └── samples/
├── streamlit_app/
│   └── app.py
├── requirements.txt
└── README.md

Model Development

Built a baseline TF-IDF + Logistic Regression classifier

Performed exploratory data analysis (EDA) on legal clauses

Fine-tuned Legal-BERT on the CUAD dataset for improved accuracy

Integrated the trained model into a reusable inference pipeline

Technology Stack

Python

Machine Learning & NLP: TF-IDF, Logistic Regression, Legal-BERT

LLM Integration: Groq API

Frontend: Streamlit

Dataset: CUAD (Contract Understanding Atticus Dataset)

Setup Instructions

Clone the repository

git clone https://github.com/your-username/legal-doc-simplifier.git
cd legal-doc-simplifier


Create and activate environment

conda create -n legalai python=3.10
conda activate legalai


Install dependencies

pip install -r requirements.txt


Set Groq API key

export GROQ_API_KEY="your_api_key"


Run the application

streamlit run streamlit_app/app.py

Use Cases

Simplifying legal contracts for non-legal users

Identifying risky clauses in agreements

Demonstrating applied NLP and ML system design

Author

Developed as a placement-focused project demonstrating end-to-end NLP system design, model fine-tuning, and deployment.
