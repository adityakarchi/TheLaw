import os
from groq import Groq

# Initialize Groq client with API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")
client = Groq(api_key=GROQ_API_KEY)

def build_prompt(clause: str) -> str:
    return f"""
Simplify the following legal clause into plain English.
Avoid legal jargon. Be concise.

Clause:
\"\"\"{clause}\"\"\"

Simplified Explanation:
"""

def simplify_clause(clause: str) -> str:
    completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": build_prompt(clause)}],
    temperature=0.3,
    max_tokens=150
)

    return completion.choices[0].message.content.strip()
