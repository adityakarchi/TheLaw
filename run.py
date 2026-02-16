"""Launch script for the Legal AI Assistant.

Usage:
    python run.py              # Start Streamlit UI
    python run.py --install    # Install dependencies first
"""

import subprocess
import sys
import os


def install_deps():
    """Install requirements."""
    print("📦 Installing dependencies …")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"
    ])
    print("✅ Dependencies installed\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Handle --install flag
    if "--install" in sys.argv:
        install_deps()

    # Check for .env
    if not os.path.exists(".env"):
        print("⚠️  Warning: .env file not found!")
        print("   Copy .env.example → .env and add your GROQ_API_KEY")
        print("   Get your free key at: https://console.groq.com/\n")

    print("🚀 Starting Legal AI Assistant …")
    print("   Architecture: RAG + LangGraph + LangChain + Groq")
    print("   Open: http://localhost:8501\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "streamlit_app.py",
        "--server.headless", "true",
    ])


if __name__ == "__main__":
    main()
