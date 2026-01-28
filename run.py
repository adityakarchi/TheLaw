
import subprocess
import sys
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if not os.path.exists(".env"):
        print("⚠️  Warning: .env file not found!")
        print("   Get API key at: https://console.groq.com/")
    
    print("🚀 Starting Legal Document Simplifier...")
    print("   Open http://localhost:8501")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_app.py",
        "--server.headless", "true"
    ])

if __name__ == "__main__":
    main()
