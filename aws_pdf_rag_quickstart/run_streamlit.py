#!/usr/bin/env python3
"""
Startup script for the Streamlit application.
Sets up environment and runs the Streamlit app using uv.
"""

import os
import sys
import subprocess

def setup_environment():
    """Set up environment variables if not already set"""
    
    # Default environment variables
    defaults = {
        "LOG_LEVEL": "INFO",
        "LOCAL": "1",  # Default to local mode
        "INDEX_NAME": "rag-index",
        "AOSS_HOST": "localhost",
        "AOSS_PORT": "9200",
    }
    
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"Set {key}={value}")

def main():
    """Main function to run the Streamlit app"""
    
    print("🛡️ AWS RAG Chatbot with PII Protection - Streamlit + uv")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Print current configuration
    print("\nCurrent Configuration:")
    print(f"  LOCAL: {os.getenv('LOCAL')}")
    print(f"  LOG_LEVEL: {os.getenv('LOG_LEVEL')}")
    print(f"  INDEX_NAME: {os.getenv('INDEX_NAME')}")
    
    # Run Streamlit with uv
    print("\nStarting Streamlit application with uv...")
    print("Access the app at: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application")
    print("=" * 60)
    
    try:
        subprocess.run([
            "uv", "run", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
        
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
    except Exception as e:
        print(f"\nError running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 