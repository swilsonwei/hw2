#!/usr/bin/env python3
"""
Test Runner for SmartInfo RAG System
Simple script to run integration tests with proper setup and reporting.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_server_running():
    """Check if the FastAPI server is running."""
    try:
        import requests
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_server():
    """Start the FastAPI server if not running."""
    if check_server_running():
        print("✅ Server is already running")
        return True
    
    print("🚀 Starting FastAPI server...")
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found. Please run: python3 -m venv venv")
        return False
    
    # Activate virtual environment and start server
    try:
        if os.name == 'nt':  # Windows
            activate_cmd = "venv\\Scripts\\activate"
            python_cmd = "venv\\Scripts\\python"
        else:  # Unix/Linux/macOS
            activate_cmd = "source venv/bin/activate"
            python_cmd = "venv/bin/python"
        
        # Start server in background
        server_process = subprocess.Popen(
            [python_cmd, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            if check_server_running():
                print("✅ Server started successfully")
                return True
        
        print("❌ Server failed to start within 30 seconds")
        server_process.terminate()
        return False
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

def install_dependencies():
    """Install required dependencies for testing."""
    print("📦 Installing test dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "requests", "python-dotenv"
        ], check=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_tests():
    """Run the integration tests."""
    print("🧪 Running RAG Integration Tests...")
    print("=" * 50)
    
    try:
        # Run the test script
        result = subprocess.run([
            sys.executable, "test_rag.py"
        ], capture_output=True, text=True)
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def main():
    """Main function to orchestrate the testing process."""
    print("🔬 SmartInfo RAG System - Integration Test Suite")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ main.py not found. Please run this script from the project root.")
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Start server if needed
    if not start_server():
        print("❌ Cannot run tests without a running server")
        return 1
    
    # Run tests
    success = run_tests()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        return 0
    else:
        print("\n💥 Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 