#!/usr/bin/env python3
"""
Debug script for RAG system issues.
Run this to diagnose common problems.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

def check_environment():
    """Check environment setup"""
    print("🔍 Environment Check")
    print("-" * 40)
    
    # Load environment
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("✅ GEMINI_API_KEY is set")
        print(f"   Key preview: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else ''}")
    else:
        print("❌ GEMINI_API_KEY is not set")
        print("   Create a .env file with: GEMINI_API_KEY=your-api-key")
    
    print()

def check_index_files():
    """Check index file integrity"""
    print("📁 Index Files Check")
    print("-" * 40)
    
    index_files = [
        "rag_index.faiss",
        "rag_index_texts.json", 
        "rag_index_metadata.json",
        "rag_index_chunks.json"
    ]
    
    all_exist = True
    for file in index_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"✅ {file} ({size} bytes)")
            
            # Check if JSON files have content
            if file.endswith('.json'):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            print(f"   Contains {len(data)} items")
                        elif isinstance(data, dict):
                            print(f"   Contains {len(data)} keys")
                except Exception as e:
                    print(f"   ⚠️  Error reading JSON: {e}")
        else:
            print(f"❌ {file} (missing)")
            all_exist = False
    
    if not all_exist:
        print("\n💡 Missing index files. Try rebuilding:")
        print("   rm rag_index*")
        print("   uv run main.py --query 'test' --docs ./documents")
    
    print()

def check_documents():
    """Check document directory"""
    print("📄 Documents Check")
    print("-" * 40)
    
    docs_dir = Path("documents")
    if docs_dir.exists():
        files = list(docs_dir.glob("*"))
        if files:
            print(f"✅ Found {len(files)} files in documents/")
            for file in files[:5]:  # Show first 5
                size = file.stat().st_size
                print(f"   - {file.name} ({size} bytes)")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more files")
        else:
            print("❌ Documents directory is empty")
    else:
        print("❌ Documents directory doesn't exist")
        print("   Create it with: mkdir documents")
        print("   Add PDF, DOCX, TXT files to process")
    
    print()

def test_basic_imports():
    """Test if all dependencies can be imported"""
    print("📦 Imports Check")
    print("-" * 40)
    
    try:
        import pocketflow
        print("✅ pocketflow imported successfully")
    except ImportError as e:
        print(f"❌ pocketflow import failed: {e}")
        print("   Try: uv sync")
    
    try:
        import faiss
        print("✅ faiss imported successfully")
    except ImportError as e:
        print(f"❌ faiss import failed: {e}")
        print("   This should be installed with uv sync")
    
    try:
        from google import genai
        print("✅ Google AI SDK imported successfully")
    except ImportError as e:
        print(f"❌ Google AI SDK import failed: {e}")
        print("   This should be installed with uv sync")
    
    print()

def suggest_solutions():
    """Suggest solutions based on findings"""
    print("💡 Suggested Solutions")
    print("-" * 40)
    
    print("For 'Operation failed:' with empty message:")
    print("1. Check API key is set correctly")
    print("2. Try verbose mode: --verbose")
    print("3. Rebuild index if corrupted:")
    print("   rm rag_index*")
    print("   uv run main.py --query 'test' --docs ./documents --verbose")
    print()
    
    print("For immediate testing:")
    print("1. Simple test: uv run main.py --query 'hello' --use-existing-index --verbose")
    print("2. Example script: uv run examples/query_existing_index.py")
    print("3. Interactive mode: uv run examples/query_existing_index.py --interactive")

def main():
    """Run all diagnostic checks"""
    print("🔧 RAG System Diagnostic Tool")
    print("=" * 50)
    print()
    
    check_environment()
    check_index_files()
    check_documents()
    test_basic_imports()
    suggest_solutions()
    
    print("\n" + "=" * 50)
    print("🏁 Diagnostic complete!")

if __name__ == "__main__":
    main()
