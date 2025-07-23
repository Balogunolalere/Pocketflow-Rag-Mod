"""
Basic usage example of the production RAG system.
"""

import sys
from pathlib import Path
# Add project root to path to allow imports from other directories
sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import ProductionRAGSystem
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

def basic_example():
    """Basic example using text documents."""
    # Initialize the RAG system
    rag_system = ProductionRAGSystem()
    
    # Sample documents (you can replace these with actual file paths)
    sample_docs = [
        "PocketFlow is a minimalist framework for building LLM applications with just 100 lines of code.",
        "Machine learning models require large amounts of data for training and validation.",
        "The Gemini API provides powerful language models for various AI applications."
    ]
    
    # For this example, we'll put the sample docs in the shared state manually
    shared = {
        "texts": sample_docs,
        "document_chunks": [{"text": doc, "metadata": {"source": "sample"}} for doc in sample_docs],
        "embeddings": None,
        "index": None
    }
    
    # Run offline flow to create embeddings and index
    rag_system.offline_flow.run(shared)
    
    # Query the documents
    query = "What is PocketFlow?"
    result = rag_system.query_documents(query, shared)
    
    print("Query:", query)
    print("Answer:", result["answer"])
    print("Sources used:", result["num_sources_used"])

def file_example():
    """Example using actual files (requires files to exist)."""
    import os
    
    # Check if we have any sample files
    test_files = ["test.txt", "sample.pdf", "example.docx"]
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if not existing_files:
        print("No test files found. Create test.txt with some content to test file parsing.")
        return
    
    # Initialize the RAG system
    rag_system = ProductionRAGSystem()
    
    # Run full pipeline
    query = "What is mentioned in these documents?"
    result = rag_system.run_full_pipeline(existing_files, query)
    
    print("Files processed:", existing_files)
    print("Query:", query)
    print("Answer:", result["query_result"]["answer"])

if __name__ == "__main__":
    print("Running basic example...")
    basic_example()
    
    print("\n" + "="*50)
    print("Running file example...")
    file_example()
