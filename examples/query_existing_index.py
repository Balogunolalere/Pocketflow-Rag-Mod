"""
Query existing FAISS index for document summarization and Q&A.
This script demonstrates how to use a pre-built index without re-indexing documents.
"""

import sys
from pathlib import Path
# Add project root to path to allow imports from other directories
sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import ProductionRAGSystem
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_existing_index():
    """Query the existing FAISS index for various types of questions."""
    
    # Initialize the RAG system
    rag_system = ProductionRAGSystem()
    
    # List of different types of queries to test
    queries = [
        # Summarization queries
        "Provide a comprehensive summary of the main topics covered in the documents",
        "What are the key findings and conclusions from this research?",
        "Summarize the methodology used in this study",
        
        # Specific technical queries
        "What factors affect latency in video streaming?",
        "What optimization techniques are discussed?",
        "What are the main challenges in video streaming latency?",
        
        # Analysis queries
        "What are the results and discussion points?",
        "What literature was reviewed in this research?",
    ]
    
    print("🔍 Querying existing FAISS index...")
    print(f"Index contains {51} document chunks\n")
    print("="*80)
    
    for i, query in enumerate(queries, 1):
        try:
            print(f"\n📋 Query {i}: {query}")
            print("-" * 60)
            
            # Query the existing index (this will automatically load the .faiss file)
            result = rag_system.query_documents(query)
            
            print(f"✅ Answer: {result['answer']}")
            print(f"📊 Sources used: {result['num_sources_used']}")
            
            # Show some source information if available
            if 'retrieved_documents' in result:
                print(f"🔗 Top sources:")
                for j, doc in enumerate(result['retrieved_documents'][:3], 1):
                    similarity = doc.get('similarity', 0)
                    preview = doc.get('text', '')[:100] + "..." if len(doc.get('text', '')) > 100 else doc.get('text', '')
                    print(f"   {j}. Similarity: {similarity:.3f} - {preview}")
            
        except Exception as e:
            logger.error(f"Error processing query {i}: {str(e)}")
            continue
        
        print("\n" + "="*80)
    
    print("\n🎉 Query session completed!")

def interactive_query():
    """Interactive mode for custom queries."""
    
    print("\n🤖 Interactive Query Mode")
    print("Type 'quit' or 'exit' to stop")
    print("-" * 40)
    
    rag_system = ProductionRAGSystem()
    
    while True:
        try:
            query = input("\n💬 Enter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print("🔎 Searching...")
            result = rag_system.query_documents(query)
            
            print(f"\n✅ Answer: {result['answer']}")
            print(f"📊 Sources used: {result['num_sources_used']}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query existing FAISS index")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--query", "-q", type=str,
                       help="Single query to run")
    
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        rag_system = ProductionRAGSystem()
        result = rag_system.query_documents(args.query)
        print(f"Query: {args.query}")
        print(f"Answer: {result['answer']}")
        print(f"Sources used: {result['num_sources_used']}")
    elif args.interactive:
        # Interactive mode
        interactive_query()
    else:
        # Default: run predefined queries
        query_existing_index()
