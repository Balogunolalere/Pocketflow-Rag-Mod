"""
Production-ready RAG system with document parsing support.

Usage:
    python main.py --query "Your question here" --docs /path/to/documents
    python main.py --query "Your question here" --docs file1.pdf file2.docx
    python main.py --config config.json
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.rag_system.flow import get_production_flows
from src.rag_system.document_parser import get_supported_file_types

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionRAGSystem:
    """Production-ready RAG system with file and directory support."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the RAG system with configuration."""
        self.config = config or self._get_default_config()
        self.offline_flow, self.online_flow = get_production_flows(self.config)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "offline": {
                "chunk_size": 2000,
                "chunk_overlap": 200,
                "index_path": "rag_index.faiss",
                "save_index": True
            },
            "online": {
                "k": 5,
                "min_similarity_threshold": 0.0
            }
        }
    
    def index_documents(self, document_sources: List[str]) -> Dict[str, Any]:
        """
        Index documents from files or directories.
        
        Args:
            document_sources: List of file paths or directory paths
            
        Returns:
            Dictionary with indexing results
        """
        logger.info(f"Starting document indexing for {len(document_sources)} sources")
        
        # Validate document sources
        valid_sources = []
        for source in document_sources:
            source_path = Path(source)
            if source_path.exists():
                valid_sources.append(str(source_path))
            else:
                logger.warning(f"Source not found: {source}")
        
        if not valid_sources:
            raise ValueError("No valid document sources provided")
        
        # Prepare shared state
        shared = {
            "document_sources": valid_sources,
            "texts": [],
            "document_chunks": [],
            "embeddings": None,
            "index": None
        }
        
        # Run offline flow
        try:
            self.offline_flow.run(shared)
            
            result = {
                "status": "success",
                "indexed_sources": len(valid_sources),
                "total_chunks": len(shared.get("texts", [])),
                "index_path": self.config["offline"]["index_path"]
            }
            
            logger.info(f"✅ Indexing completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Indexing failed: {str(e)}")
            raise
    
    def query_documents(self, query: str, shared_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Query the indexed documents.
        
        Args:
            query: User query
            shared_state: Existing shared state (optional)
            
        Returns:
            Dictionary with query results
        """
        logger.info(f"Processing query: {query}")
        
        # Use provided shared state or create minimal one
        if shared_state is None:
            # Try to load existing index
            shared_state = {
                "texts": [],
                "document_chunks": [],
                "embeddings": None,
                "index": None
            }
            
            # This will trigger loading of existing index in PersistentIndexNode
            try:
                self.offline_flow.run(shared_state)
            except Exception as e:
                logger.error(f"Failed to load existing index: {str(e)}")
                raise ValueError("No indexed documents found. Please run indexing first.")
        
        # Add query to shared state
        shared_state.update({
            "query": query,
            "query_embedding": None,
            "retrieved_documents": None,
            "generated_answer": None
        })
        
        # Run online flow
        try:
            self.online_flow.run(shared_state)
            
            result = {
                "status": "success",
                "query": query,
                "answer": shared_state.get("generated_answer", "No answer generated"),
                "retrieved_documents": shared_state.get("retrieved_documents", []),
                "num_sources_used": len(shared_state.get("retrieved_documents", []))
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise
    
    def run_full_pipeline(self, document_sources: List[str], query: str) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline: index documents and process query.
        
        Args:
            document_sources: List of file paths or directory paths
            query: User query
            
        Returns:
            Dictionary with complete results
        """
        # Index documents
        indexing_result = self.index_documents(document_sources)
        
        # Create shared state from indexing
        shared = {
            "document_sources": document_sources,
            "texts": [],
            "document_chunks": [],
            "embeddings": None,
            "index": None
        }
        
        # Re-run offline flow to get shared state
        self.offline_flow.run(shared)
        
        # Process query
        query_result = self.query_documents(query, shared)
        
        # Combine results
        return {
            "indexing": indexing_result,
            "query_result": query_result
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        raise


def create_sample_config(output_path: str = "rag_config.json"):
    """Create a sample configuration file."""
    config = {
        "offline": {
            "chunk_size": 2000,
            "chunk_overlap": 200,
            "index_path": "rag_index.faiss",
            "save_index": True
        },
        "online": {
            "k": 5,
            "min_similarity_threshold": 0.0
        },
        "default_documents": [
            "./documents",
            "./data"
        ],
        "default_query": "What is the main topic of these documents?"
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Sample configuration created: {output_path}")


def main():
    """Main entry point for the production RAG system."""
    parser = argparse.ArgumentParser(
        description="Production RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Index documents and query
  python {sys.argv[0]} --query "How to install software?" --docs ./documents
  
  # Use specific files
  python {sys.argv[0]} --query "What is mentioned about AI?" --docs file1.pdf file2.docx
  
  # Use configuration file
  python {sys.argv[0]} --config config.json
  
  # Create sample config
  python {sys.argv[0]} --create-config
  
  # Query existing index
  python {sys.argv[0]} --query "Your question" --use-existing-index

Supported file types: {', '.join(get_supported_file_types())}
        """
    )
    
    parser.add_argument("--query", "-q", type=str, help="Query to ask")
    parser.add_argument("--docs", "-d", nargs="+", help="Document files or directories to index")
    parser.add_argument("--config", "-c", type=str, help="Configuration file path")
    parser.add_argument("--create-config", action="store_true", help="Create sample configuration file")
    parser.add_argument("--use-existing-index", action="store_true", help="Use existing index without re-indexing")
    parser.add_argument("--output", "-o", type=str, help="Output file for results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config()
        return
    
    # Load configuration
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Initialize RAG system
    rag_system = ProductionRAGSystem(config)
    
    try:
        # Determine operation mode
        if args.use_existing_index:
            # Query existing index
            if not args.query:
                logger.error("Query is required when using existing index")
                return
            
            result = rag_system.query_documents(args.query)
            
        else:
            # Full pipeline or configuration-based operation
            query = args.query
            docs = args.docs
            
            # Use config defaults if not provided
            if config:
                if not query:
                    query = config.get("default_query", "What is the main topic?")
                if not docs:
                    docs = config.get("default_documents", [])
            
            if not query:
                logger.error("Query is required")
                return
            
            if not docs:
                logger.error("Document sources are required")
                return
            
            # Run full pipeline
            result = rag_system.run_full_pipeline(docs, query)
        
        # Output results
        print("\n" + "="*60)
        print("RAG SYSTEM RESULTS")
        print("="*60)
        
        if "query_result" in result:
            query_result = result["query_result"]
            print(f"\nQuery: {query_result['query']}")
            print(f"\nAnswer:\n{query_result['answer']}")
            print(f"\nSources used: {query_result['num_sources_used']}")
        else:
            print(f"\nQuery: {result['query']}")
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nSources used: {result['num_sources_used']}")
        
        # Save results to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")
    
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
