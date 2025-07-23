from pocketflow import Flow, Node
import numpy as np
from .enhanced_nodes import (
    DocumentLoaderNode, 
    EnhancedEmbedDocumentsNode, 
    PersistentIndexNode,
    EnhancedRetrieveDocumentNode, 
    ContextualAnswerNode
)
from .utils import get_embedding
import logging

logger = logging.getLogger(__name__)


class EmbedQueryNode(Node):
    """Node to embed the user query"""
    
    def prep(self, shared):
        """Get query from shared store"""
        return shared["query"]
    
    def exec(self, query):
        """Embed the query"""
        logger.info(f"🔍 Embedding query: {query}")
        query_embedding = get_embedding(query)
        return np.array([query_embedding], dtype=np.float32)
    
    def post(self, shared, prep_res, exec_res):
        """Store query embedding in shared store"""
        shared["query_embedding"] = exec_res
        return "default"


def get_offline_flow(chunk_size: int = 2000, chunk_overlap: int = 200, 
                    index_path: str = "rag_index.faiss", save_index: bool = True):
    """
    Create offline flow for document indexing.
    
    Args:
        chunk_size: Maximum size of each text chunk
        chunk_overlap: Number of characters to overlap between chunks
        index_path: Path to save the FAISS index
        save_index: Whether to save the index to disk
    """
    # Create offline flow for document indexing
    doc_loader_node = DocumentLoaderNode(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embed_docs_node = EnhancedEmbedDocumentsNode()
    create_index_node = PersistentIndexNode(index_path=index_path, save_index=save_index)
    
    # Connect the nodes
    doc_loader_node >> embed_docs_node >> create_index_node
    
    offline_flow = Flow(start=doc_loader_node)
    return offline_flow


def get_online_flow(k: int = 5, min_similarity_threshold: float = 0.0):
    """
    Create online flow for document retrieval and answer generation.
    
    Args:
        k: Number of documents to retrieve
        min_similarity_threshold: Minimum similarity score for retrieved documents
    """
    # Create online flow for document retrieval and answer generation
    embed_query_node = EmbedQueryNode()
    retrieve_doc_node = EnhancedRetrieveDocumentNode(k=k, min_similarity_threshold=min_similarity_threshold)
    generate_answer_node = ContextualAnswerNode()
    
    # Connect the nodes
    embed_query_node >> retrieve_doc_node >> generate_answer_node
    
    online_flow = Flow(start=embed_query_node)
    return online_flow


def get_production_flows(config: dict = None):
    """
    Get production-ready flows with configuration.
    
    Args:
        config: Configuration dictionary with flow parameters
    """
    if config is None:
        config = {}
    
    offline_config = config.get("offline", {})
    online_config = config.get("online", {})
    
    offline_flow = get_offline_flow(
        chunk_size=offline_config.get("chunk_size", 2000),
        chunk_overlap=offline_config.get("chunk_overlap", 200),
        index_path=offline_config.get("index_path", "rag_index.faiss"),
        save_index=offline_config.get("save_index", True)
    )
    
    online_flow = get_online_flow(
        k=online_config.get("k", 5),
        min_similarity_threshold=online_config.get("min_similarity_threshold", 0.0)
    )
    
    return offline_flow, online_flow


# Initialize default flows
# These are not used directly by the production system, but can be useful for testing
offline_flow = get_offline_flow()
online_flow = get_online_flow()