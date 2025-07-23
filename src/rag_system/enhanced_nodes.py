"""
Enhanced nodes for production-ready RAG system with document parsing support.
"""

from pocketflow import Node, BatchNode
import numpy as np
import faiss
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from .utils import call_llm, get_embedding, fixed_size_chunk
from .document_parser import DocumentParser, DocumentMetadata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoaderNode(BatchNode):
    """
    Load and parse documents from files or directories.
    Supports PDF, DOCX, TXT, MD, HTML files.
    """
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        super().__init__()
        self.parser = DocumentParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def prep(self, shared):
        """Get document sources from shared store"""
        doc_sources = shared.get("document_sources", [])
        
        if not doc_sources:
            # Fall back to old behavior with text samples
            return shared.get("texts", [])
        
        return doc_sources
    
    def exec(self, source):
        """Process a single document source (file path or directory)"""
        try:
            # This check can fail for long strings that are not paths.
            is_file = Path(source).is_file()
        except (OSError, TypeError):
            is_file = False

        try:
            is_dir = Path(source).is_dir()
        except (OSError, TypeError):
            is_dir = False

        if is_file:
            try:
                text, metadata = self.parser.parse_file(source)
                chunks = self.parser.smart_chunk_text(text, metadata)
                return chunks
            except Exception as e:
                logger.error(f"Failed to parse file {source}: {str(e)}")
                return []
        
        elif is_dir:
            try:
                documents = self.parser.parse_directory(source, recursive=True)
                all_chunks = []
                for text, metadata in documents:
                    chunks = self.parser.smart_chunk_text(text, metadata)
                    all_chunks.extend(chunks)
                return all_chunks
            except Exception as e:
                logger.error(f"Failed to parse directory {source}: {str(e)}")
                return []
        
        else:
            # Assume it's raw text (for backward compatibility)
            if isinstance(source, str):
                chunks = fixed_size_chunk(source, self.parser.chunk_size)
                return [{'text': chunk, 'metadata': {'source': 'raw_text'}} for chunk in chunks]
            
            logger.warning(f"Could not process source of type {type(source)}")
            return []
    
    def post(self, shared, prep_res, exec_res_list):
        """Store processed chunks in the shared store"""
        all_chunks = []
        for chunks in exec_res_list:
            if isinstance(chunks, list):
                all_chunks.extend(chunks)
        
        # Extract just the text for backward compatibility
        texts = [chunk['text'] for chunk in all_chunks]
        
        shared["texts"] = texts
        shared["document_chunks"] = all_chunks  # Store full chunk info with metadata
        
        logger.info(f"✅ Loaded {len(all_chunks)} chunks from {len(prep_res)} sources")
        return "default"


class EnhancedEmbedDocumentsNode(BatchNode):
    """
    Enhanced embedding node with error handling and progress tracking.
    """
    
    def prep(self, shared):
        """Read texts from shared store"""
        return shared["texts"]
    
    def exec(self, text):
        """Embed a single text with error handling"""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return np.zeros(768, dtype=np.float32)  # Return zero vector for empty text
            
            return get_embedding(text)
        except Exception as e:
            logger.error(f"Failed to embed text: {str(e)}")
            return np.zeros(768, dtype=np.float32)  # Return zero vector on error
    
    def post(self, shared, prep_res, exec_res_list):
        """Store embeddings with validation"""
        embeddings = []
        
        for i, embedding in enumerate(exec_res_list):
            if embedding is not None and hasattr(embedding, 'shape') and embedding.size > 0:
                # Ensure embedding is 1D
                if len(embedding.shape) > 1:
                    embedding = embedding.flatten()
                embeddings.append(embedding)
            else:
                logger.warning(f"Invalid embedding at index {i}, using zero vector")
                embeddings.append(np.zeros(768, dtype=np.float32))
        
        if embeddings:
            # Stack embeddings ensuring they all have the same shape
            try:
                embeddings_array = np.vstack([emb.reshape(1, -1) for emb in embeddings])
                shared["embeddings"] = embeddings_array
                logger.info(f"✅ Created {len(embeddings)} document embeddings with shape {embeddings_array.shape}")
            except Exception as e:
                logger.error(f"Failed to create embeddings array: {str(e)}")
                # Debug: print shapes
                for i, emb in enumerate(embeddings[:5]):  # Show first 5
                    logger.error(f"Embedding {i} shape: {emb.shape if hasattr(emb, 'shape') else type(emb)}")
                shared["embeddings"] = np.array([], dtype=np.float32)
        else:
            logger.error("No valid embeddings created")
            shared["embeddings"] = np.array([], dtype=np.float32)
        
        return "default"


class PersistentIndexNode(Node):
    """
    Enhanced index node with persistence and loading capabilities.
    """
    
    def __init__(self, index_path: Optional[str] = None, save_index: bool = True):
        super().__init__()
        self.index_path = index_path or "rag_index.faiss"
        self.metadata_path = self.index_path.replace('.faiss', '_metadata.json')
        self.texts_path = self.index_path.replace('.faiss', '_texts.json')
        self.chunks_path = self.index_path.replace('.faiss', '_chunks.json')
        self.save_index = save_index
    
    def prep(self, shared):
        """Get embeddings and check if we should load existing index"""
        embeddings = shared.get("embeddings")
        
        # Try to load existing index if no new embeddings
        if (embeddings is None or len(embeddings) == 0) and os.path.exists(self.index_path):
            logger.info(f"Loading existing index from {self.index_path}")
            try:
                index = faiss.read_index(self.index_path)
                shared["index"] = index
                
                # Load metadata if available
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        metadata = json.load(f)
                        shared["index_metadata"] = metadata
                        logger.info(f"Loaded index with {index.ntotal} vectors")
                
                # Load texts if available
                if os.path.exists(self.texts_path):
                    with open(self.texts_path, 'r') as f:
                        texts = json.load(f)
                        shared["texts"] = texts
                        logger.info(f"Loaded {len(texts)} text chunks")
                
                # Load document chunks if available
                if os.path.exists(self.chunks_path):
                    with open(self.chunks_path, 'r') as f:
                        chunks = json.load(f)
                        shared["document_chunks"] = chunks
                        logger.info(f"Loaded {len(chunks)} document chunks with metadata")
                
                return None  # Skip execution
            except Exception as e:
                logger.error(f"Failed to load existing index: {str(e)}")
        
        return embeddings
    
    def exec(self, embeddings):
        """Create FAISS index and add embeddings"""
        if embeddings is None:
            return None  # Index already loaded
        
        logger.info("🔍 Creating search index...")
        dimension = embeddings.shape[1]
        
        # Create a more sophisticated index for better performance
        if len(embeddings) > 1000:
            # Use IVF index for larger datasets
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings)
        else:
            # Use flat index for smaller datasets
            index = faiss.IndexFlatL2(dimension)
        
        # Add the embeddings to the index
        index.add(embeddings)
        
        return index
    
    def post(self, shared, prep_res, exec_res):
        """Store and optionally persist the index"""
        if exec_res is not None:
            shared["index"] = exec_res
            
            # Create metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "vector_count": exec_res.ntotal,
                "dimension": exec_res.d if hasattr(exec_res, 'd') else 0,
                "index_type": type(exec_res).__name__
            }
            shared["index_metadata"] = metadata
            
            # Save index to disk if requested
            if self.save_index:
                try:
                    faiss.write_index(exec_res, self.index_path)
                    with open(self.metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Save texts for later retrieval
                    if "texts" in shared and shared["texts"]:
                        with open(self.texts_path, 'w') as f:
                            json.dump(shared["texts"], f, indent=2)
                        logger.info(f"💾 Saved {len(shared['texts'])} text chunks")
                    
                    # Save document chunks with metadata
                    if "document_chunks" in shared and shared["document_chunks"]:
                        # Convert to JSON-serializable format
                        chunks_data = []
                        for chunk in shared["document_chunks"]:
                            if isinstance(chunk, dict):
                                chunks_data.append(chunk)
                            else:
                                # Handle other formats if needed
                                chunks_data.append({"text": str(chunk), "metadata": {}})
                        
                        with open(self.chunks_path, 'w') as f:
                            json.dump(chunks_data, f, indent=2)
                        logger.info(f"💾 Saved {len(chunks_data)} document chunks with metadata")
                    
                    logger.info(f"💾 Index and associated data saved to {self.index_path}")
                except Exception as e:
                    logger.error(f"Failed to save index: {str(e)}")
            
            logger.info(f"✅ Index created with {exec_res.ntotal} vectors")
        else:
            logger.info("✅ Using existing index")
        
        return "default"


class EnhancedRetrieveDocumentNode(Node):
    """
    Enhanced retrieval node with multiple document support and ranking.
    """
    
    def __init__(self, k: int = 5, min_similarity_threshold: float = 0.0):
        super().__init__()
        self.k = k  # Number of documents to retrieve
        self.min_similarity_threshold = min_similarity_threshold
    
    def prep(self, shared):
        """Get query embedding, index, and texts from shared store"""
        return (
            shared["query_embedding"], 
            shared["index"], 
            shared["texts"],
            shared.get("document_chunks", [])
        )
    
    def exec(self, inputs):
        """Search the index for similar documents"""
        query_embedding, index, texts, document_chunks = inputs
        
        logger.info(f"🔎 Searching for top {self.k} relevant documents...")
        
        # Search for similar documents
        distances, indices = index.search(query_embedding, k=min(self.k, len(texts)))
        
        retrieved_docs = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(texts):  # Valid index
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                
                if similarity >= self.min_similarity_threshold:
                    doc_info = {
                        "text": texts[idx],
                        "index": int(idx),
                        "distance": float(distance),
                        "similarity": float(similarity),
                        "rank": i + 1
                    }
                    
                    # Add metadata if available
                    if document_chunks and idx < len(document_chunks):
                        doc_info["metadata"] = document_chunks[idx].get("metadata", {})
                    
                    retrieved_docs.append(doc_info)
        
        return retrieved_docs
    
    def post(self, shared, prep_res, exec_res):
        """Store retrieved documents in shared store"""
        shared["retrieved_documents"] = exec_res
        
        if exec_res:
            logger.info(f"📄 Retrieved {len(exec_res)} relevant documents")
            for i, doc in enumerate(exec_res[:3]):  # Show top 3
                source = doc.get("metadata", {}).get("source_path", "unknown")
                logger.info(f"  {i+1}. {Path(source).name if source != 'unknown' else 'Raw text'} "
                          f"(similarity: {doc['similarity']:.3f})")
        else:
            logger.warning("No relevant documents found")
        
        return "default"


class ContextualAnswerNode(Node):
    """
    Enhanced answer generation with multiple document context and citations.
    """
    
    def prep(self, shared):
        """Get query and retrieved documents"""
        return shared["query"], shared.get("retrieved_documents", [])
    
    def exec(self, inputs):
        """Generate an answer using multiple retrieved documents"""
        query, retrieved_docs = inputs
        
        if not retrieved_docs:
            return "I couldn't find any relevant information to answer your question."
        
        # Prepare context from multiple documents
        context_parts = []
        sources = []
        
        for i, doc in enumerate(retrieved_docs[:3]):  # Use top 3 documents
            context_parts.append(f"[Source {i+1}] {doc['text']}")
            
            # Collect source information
            metadata = doc.get("metadata", {})
            source_file = metadata.get("source_path", "unknown")
            if source_file != "unknown":
                source_name = Path(source_file).name
                chunk_num = metadata.get("chunk_number", 0)
                sources.append(f"Source {i+1}: {source_name} (chunk {chunk_num})")
            else:
                sources.append(f"Source {i+1}: Raw text")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""
Please answer the following question based on the provided context. Be specific and cite information from the sources when relevant.

Question: {query}

Context:
{context}

Instructions:
1. Provide a comprehensive answer based on the context
2. If the context doesn't fully answer the question, mention what information is missing
3. Be accurate and don't make up information not present in the context
4. If you reference specific information, mention which source it came from

Answer:
"""
        
        answer = call_llm(prompt)
        
        # Append source information
        if sources:
            answer += f"\n\nSources:\n" + "\n".join(sources)
        
        return answer
    
    def post(self, shared, prep_res, exec_res):
        """Store generated answer in shared store"""
        shared["generated_answer"] = exec_res
        logger.info("\n🤖 Generated Answer:")
        logger.info(exec_res)
        return "default"


# Legacy aliases for backward compatibility
ChunkDocumentsNode = DocumentLoaderNode
EmbedDocumentsNode = EnhancedEmbedDocumentsNode
CreateIndexNode = PersistentIndexNode
RetrieveDocumentNode = EnhancedRetrieveDocumentNode
GenerateAnswerNode = ContextualAnswerNode
