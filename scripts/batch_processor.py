#!/usr/bin/env python3
"""
Production utilities for batch processing and monitoring.
"""

import sys
from pathlib import Path
# Add project root to path to allow imports from other directories
sys.path.append(str(Path(__file__).resolve().parents[2]))

from main import ProductionRAGSystem
from src.rag_system.document_parser import get_supported_file_types

import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_batch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result from batch processing operation."""
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_chunks: int
    processing_time: float
    index_size_mb: float
    errors: List[str]


class BatchProcessor:
    """Batch processor for large document collections."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize batch processor."""
        self.config = config or self._get_default_config()
        self.rag_system = ProductionRAGSystem(config)
        self.results = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for batch processing."""
        return {
            "offline": {
                "chunk_size": 1500,
                "chunk_overlap": 300,
                "index_path": "batch_index.faiss",
                "save_index": True,
                "batch_size": 100
            },
            "online": {
                "k": 10,
                "min_similarity_threshold": 0.1
            },
            "batch": {
                "max_file_size_mb": 50,
                "skip_corrupted_files": True,
                "parallel_processing": False,
                "checkpoint_frequency": 100
            }
        }
    
    def process_directory_tree(self, root_dir: str, patterns: List[str] = None) -> BatchResult:
        """
        Process an entire directory tree of documents.
        
        Args:
            root_dir: Root directory to process
            patterns: File patterns to include (e.g., ['*.pdf', '*.docx'])
            
        Returns:
            BatchResult with processing statistics
        """
        start_time = time.time()
        root_path = Path(root_dir)
        
        if not root_path.exists():
            raise ValueError(f"Directory not found: {root_dir}")
        
        # Discover files
        logger.info(f"Discovering files in {root_dir}...")
        file_paths = self._discover_files(root_path, patterns)
        logger.info(f"Found {len(file_paths)} files to process")
        
        # Process in batches
        successful = 0
        failed = 0
        errors = []
        total_chunks = 0
        
        batch_size = self.config.get("batch", {}).get("batch_size", 100)
        
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(file_paths)-1)//batch_size + 1}")
            
            try:
                result = self.rag_system.index_documents([str(p) for p in batch])
                successful += result.get("indexed_sources", 0)
                total_chunks += result.get("total_chunks", 0)
                
                # Save checkpoint
                if (i + batch_size) % (batch_size * 5) == 0:
                    self._save_checkpoint(i + batch_size, len(file_paths))
                    
            except Exception as e:
                error_msg = f"Batch {i//batch_size + 1} failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                failed += len(batch)
        
        # Calculate final statistics
        processing_time = time.time() - start_time
        index_size = self._get_index_size()
        
        result = BatchResult(
            total_documents=len(file_paths),
            successful_documents=successful,
            failed_documents=failed,
            total_chunks=total_chunks,
            processing_time=processing_time,
            index_size_mb=index_size,
            errors=errors
        )
        
        self._save_batch_report(result)
        return result
    
    def _discover_files(self, root_path: Path, patterns: List[str] = None) -> List[Path]:
        """Discover files to process."""
        files = []
        supported_extensions = get_supported_file_types()
        max_size_mb = self.config.get("batch", {}).get("max_file_size_mb", 50)
        
        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue
                
            # Check extension
            if file_path.suffix.lower() not in supported_extensions:
                continue
                
            # Check file size
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > max_size_mb:
                    logger.warning(f"Skipping large file: {file_path} ({size_mb:.1f}MB)")
                    continue
            except OSError:
                logger.warning(f"Cannot access file: {file_path}")
                continue
                
            # Check patterns
            if patterns:
                if not any(file_path.match(pattern) for pattern in patterns):
                    continue
                    
            files.append(file_path)
            
        return files
    
    def _save_checkpoint(self, processed: int, total: int):
        """Save processing checkpoint."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "processed": processed,
            "total": total,
            "progress": processed / total * 100
        }
        
        with open("batch_checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2)
            
        logger.info(f"Checkpoint saved: {processed}/{total} ({checkpoint['progress']:.1f}%)")
    
    def _get_index_size(self) -> float:
        """Get index file size in MB."""
        index_path = self.config["offline"]["index_path"]
        try:
            return Path(index_path).stat().st_size / (1024 * 1024)
        except FileNotFoundError:
            return 0.0
    
    def _save_batch_report(self, result: BatchResult):
        """Save batch processing report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "results": {
                "total_documents": result.total_documents,
                "successful_documents": result.successful_documents,
                "failed_documents": result.failed_documents,
                "success_rate": result.successful_documents / result.total_documents * 100,
                "total_chunks": result.total_chunks,
                "processing_time_minutes": result.processing_time / 60,
                "index_size_mb": result.index_size_mb,
                "documents_per_minute": result.total_documents / (result.processing_time / 60),
                "errors": result.errors
            }
        }
        
        with open("batch_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info("Batch report saved to batch_report.json")


def main():
    """CLI entry point for batch processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process documents for RAG")
    parser.add_argument("--directory", "-d", required=True, help="Directory to process")
    parser.add_argument("--config", "-c", help="Configuration file")
    parser.add_argument("--patterns", "-p", nargs="+", help="File patterns to include")
    parser.add_argument("--max-size", type=int, default=50, help="Max file size in MB")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    
    # Override max file size
    if config:
        config.setdefault("batch", {})["max_file_size_mb"] = args.max_size
    
    # Process documents
    processor = BatchProcessor(config)
    
    try:
        result = processor.process_directory_tree(args.directory, args.patterns)
        
        print("\n" + "="*60)
        print("BATCH PROCESSING RESULTS")
        print("="*60)
        print(f"Total documents: {result.total_documents}")
        print(f"Successful: {result.successful_documents}")
        print(f"Failed: {result.failed_documents}")
        print(f"Success rate: {result.successful_documents/result.total_documents*100:.1f}%")
        print(f"Total chunks: {result.total_chunks}")
        print(f"Processing time: {result.processing_time/60:.1f} minutes")
        print(f"Index size: {result.index_size_mb:.1f} MB")
        print(f"Rate: {result.total_documents/(result.processing_time/60):.1f} docs/minute")
        
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for error in result.errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(result.errors) > 5:
                print(f"  ... and {len(result.errors) - 5} more")
                
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
