"""
Document parsing utilities for different file formats.
Supports PDF, DOCX, TXT, MD, HTML files and directories.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import markdown
except ImportError:
    markdown = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for a parsed document"""
    source_path: str
    file_type: str
    file_size: int
    page_count: Optional[int] = None
    title: Optional[str] = None
    author: Optional[str] = None


class DocumentParser:
    """
    Production-ready document parser supporting multiple file formats.
    """
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.docx': 'docx', 
        '.doc': 'docx',
        '.txt': 'text',
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.html': 'html',
        '.htm': 'html',
    }
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        """
        Initialize the document parser.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """Check if required dependencies are installed"""
        missing_deps = []
        
        if PyPDF2 is None:
            missing_deps.append("PyPDF2 (for PDF support)")
        if DocxDocument is None:
            missing_deps.append("python-docx (for DOCX support)")
        if BeautifulSoup is None:
            missing_deps.append("beautifulsoup4 (for HTML support)")
        if markdown is None:
            missing_deps.append("markdown (for Markdown support)")
            
        if missing_deps:
            logger.warning(f"Missing optional dependencies: {', '.join(missing_deps)}")
    
    def parse_file(self, file_path: str) -> Tuple[str, DocumentMetadata]:
        """
        Parse a single file and extract text content.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            Tuple of (extracted_text, metadata)
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        file_type = self.SUPPORTED_EXTENSIONS[file_ext]
        file_size = file_path.stat().st_size
        
        logger.info(f"Parsing {file_type} file: {file_path.name}")
        
        try:
            if file_type == 'pdf':
                text, page_count = self._parse_pdf(file_path)
            elif file_type == 'docx':
                text = self._parse_docx(file_path)
                page_count = None
            elif file_type == 'text':
                text = self._parse_text(file_path)
                page_count = None
            elif file_type == 'markdown':
                text = self._parse_markdown(file_path)
                page_count = None
            elif file_type == 'html':
                text = self._parse_html(file_path)
                page_count = None
            else:
                raise ValueError(f"Parser not implemented for type: {file_type}")
            
            metadata = DocumentMetadata(
                source_path=str(file_path),
                file_type=file_type,
                file_size=file_size,
                page_count=page_count,
                title=file_path.stem
            )
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            raise
    
    def _parse_pdf(self, file_path: Path) -> Tuple[str, int]:
        """Parse PDF file and extract text"""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF parsing. Install with: pip install PyPDF2")
        
        text_content = []
        page_count = 0
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {str(e)}")
            raise
        
        return '\n\n'.join(text_content), page_count
    
    def _parse_docx(self, file_path: Path) -> str:
        """Parse DOCX file and extract text"""
        if DocxDocument is None:
            raise ImportError("python-docx is required for DOCX parsing. Install with: pip install python-docx")
        
        try:
            doc = DocxDocument(file_path)
            paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
            return '\n\n'.join(paragraphs)
        except Exception as e:
            logger.error(f"Error reading DOCX file {file_path}: {str(e)}")
            raise
    
    def _parse_text(self, file_path: Path) -> str:
        """Parse plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode text file {file_path}")
    
    def _parse_markdown(self, file_path: Path) -> str:
        """Parse Markdown file and convert to plain text"""
        text_content = self._parse_text(file_path)
        
        if markdown is None:
            logger.warning("markdown library not available, returning raw markdown")
            return text_content
        
        try:
            # Convert markdown to HTML, then extract text
            html = markdown.markdown(text_content)
            if BeautifulSoup is not None:
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
            else:
                return text_content  # Return raw markdown if BeautifulSoup not available
        except Exception as e:
            logger.warning(f"Error converting markdown: {str(e)}, returning raw content")
            return text_content
    
    def _parse_html(self, file_path: Path) -> str:
        """Parse HTML file and extract text content"""
        html_content = self._parse_text(file_path)
        
        if BeautifulSoup is None:
            raise ImportError("beautifulsoup4 is required for HTML parsing. Install with: pip install beautifulsoup4")
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Error parsing HTML file {file_path}: {str(e)}")
            raise
    
    def parse_directory(self, directory_path: str, recursive: bool = True) -> List[Tuple[str, DocumentMetadata]]:
        """
        Parse all supported files in a directory.
        
        Args:
            directory_path: Path to the directory to scan
            recursive: Whether to scan subdirectories recursively
            
        Returns:
            List of (text_content, metadata) tuples
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        documents = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    text, metadata = self.parse_file(file_path)
                    documents.append((text, metadata))
                except Exception as e:
                    logger.error(f"Failed to parse {file_path}: {str(e)}")
                    continue
        
        logger.info(f"Successfully parsed {len(documents)} documents from {directory_path}")
        return documents
    
    def smart_chunk_text(self, text: str, metadata: DocumentMetadata) -> List[Dict[str, any]]:
        """
        Intelligently chunk text while preserving context.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        chunks = []
        
        # Split by paragraphs first for better context preservation
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_number = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, save current chunk
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(self._create_chunk_dict(current_chunk, metadata, chunk_number))
                chunk_number += 1
                
                # Start new chunk with overlap from previous chunk
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(self._create_chunk_dict(current_chunk, metadata, chunk_number))
        
        return chunks
    
    def _create_chunk_dict(self, text: str, metadata: DocumentMetadata, chunk_number: int) -> Dict[str, any]:
        """Create a chunk dictionary with text and metadata"""
        return {
            'text': text.strip(),
            'metadata': {
                'source_file': metadata.source_path,
                'file_type': metadata.file_type,
                'chunk_number': chunk_number,
                'title': metadata.title,
                'file_size': metadata.file_size,
                'page_count': metadata.page_count
            }
        }


def get_supported_file_types() -> List[str]:
    """Get list of supported file extensions"""
    return list(DocumentParser.SUPPORTED_EXTENSIONS.keys())


if __name__ == "__main__":
    # Example usage
    parser = DocumentParser(chunk_size=1000, chunk_overlap=100)
    
    # Test parsing a single file (if it exists)
    test_files = ["test.pdf", "test.docx", "test.txt"]
    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                text, metadata = parser.parse_file(test_file)
                chunks = parser.smart_chunk_text(text, metadata)
                print(f"Parsed {test_file}: {len(chunks)} chunks")
                print(f"First chunk: {chunks[0]['text'][:200]}...")
                break
            except Exception as e:
                print(f"Error parsing {test_file}: {e}")
    else:
        print("No test files found. Create test.pdf, test.docx, or test.txt to test parsing.")
    
    print(f"Supported file types: {get_supported_file_types()}")
