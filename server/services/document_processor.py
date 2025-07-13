import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod
import json
import re
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, source: Union[str, Path, bytes]) -> List[Dict[str, Any]]:
        """Load and process a document.
        
        Args:
            source: Path to the document, URL, or raw bytes
            
        Returns:
            List of document chunks with metadata
        """
        pass

class PDFLoader(DocumentLoader):
    """Loader for PDF documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load(self, source: Union[str, Path, bytes]) -> List[Dict[str, Any]]:
        """Load and chunk a PDF document."""
        try:
            if isinstance(source, (str, Path)):
                with open(source, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            else:  # bytes
                pdf_reader = PyPDF2.PdfReader(BytesIO(source))
                text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            
            return self._chunk_text(text, {
                'source': str(source) if not isinstance(source, bytes) else 'uploaded_pdf',
                'type': 'pdf'
            })
            
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise

    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk_meta = metadata.copy()
            chunk_meta['chunk'] = len(chunks)
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_meta
            })
            
            if i + self.chunk_size >= len(words):
                break
                
        return chunks

class WebPageLoader(DocumentLoader):
    """Loader for web pages."""
    
    def load(self, url: str) -> List[Dict[str, Any]]:
        """Load and process a web page."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text and clean it up
            text = soup.get_text(separator='\n', strip=True)
            text = '\n'.join([line for line in text.split('\n') if line.strip()])
            
            return [{
                'text': text,
                'metadata': {
                    'source': url,
                    'type': 'webpage',
                    'title': soup.title.string if soup.title else 'Untitled',
                    'url': url
                }
            }]
            
        except Exception as e:
            logger.error(f"Error loading webpage {url}: {str(e)}")
            raise

class TextLoader(DocumentLoader):
    """Loader for plain text files and notes."""
    
    def load(self, source: Union[str, Path, bytes]) -> List[Dict[str, Any]]:
        """Load a text file or process text content."""
        try:
            if isinstance(source, (str, Path)):
                with open(source, 'r', encoding='utf-8') as f:
                    text = f.read()
                source_str = str(source)
            else:  # bytes
                text = source.decode('utf-8')
                source_str = 'uploaded_text'
                
            return [{
                'text': text,
                'metadata': {
                    'source': source_str,
                    'type': 'text',
                    'title': Path(source_str).stem if isinstance(source, (str, Path)) else 'Note'
                }
            }]
            
        except Exception as e:
            logger.error(f"Error loading text: {str(e)}")
            raise

class DocumentProcessor:
    """Main document processing class that handles multiple document types."""
    
    def __init__(self, vector_store):
        # Create a separate vector store for documents
        from ..services.vector_store import VectorMemoryStore
        self.vector_store = VectorMemoryStore(
            persist_directory=str(Path('data/document_store')),
            model_name='all-MiniLM-L6-v2'  # Same model as main store for compatibility
        )
        self.loaders = {
            '.pdf': PDFLoader(),
            '.txt': TextLoader(),
            '.md': TextLoader(),
            '.json': TextLoader(),
            'web': WebPageLoader()
        }
    
    async def process_file(self, file_path: Union[str, Path]) -> bool:
        """Process a file and add it to the vector store.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            bool: True if processing was successful
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
                
            # Get file extension
            ext = file_path.suffix.lower()
            
            # Determine the loader to use
            loader = self._get_loader(file_path, source_type=ext[1:] if ext else None)
            if not loader:
                logger.error(f"No suitable loader found for file: {file_path}")
                return False
            
            # Load and process the document
            documents = loader.load(file_path)
            
            # Add to vector store
            for doc in documents:
                # Add document metadata to the text to improve search relevance
                doc_text = f"Document: {doc['metadata'].get('title', file_path.name)}\n\n{doc['text']}"
                self.vector_store.add_memory(
                    conversation_id='documents',
                    text=doc_text,
                    metadata={
                        **doc['metadata'],
                        'source': str(file_path),
                        'type': 'document'
                    }
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return False
    
    def _get_loader(self, source: Union[str, Path, bytes], source_type: str = None):
        """Get the appropriate loader for the given source."""
        if source_type:
            if source_type in ['pdf']:
                return self.loaders['.pdf']
            elif source_type in ['web', 'url']:
                return self.loaders['web']
            elif source_type in ['text', 'txt', 'md', 'json']:
                return self.loaders['.txt']
        
        if isinstance(source, bytes):
            # For bytes, we need source_type to be specified
            return None
            
        source_str = str(source).lower()
        
        # Check file extension
        if source_str.startswith(('http://', 'https://')):
            return self.loaders['web']
            
        # Get file extension
        ext = os.path.splitext(source_str)[1].lower()
        return self.loaders.get(ext)

    def process_directory(self, directory: Union[str, Path], recursive: bool = True) -> dict:
        """Process all supported documents in a directory.
        
        Args:
            directory: Path to the directory
            recursive: Whether to process subdirectories
            
        Returns:
            dict: Processing results with counts and errors
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
            
        results = {
            'processed': 0,
            'failed': 0,
            'errors': []
        }
        
        # Supported extensions
        extensions = {'.pdf', '.txt', '.md', '.json'}
        
        # Walk through directory
        for item in directory.rglob('*') if recursive else directory.glob('*'):
            if item.is_file() and item.suffix.lower() in extensions:
                try:
                    if self.process_document(item):
                        results['processed'] += 1
                        logger.info(f"Processed: {item}")
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Failed to process: {item}")
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Error processing {item}: {str(e)}")
                    logger.exception(f"Error processing {item}")
        
        return results
