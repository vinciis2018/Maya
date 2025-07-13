import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from fastapi import UploadFile, HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentController:
    """Controller for handling document-related operations with FastAPI."""
    
    def __init__(self, document_processor, upload_folder: str = 'uploads'):
        """Initialize the document controller.
        
        Args:
            document_processor: Instance of DocumentProcessor
            upload_folder: Directory to store uploaded files
        """
        self.document_processor = document_processor
        self.upload_folder = Path(upload_folder)
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        
        # Allowed file extensions
        self.allowed_extensions = {'pdf', 'txt', 'md', 'json'}
    
    async def upload_file(self, file_path: str) -> Dict[str, Any]:
        """Process an uploaded file.
        
        Args:
            file_path: Path to the uploaded file
            
        Returns:
            Dict with success status and result/error message
        """
        try:
            if not file_path or not os.path.exists(file_path):
                return {'success': False, 'error': 'File not found'}, 400
                
            filename = os.path.basename(file_path)
            if not self._allowed_file(filename):
                return {
                    'success': False,
                    'error': f'File type not allowed. Allowed types: {self.allowed_extensions}'
                }, 400
            
            # Process the file
            result = await self.document_processor.process_file(file_path)
            return {'success': True, 'result': result}
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {'success': False, 'error': str(e)}, 500
    
    def process_url(self, data: Dict[str, str]) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
        """Process a document from a URL.
        
        Args:
            data: Dictionary containing 'url' key with the URL to process
            
        Returns:
            Processing result or error message
        """
        try:
            url = data.get('url', '').strip()
            if not url:
                return {'success': False, 'error': 'No URL provided'}, 400
                
            # Process the URL
            result = self.document_processor.process_url(url)
            return {'success': True, 'result': result}
            
        except Exception as e:
            logger.error(f"Error processing URL {data.get('url')}: {str(e)}")
            return {'success': False, 'error': str(e)}, 500
    
    def process_directory(self, data: Dict[str, str]) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
        """Process all supported documents in a directory.
        
        Args:
            data: Dictionary containing 'directory_path' with the path to process
            
        Returns:
            Processing results or error message
        """
        try:
            dir_path = data.get('directory_path', '').strip()
            if not dir_path:
                return {'success': False, 'error': 'No directory path provided'}, 400
                
            dir_path = Path(dir_path)
            if not dir_path.exists() or not dir_path.is_dir():
                return {'success': False, 'error': 'Invalid directory path'}, 400
                
            # Process all supported files in the directory
            results = []
            for ext in self.allowed_extensions:
                for file_path in dir_path.glob(f'*.{ext}'):
                    try:
                        result = self.document_processor.process_file(str(file_path))
                        results.append({
                            'file': str(file_path),
                            'success': True,
                            'result': result
                        })
                    except Exception as e:
                        results.append({
                            'file': str(file_path),
                            'success': False,
                            'error': str(e)
                        })
                        
            return {'success': True, 'results': results}
            
        except Exception as e:
            logger.error(f"Error processing directory {data.get('directory_path')}: {str(e)}")
            return {'success': False, 'error': str(e)}, 500
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if the file has an allowed extension."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
