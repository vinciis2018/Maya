from flask import request, jsonify, current_app
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentController:
    """Controller for handling document-related operations."""
    
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
    
    def upload_file(self) -> Dict[str, Any]:
        """Handle file upload and processing."""
        try:
            # Check if the post request has the file part
            if 'file' not in request.files:
                return {'success': False, 'error': 'No file part'}, 400
            
            file = request.files['file']
            
            # If user does not select file, browser might
            # submit an empty part without filename
            if file.filename == '':
                return {'success': False, 'error': 'No selected file'}, 400
            
            if file and self._allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = self.upload_folder / filename
                
                # Save the file temporarily
                file.save(str(filepath))
                
                # Process the file
                success = self.document_processor.process_document(filepath)
                
                # Clean up the temporary file
                try:
                    filepath.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {filepath}: {str(e)}")
                
                if success:
                    return {
                        'success': True,
                        'message': f'Successfully processed {filename}'
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Failed to process {filename}'
                    }, 500
            
            return {
                'success': False,
                'error': f'File type not allowed. Allowed types: {self.allowed_extensions}'
            }, 400
            
        except Exception as e:
            logger.error(f"Error processing file upload: {str(e)}")
            return {
                'success': False,
                'error': f'Internal server error: {str(e)}'
            }, 500
    
    def process_url(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a web page URL."""
        try:
            url = data.get('url')
            if not url:
                return {'success': False, 'error': 'URL is required'}, 400
            
            success = self.document_processor.process_document(url, source_type='web')
            
            if success:
                return {
                    'success': True,
                    'message': f'Successfully processed URL: {url}'
                }
            else:
                return {
                    'success': False,
                    'error': f'Failed to process URL: {url}'
                }, 500
                
        except Exception as e:
            logger.error(f"Error processing URL: {str(e)}")
            return {
                'success': False,
                'error': f'Error processing URL: {str(e)}'
            }, 500
    
    def process_directory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process all supported documents in a directory."""
        try:
            directory = data.get('directory')
            if not directory:
                return {'success': False, 'error': 'Directory path is required'}, 400
            
            recursive = data.get('recursive', True)
            
            results = self.document_processor.process_directory(
                directory=directory,
                recursive=recursive
            )
            
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error processing directory: {str(e)}")
            return {
                'success': False,
                'error': f'Error processing directory: {str(e)}'
            }, 500
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if the file has an allowed extension."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
