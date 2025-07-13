from flask import Blueprint, request, jsonify
from server.controllers.document_controller import DocumentController

def create_document_routes(document_controller):
    """Create and configure the document routes.
    
    Args:
        document_controller: Instance of DocumentController
        
    Returns:
        Blueprint: Configured Blueprint instance
    """
    bp = Blueprint('documents', __name__)
    
    @bp.route('/documents/upload', methods=['POST'])
    def upload_document():
        """Handle file uploads."""
        result = document_controller.upload_file()
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)
    
    @bp.route('/documents/process-url', methods=['POST'])
    def process_url():
        """Process a web page URL."""
        data = request.get_json() or {}
        result = document_controller.process_url(data)
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)
    
    @bp.route('/documents/process-directory', methods=['POST'])
    def process_directory():
        """Process all supported documents in a directory."""
        data = request.get_json() or {}
        result = document_controller.process_directory(data)
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)
    
    @bp.route('/documents/supported-formats', methods=['GET'])
    def get_supported_formats():
        """Get the list of supported document formats."""
        return jsonify({
            'success': True,
            'formats': [
                {
                    'type': 'PDF',
                    'extensions': ['.pdf'],
                    'description': 'Portable Document Format'
                },
                {
                    'type': 'Text',
                    'extensions': ['.txt', '.md'],
                    'description': 'Plain text and markdown files'
                },
                {
                    'type': 'JSON',
                    'extensions': ['.json'],
                    'description': 'JavaScript Object Notation'
                },
                {
                    'type': 'Web Page',
                    'extensions': ['http://', 'https://'],
                    'description': 'Web page URLs'
                }
            ]
        })
    
    return bp
