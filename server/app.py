from flask import Flask, jsonify
import json
import os
from pathlib import Path
from server.assistant import AIAssistant
from server.routes.assistant_routes import create_assistant_routes
from server.controllers.assistant_controller import AssistantController
from server.services.vector_store import VectorMemoryStore
from server.services.document_processor import DocumentProcessor
from server.controllers.document_controller import DocumentController
from server.routes.document_routes import create_document_routes

def create_app():
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    with open(config_path) as config_file:
        config = json.load(config_file)

    # Create Flask app
    app = Flask(__name__)
    
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Initialize vector store for document storage
    vector_store = VectorMemoryStore(
        persist_directory=str(data_dir / 'document_store'),
        model_name=config.get('embedding_model', 'all-MiniLM-L6-v2')
    )
    
    # Initialize AI Assistant
    assistant = AIAssistant(
        agent_name=config['agent_name'],
        ollama_base_url=config['ollama_base_url'],
        personalities_config=config['personalities'],
        conversation_config=config['conversation_history']
    )
    
    # Initialize document processor
    document_processor = DocumentProcessor(vector_store)
    
    # Initialize controllers
    assistant_controller = AssistantController(assistant)
    document_controller = DocumentController(
        document_processor=document_processor,
        upload_folder=str(data_dir / 'uploads')
    )
    
    # Register blueprints
    app.register_blueprint(
        create_assistant_routes(assistant_controller),
        url_prefix='/api'
    )
    
    app.register_blueprint(
        create_document_routes(document_controller),
        url_prefix='/api'
    )
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'healthy'}), 200
    
    return app, config

if __name__ == '__main__':
    app, config = create_app()
    app.run(host='0.0.0.0', port=config['server_port'], debug=True)