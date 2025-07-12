from flask import Flask, jsonify
import json
import os
from server.assistant import AIAssistant
from server.routes.assistant_routes import create_assistant_routes
from server.controllers.assistant_controller import AssistantController

def create_app():
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    with open(config_path) as config_file:
        config = json.load(config_file)

    # Create Flask app
    app = Flask(__name__)
    
    # Initialize AI Assistant
    assistant = AIAssistant(
        agent_name=config['agent_name'],
        ollama_base_url=config['ollama_base_url'],
        personalities_config=config['personalities'],
        conversation_config=config['conversation_history']
    )
    
    # Initialize controller
    assistant_controller = AssistantController(assistant)
    
    # Register blueprints
    app.register_blueprint(
        create_assistant_routes(assistant_controller),
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