from flask import Flask, request, jsonify
import json
from assistant import AIAssistant

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

app = Flask(__name__)
assistant = AIAssistant(
    agent_name=config['agent_name'],
    ollama_base_url=config['ollama_base_url'],
    personalities_config=config['personalities'],
    conversation_config=config['conversation_history']
)

@app.route('/agent_info', methods=['GET'])
def get_agent_info():
    return jsonify({
        'name': assistant.agent_name,
        'personalities': assistant.get_personalities(),
        'conversation_enabled': assistant.conversation_config['enabled']
    })

@app.route('/conversation/new', methods=['POST'])
def new_conversation():
    success = assistant.start_new_conversation()
    return jsonify({'status': 'success' if success else 'error'})

@app.route('/conversation/load', methods=['POST'])
def load_conversation():
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    success = assistant.load_conversation(conversation_id)
    return jsonify({'status': 'success' if success else 'error'})

@app.route('/set_personality', methods=['POST'])
def set_personality():
    data = request.get_json()
    personality = data.get('personality')
    success = assistant.set_personality(personality)
    return jsonify({'status': 'success' if success else 'error'})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    response = assistant.process_input(user_input)
    return jsonify({
        'response': response,
        'conversation_id': assistant.current_conversation_id
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config['server_port'], debug=True)