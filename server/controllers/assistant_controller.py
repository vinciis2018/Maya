from flask import jsonify
from server.assistant import AIAssistant

class AssistantController:
    def __init__(self, assistant: AIAssistant):
        self.assistant = assistant

    def get_agent_info(self):
        return jsonify({
            'name': self.assistant.agent_name,
            'personalities': self.assistant.get_personalities(),
            'conversation_enabled': self.assistant.conversation_config['enabled']
        })

    def new_conversation(self):
        success = self.assistant.start_new_conversation()
        return jsonify({'status': 'success' if success else 'error'})

    def load_conversation(self, data):
        conversation_id = data.get('conversation_id')
        success = self.assistant.load_conversation(conversation_id)
        return jsonify({'status': 'success' if success else 'error'})

    def set_personality(self, data):
        personality = data.get('personality')
        success = self.assistant.set_personality(personality)
        return jsonify({'status': 'success' if success else 'error'})

    def chat(self, data):
        user_input = data.get('message', '')
        response = self.assistant.process_input(user_input)
        return jsonify({
            'response': response,
            'conversation_id': self.assistant.current_conversation_id
        })
