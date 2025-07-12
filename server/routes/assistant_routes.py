from flask import Blueprint, request
from server.controllers.assistant_controller import AssistantController

def create_assistant_routes(assistant_controller: AssistantController):
    bp = Blueprint('assistant', __name__)
    
    @bp.route('/agent_info', methods=['GET'])
    def get_agent_info():
        return assistant_controller.get_agent_info()
    
    @bp.route('/conversation/new', methods=['POST'])
    def new_conversation():
        return assistant_controller.new_conversation()
    
    @bp.route('/conversation/load', methods=['POST'])
    def load_conversation():
        return assistant_controller.load_conversation(request.get_json())
    
    @bp.route('/set_personality', methods=['POST'])
    def set_personality():
        return assistant_controller.set_personality(request.get_json())
    
    @bp.route('/chat', methods=['POST'])
    def chat():
        return assistant_controller.chat(request.get_json())
    
    return bp
