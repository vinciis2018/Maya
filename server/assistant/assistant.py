import requests
import json
import os
from datetime import datetime
import uuid

class AIAssistant:
    def __init__(self, agent_name, ollama_base_url, personalities_config, conversation_config):
        self.agent_name = agent_name
        self.ollama_base_url = ollama_base_url
        self.personalities_config = personalities_config
        self.conversation_config = conversation_config
        self.current_personality = 'default'
        self.current_conversation_id = None
        self.conversations = {}
        
        # Create conversations directory if it doesn't exist
        os.makedirs('conversations', exist_ok=True)
        
        # Verify Ollama connection
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError("Could not connect to Ollama")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Ollama connection failed: {e}")
    
    def get_personalities(self):
        return {name: details['description'] for name, details in self.personalities_config.items()}
    
    def set_personality(self, personality):
        if personality in self.personalities_config:
            self.current_personality = personality
            return True
        return False
    
    def _get_current_config(self):
        return self.personalities_config.get(self.current_personality, 
                                          self.personalities_config['default'])
    
    def _get_personality_name(self):
        personality_map = {
            'default': self.agent_name,
            'coder': f"Code{self.agent_name}",
            'visual': f"Vision{self.agent_name}",
            'creative': f"Muse{self.agent_name}",
            'concise': f"Brief{self.agent_name}"
        }
        return personality_map.get(self.current_personality, self.agent_name)
    
    def _load_conversation(self, conversation_id):
        """Load a conversation from file"""
        file_path = f"conversations/{conversation_id}.json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def _save_conversation(self, conversation_id, messages):
        """Save conversation to file"""
        file_path = f"conversations/{conversation_id}.json"
        with open(file_path, 'w') as f:
            json.dump({
                'id': conversation_id,
                'personality': self.current_personality,
                'messages': messages,
                'last_updated': datetime.now().isoformat()
            }, f)
    
    def _cleanup_old_conversations(self):
        """Remove oldest conversations if we exceed the limit"""
        if not self.conversation_config['enabled']:
            return
        
        conv_files = [f for f in os.listdir('conversations') if f.endswith('.json')]
        if len(conv_files) > self.conversation_config['max_conversations']:
            # Sort by modification time (oldest first)
            conv_files.sort(key=lambda x: os.path.getmtime(f"conversations/{x}"))
            # Delete oldest files
            for f in conv_files[:len(conv_files) - self.conversation_config['max_conversations']]:
                os.remove(f"conversations/{f}")
    
    def start_new_conversation(self):
        """Initialize a new conversation"""
        if not self.conversation_config['enabled']:
            self.current_conversation_id = None
            return True
        
        self.current_conversation_id = str(uuid.uuid4())
        self._save_conversation(self.current_conversation_id, [])
        self._cleanup_old_conversations()
        return True
    
    def load_conversation(self, conversation_id):
        """Load an existing conversation"""
        if not self.conversation_config['enabled']:
            return False
        
        conversation = self._load_conversation(conversation_id)
        if conversation:
            self.current_conversation_id = conversation_id
            self.set_personality(conversation['personality'])
            return True
        return False
    
    def get_conversation_history(self):
        """Get current conversation messages"""
        if not self.conversation_config['enabled'] or not self.current_conversation_id:
            return []
        
        conversation = self._load_conversation(self.current_conversation_id)
        return conversation['messages'] if conversation else []
    
    def add_to_conversation(self, role, content):
        """Add a message to current conversation"""
        if not self.conversation_config['enabled'] or not self.current_conversation_id:
            return
        
        messages = self.get_conversation_history()
        messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trim conversation if it exceeds max messages
        if len(messages) > self.conversation_config['max_messages']:
            messages = messages[-self.conversation_config['max_messages']:]
        
        self._save_conversation(self.current_conversation_id, messages)
    
    def process_input(self, user_input):
        if not user_input.strip():
            return f"{self._get_personality_name()}: I didn't receive any input. Please try again."
        
        config = self._get_current_config()
        
        # Start new conversation if none exists
        if self.conversation_config['enabled'] and not self.current_conversation_id:
            self.start_new_conversation()
        
        # Format messages for Ollama API
        messages = [
            {"role": "system", "content": config['system_prompt']}
        ]
        
        # Add conversation history if enabled
        if self.conversation_config['enabled']:
            for msg in self.get_conversation_history()[-6:]:  # Use last 6 messages as context
                messages.append({"role": msg['role'], "content": msg['content']})
        
        # Add current message
        messages.append({"role": "user", "content": user_input})
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": config['model'],
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_ctx": 2048
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result['message']['content']
                
                # Save to conversation history
                if self.conversation_config['enabled']:
                    self.add_to_conversation("user", user_input)
                    self.add_to_conversation("assistant", assistant_response)
                
                # Format response
                if not assistant_response.startswith(self._get_personality_name()):
                    assistant_response = f"{self._get_personality_name()}: {assistant_response}"
                
                return assistant_response
            else:
                return f"{self._get_personality_name()}: Error processing your request (Status {response.status_code})"
                
        except requests.exceptions.RequestException as e:
            return f"{self._get_personality_name()}: I'm having trouble connecting to my backend. Please try again later."