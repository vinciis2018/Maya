import requests
import json
import os
from datetime import datetime
import uuid
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import logging
import time
from ..services.vector_store import VectorMemoryStore
from ..services.user_preference_manager import UserPreferenceManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAssistant:
    def __init__(self, agent_name, ollama_base_url, personalities_config, conversation_config):
        self.agent_name = agent_name
        self.ollama_base_url = ollama_base_url
        self.personalities_config = personalities_config
        self.conversation_config = conversation_config
        self.current_personality = 'default'
        self.current_conversation_id = None
        self.conversations = {}
        self.current_user_id = "default_user"  # In a real app, this would come from authentication
        
        # Initialize vector memory store
        self.memory_store = VectorMemoryStore(
            persist_directory=str(Path('data/vector_store'))
        )
        
        # Initialize user preference manager
        self.preference_manager = UserPreferenceManager(
            user_id=self.current_user_id,
            data_dir=str(Path('data/user_preferences'))
        )

        # Personality tips database
        self.tips = {
            "coder": [
                "Use list comprehensions for cleaner Python code",
                "Always validate user input in web applications",
                "Regular commits make version control easier"
            ],
            "creative": [
                "Try freewriting to overcome writer's block",
                "Read your dialogue out loud to test its flow",
                "Use sensory details to immerse your reader"
            ]
        }
        
        # Create data directories if they don't exist
        os.makedirs('conversations', exist_ok=True)
        os.makedirs('data/user_preferences', exist_ok=True)
        
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
    
    def _store_memory(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> str:
        """Store a memory in the vector store.
        
        Args:
            conversation_id: ID of the conversation
            role: 'user' or 'assistant'
            content: The message content
            metadata: Additional metadata to store
            
        Returns:
            str: ID of the stored memory
        """
        if metadata is None:
            metadata = {}
            
        # Add role and timestamp to metadata
        metadata.update({
            'role': role,
            'timestamp': datetime.utcnow().isoformat(),
            'personality': self.current_personality
        })
        
        # Store the memory
        return self.memory_store.add_memory(
            conversation_id=conversation_id,
            text=content,
            metadata=metadata
        )
    
    def search_memories(self, query: str, conversation_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant memories.
        
        Args:
            query: Search query
            conversation_id: Optional conversation ID to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of relevant memories with scores
        """
        return self.memory_store.search_memories(
            query=query,
            conversation_id=conversation_id,
            limit=limit
        )
    
    def get_conversation_context(self, conversation_id: str, query: Optional[str] = None, limit: int = 5) -> str:
        """Get relevant context from conversation history.
        
        Args:
            conversation_id: ID of the conversation
            query: Optional query to find relevant context
            limit: Maximum number of context items to return
            
        Returns:
            str: Formatted context string
        """
        if query:
            # Semantic search for relevant memories
            memories = self.search_memories(
                query=query,
                conversation_id=conversation_id,
                limit=limit
            )
        else:
            # Get most recent memories
            memories = self.memory_store.get_conversation_memories(
                conversation_id=conversation_id,
                limit=limit
            )
        
        # Format memories into context string
        context_parts = []
        for i, memory in enumerate(memories, 1):
            role = memory.get('metadata', {}).get('role', 'unknown')
            context_parts.append(f"{i}. [{role.upper()}] {memory['text']}")
        
        return "\n".join(context_parts) if context_parts else "No relevant context found."
    
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
    
    def _get_prompt_with_preferences(self, base_prompt: str, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """Enhance the base prompt with user preferences and context."""
        # Get user preferences
        style = self.preference_manager.get_preferred_response_style()
        top_topics = self.preference_manager.get_top_topics()
        preferred_length = self.preference_manager.get_preferred_response_length()
        
        # Build style instructions
        style_instructions = [
            f"- Formality level: {'formal' if style['formality'] > 0.6 else 'casual' if style['formality'] < 0.4 else 'neutral'}",
            f"- Verbosity: {'detailed' if style['verbosity'] > 0.6 else 'concise' if style['verbosity'] < 0.4 else 'moderate'}",
            f"- Humor: {'playful' if style['humor_level'] > 0.6 else 'serious' if style['humor_level'] < 0.3 else 'slightly playful'}"
        ]
        
        # Add topic awareness if we have strong topic preferences
        topic_awareness = ""
        if top_topics and top_topics[0][1] > 0.5:  # If strongest topic has score > 0.5
            topic_awareness = "\n\nUser's interests (in order of preference):\n"
            topic_awareness += "\n".join([f"- {topic} (relevance: {score:.1f})" for topic, score in top_topics])
        
        # Build the enhanced prompt
        enhanced_prompt = (
            f"{base_prompt}\n\n"
            "## Response Style Guidelines\n"
            f"{chr(10).join(style_instructions)}\n"
            f"- Target response length: {preferred_length} tokens\n"
            f"{topic_awareness}\n\n"
            "## Current Conversation\n"
        )
        
        # Prepare generation parameters based on preferences
        gen_params = {
            'temperature': min(0.7, 0.4 + (style['humor_level'] * 0.3)),  # More creative if user appreciates humor
            'max_tokens': preferred_length,
            'top_p': 0.9,
            'presence_penalty': 0.5 - (style['verbosity'] * 0.3),  # More focused if user prefers concise
            'frequency_penalty': 0.5 - (style['formality'] * 0.3)  # More formal if user prefers formality
        }
        
        return enhanced_prompt, gen_params
    
    def _generate_response(self, user_input: str, context: Optional[str] = None) -> str:
        """Generate a response using the Ollama API with user preferences."""
        messages = []
        
        # Get base system prompt with personality
        personality_config = self._get_current_config()
        system_prompt = personality_config.get('system_prompt', '')
        
        # Enhance prompt with user preferences and context
        enhanced_prompt, gen_params = self._get_prompt_with_preferences(system_prompt, user_input)
        
        # Add context if available
        if context:
            enhanced_prompt += f"\nContext from previous conversations:\n{context}\n\n"
        
        messages.append({"role": "system", "content": enhanced_prompt})
        
        # Add conversation history
        for msg in self.get_conversation_history()[-6:]:  # Use last 6 messages as context
            messages.append({"role": msg['role'], "content": msg['content']})
        
        # Add current message
        messages.append({"role": "user", "content": user_input})
        
        try:
            # Prepare the request payload with generation parameters
            payload = {
                "model": personality_config.get('model', 'llama2'),
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": gen_params['temperature'],
                    "top_p": gen_params['top_p'],
                    "num_ctx": personality_config.get('num_ctx', 2048),
                    "presence_penalty": gen_params['presence_penalty'],
                    "frequency_penalty": gen_params['frequency_penalty'],
                    "max_tokens": gen_params['max_tokens']
                }
            }
            
            # Make the API request
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                timeout=30  # 30 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result['message']['content']
                
                # Update user preferences based on this interaction
                self.preference_manager.update_from_interaction(user_input, assistant_response)
                
                # Save to conversation history
                if self.conversation_config['enabled']:
                    self.add_to_conversation("user", user_input)
                    self.add_to_conversation("assistant", assistant_response)
                
                # Format response
                if not assistant_response.startswith(self._get_personality_name()):
                    assistant_response = f"{self._get_personality_name()}: {assistant_response}"
                
                return assistant_response
            else:
                error_msg = f"Error processing your request (Status {response.status_code})"
                logger.error(f"API Error: {error_msg}")
                return f"{self._get_personality_name()}: {error_msg}"
                
        except requests.exceptions.RequestException as e:
            error_msg = f"I'm having trouble connecting to my backend. Please try again later. ({str(e)})"
            logger.error(f"Connection Error: {error_msg}")
            return f"{self._get_personality_name()}: {error_msg}"
    
    def process_input(self, user_input, conversation_id=None, personality=None, use_memory: bool = True):
        """Process user input and generate a response.
        
        Args:
            user_input: The user's input text
            conversation_id: Optional conversation ID to continue
            personality: Optional personality to use
            use_memory: Whether to use vector memory for context
            
        Returns:
            dict: Response containing the assistant's reply and metadata
        """
        # Set personality if provided
        if personality and personality in self.personalities_config:
            self.current_personality = personality
            
        # Get or create conversation
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
            self.conversations[conversation_id] = []
        elif conversation_id not in self.conversations:
            loaded_conv = self._load_conversation(conversation_id)
            self.conversations[conversation_id] = loaded_conv['messages'] if loaded_conv and 'messages' in loaded_conv else []
            
        self.current_conversation_id = conversation_id
        
        # Store user message in vector memory
        self._store_memory(
            conversation_id=conversation_id,
            role="user",
            content=user_input,
            metadata={
                "personality": self.current_personality,
                "type": "user_input"
            }
        )
        
        # Add user message to conversation
        self.conversations[conversation_id].append({"role": "user", "content": user_input})
        
        # Get relevant context if using memory
        context = ""
        if use_memory:
            context = self.get_conversation_context(
                conversation_id=conversation_id,
                query=user_input,
                limit=3  # Use top 3 most relevant memories
            )
        
        # Generate response with context
        response = self._generate_response(user_input, context=context if use_memory else None)
        
        # Store assistant response in vector memory
        self._store_memory(
            conversation_id=conversation_id,
            role="assistant",
            content=response,
            metadata={
                "personality": self.current_personality,
                "type": "assistant_response"
            }
        )
        
        # Add assistant response to conversation
        self.conversations[conversation_id].append({"role": "assistant", "content": response})
        
        # Save conversation
        if self.conversation_config['enabled']:
            self._save_conversation(conversation_id, self.conversations[conversation_id])
            self._cleanup_old_conversations()
            
        # Return a properly formatted response
        return {
            'response': response,
            'conversation_id': conversation_id,
            'personality': self.current_personality,
            'context_used': context if use_memory else None
        }