from flask import jsonify, request
from typing import Dict, Any, Optional, List
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssistantController:
    """Controller for handling assistant requests with enhanced context awareness."""
    
    def __init__(self, assistant):
        self.assistant = assistant
        self.min_relevance_score = 0.3  # Minimum relevance score for context inclusion
        self.max_context_memories = 5    # Maximum number of context memories to include
        
    def get_personalities(self) -> Dict[str, Any]:
        """Get available personalities."""
        personalities = self.assistant.get_personalities()
        return jsonify({'personalities': personalities})
        
    def new_conversation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new conversation."""
        conversation_id = self.assistant.start_new_conversation()
        return jsonify({
            'status': 'success',
            'conversation_id': conversation_id
        })
        
    def load_conversation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load an existing conversation."""
        conversation_id = data.get('conversation_id')
        if not conversation_id:
            return jsonify({'status': 'error', 'message': 'No conversation_id provided'}), 400
            
        success = self.assistant.load_conversation(conversation_id)
        if not success:
            return jsonify({'status': 'error', 'message': 'Conversation not found'}), 404
            
        # Get conversation history for context
        history = self.assistant.get_conversation_history()
        
        return jsonify({
            'status': 'success',
            'conversation_id': conversation_id,
            'history': history
        })

    def set_personality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Set the assistant's personality."""
        personality = data.get('personality')
        if not personality:
            return jsonify({'status': 'error', 'message': 'No personality specified'}), 400
            
        success = self.assistant.set_personality(personality)
        if not success:
            return jsonify({'status': 'error', 'message': 'Invalid personality'}), 400
            
        return jsonify({
            'status': 'success',
            'personality': personality,
            'message': f'Personality set to {personality}'
        })

    def _get_relevant_context(self, user_input: str, conversation_id: str) -> str:
        """Retrieve relevant context from memory store."""
        try:
            # Get semantically similar memories
            relevant_memories = self.assistant.memory_store.search_memories(
                query=user_input,
                conversation_id=conversation_id,
                limit=self.max_context_memories,
                min_score=self.min_relevance_score
            )
            
            if not relevant_memories:
                return ""
                
            # Format context from relevant memories
            context_parts = []
            for memory in relevant_memories:
                # Skip very recent memories to avoid redundancy
                memory_time = memory['metadata'].get('timestamp', 0)
                if time.time() - memory_time < 60:  # Skip memories from last minute
                    continue
                    
                # Add memory text with metadata
                context_parts.append(
                    f"[{datetime.fromtimestamp(memory_time).strftime('%Y-%m-%d %H:%M')}] "
                    f"{memory['metadata'].get('role', 'user').title()}: {memory['text']}"
                )
            
            if not context_parts:
                return ""
                
            return "\n".join([
                "Relevant context from previous conversations:",
                "-" * 50,
                *context_parts,
                "-" * 50
            ])
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return ""

    def chat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a chat message with enhanced context awareness."""
        user_input = data.get('message', '').strip()
        if not user_input:
            return jsonify({
                'status': 'error',
                'message': 'Empty message'
            }), 400
        
        # Get or create conversation ID
        conversation_id = data.get('conversation_id')
        if not conversation_id:
            conversation_id = self.assistant.start_new_conversation()
        
        # Get relevant context from memory
        context = self._get_relevant_context(user_input, conversation_id)
        
        try:
            # Process the input with context
            response = self.assistant.process_input(
                user_input=user_input,
                conversation_id=conversation_id,
                use_memory=True
            )
            
            # Extract response text and metadata
            response_text = response.get('response', '')
            
            return jsonify({
                'status': 'success',
                'response': response_text,
                'conversation_id': conversation_id,
                'context_used': bool(context)
            })
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Error processing your message',
                'conversation_id': conversation_id
            }), 500
