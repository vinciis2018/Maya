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
    
    def search_memories(self, query: str, conversation_id: Optional[str] = None, 
                      limit: int = 5, search_type: str = 'hybrid', 
                      min_score: float = 0.1) -> List[Dict[str, Any]]:
        """Search for relevant memories.
        
        Args:
            query: Search query
            conversation_id: Optional conversation ID to filter by
            limit: Maximum number of results to return
            search_type: Type of search - 'hybrid', 'semantic', or 'keyword'
            min_score: Minimum relevance score for results
            
        Returns:
            List of relevant memories with scores and search types
        """
        return self.memory_store.search_memories(
            query=query,
            conversation_id=conversation_id,
            search_type=search_type,
            min_score=min_score,
            limit=limit
        )
    
    def get_conversation_context(self, conversation_id: str, query: str = "", limit: int = 5) -> str:
        """Get relevant context from conversation history with enhanced semantic search.
        
        Args:
            conversation_id: ID of the conversation
            query: The current user query to find relevant context
            limit: Maximum number of context items to return
            
        Returns:
            str: Formatted context string with relevant information
        """
        from pathlib import Path
        context_parts = []
        
        # 1. Get semantic search results for the current query from both conversation and documents
        if query:
            # Search in current conversation
            semantic_memories = self.search_memories(
                query=query,
                conversation_id=conversation_id,
                limit=min(limit, 2)  # Limit to top 2 most relevant from conversation
            )
            
            # Also search in documents using the document processor's vector store
            try:
                from ..services.document_processor import DocumentProcessor
                doc_processor = DocumentProcessor(None)  # We only need the vector store
                doc_memories = doc_processor.vector_store.search_memories(
                    query=query,
                    limit=min(limit, 3)  # Get slightly more from documents
                )
                
                # Format document memories to match conversation memory format
                formatted_doc_memories = [{
                    'text': mem.get('text', ''),
                    'metadata': {
                        **mem.get('metadata', {}),
                        'source': mem.get('metadata', {}).get('source', 'document')
                    },
                    'score': mem.get('score', 0.0)
                } for mem in doc_memories if mem.get('score', 0) > 0.3]  # Filter by minimum score
                
            except Exception as e:
                logger.error(f"Error searching document store: {str(e)}")
                formatted_doc_memories = []
            
            # Combine and sort all memories by relevance
            all_memories = semantic_memories + formatted_doc_memories
            all_memories.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            if all_memories:
                context_parts.append("## Relevant information from knowledge base:")
                for mem in all_memories[:limit]:  # Apply limit after combining
                    source = mem.get('metadata', {}).get('source', 'conversation')
                    if source == 'conversation':
                        role = mem.get('metadata', {}).get('role', 'user').upper()
                        context_parts.append(f"- [Conversation] [{role}] {mem.get('text', '')}")
                    else:
                        source_name = Path(source).name if source != 'document' else 'document'
                        context_parts.append(f"- [Document: {source_name}] {mem.get('text', '')}")
        
        # 2. Get recent messages for flow
        recent_memories = self.memory_store.get_conversation_memories(
            conversation_id=conversation_id,
            limit=3  # Last 3 messages for flow
        )
        
        if recent_memories:
            context_parts.append("\n## Recent conversation:")
            for mem in recent_memories:
                role = mem.get('metadata', {}).get('role', 'user').upper()
                context_parts.append(f"- [{role}] {mem.get('text', '')}")
        
        # 3. Extract and include important facts
        important_facts = self._extract_important_facts(conversation_id)
        if important_facts:
            context_parts.append("\n## Important information:")
            for fact in important_facts:
                context_parts.append(f"- {fact}")
        
        return "\n".join(context_parts) if context_parts else "No relevant context found."
    
    def _extract_important_facts(self, conversation_id: str) -> List[str]:
        """Extract important facts and entities from the conversation."""
        facts = set()
        
        # Get recent user messages
        messages = [msg for msg in self.get_conversation_history() 
                   if msg.get('role') == 'user']
        
        if not messages:
            return []
            
        # Combine recent messages for analysis
        recent_text = " ".join([msg.get('content', '') for msg in messages[-5:]])  # Last 5 user messages
        
        # Simple fact extraction (can be enhanced with NER in the future)
        import re
        
        # Extract names (simple pattern matching)
        names = re.findall(r'\b(?:my name is|i am|i\'m) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 
                          recent_text, re.IGNORECASE)
        if names:
            facts.add(f"The user's name is {names[-1]}")
        
        # Extract preferences
        pref_patterns = [
            (r'i (like|love|enjoy) ([^\.,;!?]+)', "User likes: {}"),
            (r'my (favorite|least favorite) (\w+) is ([^\.,;!?]+)', "User's favorite {}: {}"),
            (r'i (am|feel) ([^\.,;!?]+)', "User is feeling: {}")
        ]
        
        for pattern, template in pref_patterns:
            matches = re.finditer(pattern, recent_text, re.IGNORECASE)
            for match in matches:
                groups = [g for g in match.groups() if g and len(g) > 2]  # Filter out short matches
                if groups:
                    fact = template.format(*groups[-2:])  # Use last 1-2 groups
                    facts.add(fact.capitalize())
        
        return list(facts)[:5]  # Return up to 5 most recent facts
    
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
        """Enhance the base prompt with user preferences, context, and conversation history."""
        # Get user preferences and style
        style = self.preference_manager.get_preferred_response_style()
        top_topics = self.preference_manager.get_top_topics()
        
        # Get relevant context based on current conversation
        context = self.get_conversation_context(
            self.current_conversation_id,
            query=user_input,
            limit=5
        )
        
        # Build style instructions
        formality = 'formal' if style['formality'] > 0.6 else 'casual' if style['formality'] < 0.4 else 'neutral'
        verbosity = 'detailed' if style['verbosity'] > 0.6 else 'concise' if style['verbosity'] < 0.4 else 'moderate'
        
        # Build the enhanced prompt
        enhanced_prompt = f"""{base_prompt}

## Response Guidelines
- Tone: {formality.capitalize()} and {verbosity}
- Style: {'Engaging and conversational' if formality != 'formal' else 'Professional and structured'}
- Key Focus: {user_input}

## Current Context
{context}

## Instructions
1. Consider the entire conversation context when responding
2. Acknowledge and reference important details from earlier in the conversation
3. Keep responses clear and to the point
4. Maintain a {formality} tone
5. Provide {verbosity} responses

## Important Notes
- User's name and preferences should be remembered and used appropriately
- If the user shares personal information, remember it for future reference
- Be consistent with previously established facts
"""
        
        # Get creativity with default value of 0.5 if not present
        creativity = style.get('creativity', 0.5)
        
        # Prepare generation parameters
        gen_params = {
            'temperature': min(0.8, 0.3 + (creativity * 0.5)),  # 0.3 to 0.8 based on creativity
            'max_tokens': 500,  # Will be overridden by the API call if needed
            'top_p': 0.9,
            'presence_penalty': 0.2,  # Slight penalty for repetition
            'frequency_penalty': 0.1,  # Slight penalty for frequent tokens
        }
        
        # Adjust parameters based on verbosity preference
        if verbosity == 'detailed':
            gen_params['max_tokens'] = 800
        elif verbosity == 'concise':
            gen_params['max_tokens'] = 300
            gen_params['temperature'] = max(0.2, gen_params['temperature'] - 0.1)  # Slightly more focused
        
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
        """Process user input and generate a response with enhanced context awareness.
        
        Args:
            user_input: The user's input text
            conversation_id: Optional conversation ID to continue
            personality: Optional personality to use
            use_memory: Whether to use vector memory for context
            
        Returns:
            dict: Response containing the assistant's reply and metadata
        """
        start_time = time.time()
        logger.info(f"Processing input: {user_input[:100]}...")
        
        try:
            # Set personality if provided
            if personality and personality in self.personalities_config:
                self.current_personality = personality
                logger.debug(f"Set personality to: {personality}")
            
            # Get or create conversation
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())
                self.conversations[conversation_id] = []
                logger.info(f"Created new conversation: {conversation_id}")
            elif conversation_id not in self.conversations:
                loaded_conv = self._load_conversation(conversation_id)
                if loaded_conv and 'messages' in loaded_conv:
                    self.conversations[conversation_id] = loaded_conv['messages']
                    logger.info(f"Loaded existing conversation: {conversation_id} with {len(loaded_conv['messages'])} messages")
                else:
                    self.conversations[conversation_id] = []
                    logger.warning(f"Conversation {conversation_id} not found, created new one")
            
            self.current_conversation_id = conversation_id
            
            # Store user message in vector memory with enhanced metadata
            memory_metadata = {
                "personality": self.current_personality,
                "type": "user_input",
                "timestamp": datetime.utcnow().isoformat(),
                "message_length": len(user_input),
                "word_count": len(user_input.split())
            }
            
            # Extract and store entities from user input
            entities = self._extract_entities(user_input)
            if entities:
                memory_metadata["entities"] = entities
            
            self._store_memory(
                conversation_id=conversation_id,
                role="user",
                content=user_input,
                metadata=memory_metadata
            )
            logger.debug(f"Stored user input in memory with metadata: {memory_metadata}")
            
            # Add user message to conversation history
            self.conversations[conversation_id].append({"role": "user", "content": user_input})
            
            # Get relevant context if using memory
            context = ""
            if use_memory and self.memory_store:
                context = self.get_conversation_context(conversation_id, user_input)
                logger.debug(f"Retrieved context: {context}")
            
            # Generate response
            response = self._generate_response(user_input, context)
            
            # Store assistant's response in memory
            if use_memory and self.memory_store:
                self._store_memory(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=response,
                    metadata={
                        "personality": self.current_personality,
                        "type": "response",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
            # Add assistant's response to conversation
            self.conversations[conversation_id].append({"role": "assistant", "content": response})
            
            # Save conversation
            if self.conversation_config['enabled']:
                self._save_conversation(conversation_id, self.conversations[conversation_id])
            
            # Log processing time
            processing_time = time.time() - start_time
            logger.info(f"Processed input in {processing_time:.2f}s")
            
            return {
                'response': response,
                'conversation_id': conversation_id,
                'personality': self.current_personality,
                'context_used': bool(context),
                'processing_time_seconds': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}", exc_info=True)
            return {
                'response': "I'm sorry, I encountered an error processing your request. Please try again.",
                'conversation_id': conversation_id or str(uuid.uuid4()),
                'error': str(e),
                'status': 'error'
            }
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text using simple pattern matching.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            Dictionary of entity types to lists of entity values
        """
        import re
        from collections import defaultdict
        
        entities = defaultdict(list)
        
        # Common patterns for entity extraction
        patterns = {
            'name': [
                r'\b(?:my name is|i am|i\'m|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'\b(name\s*:?|i\'m\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'phone': [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'
            ],
            'date': [
                r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}\b',
                r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
            ],
            'url': [
                r'\bhttps?://[^\s/$.?#].[^\s]*\b'
            ]
        }
        
        # Extract entities using patterns
        for entity_type, regex_list in patterns.items():
            for pattern in regex_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Get the first non-None group that's not the entire match
                    value = next((g for g in match.groups() if g and g != match.group(0)), match.group(0))
                    if value and value.lower() not in ['name', 'i am', "i'm", 'call me']:
                        entities[entity_type].append(value.strip())
        
        # Remove duplicates while preserving order
        for key in entities:
            seen = set()
            entities[key] = [x for x in entities[key] if not (x in seen or seen.add(x))]
            
        return dict(entities)