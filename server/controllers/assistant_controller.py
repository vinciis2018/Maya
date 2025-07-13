from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response Models
class PersonalityResponse(BaseModel):
    personalities: Dict[str, Any]

class NewConversationResponse(BaseModel):
    status: str
    conversation_id: str

class ConversationResponse(BaseModel):
    status: str
    conversation_id: str
    history: List[Dict[str, Any]]

class MessageRequest(BaseModel):
    message: str
    conversation_id: str
    personality: Optional[str] = None

class MessageResponse(BaseModel):
    status: str
    response: str
    conversation_id: str
    timestamp: float

class ErrorResponse(BaseModel):
    status: str
    message: str

class SearchRequest(BaseModel):
    query: str
    search_type: str = 'hybrid'
    conversation_id: Optional[str] = None
    limit: int = 5
    min_score: float = 0.1

class SearchResult(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}
    score: float
    search_type: str

class SearchResponse(BaseModel):
    status: str
    results: List[SearchResult]
    query: str
    search_type: str

class AssistantController:
    """Controller for handling assistant requests with enhanced context awareness."""
    
    def __init__(self, assistant):
        self.assistant = assistant
        self.min_relevance_score = 0.3  # Minimum relevance score for context inclusion
        self.max_context_memories = 5    # Maximum number of context memories to include
        
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent."""
        return {
            "status": "success",
            "agent_name": self.assistant.agent_name,
            "personalities": self.assistant.get_personalities(),
            "capabilities": ["chat", "context_awareness", "memory"]
        }
        
    def get_personalities(self) -> PersonalityResponse:
        """Get available personalities."""
        personalities = self.assistant.get_personalities()
        return PersonalityResponse(personalities=personalities)
        
    def new_conversation(self) -> NewConversationResponse:
        """Start a new conversation."""
        success = self.assistant.start_new_conversation()
        if not success or not self.assistant.current_conversation_id:
            raise HTTPException(status_code=500, detail="Failed to start a new conversation")
            
        return NewConversationResponse(
            status="success",
            conversation_id=self.assistant.current_conversation_id
        )
        
    def load_conversation(self, conversation_id: str) -> ConversationResponse:
        """Load an existing conversation."""
        if not conversation_id:
            raise HTTPException(status_code=400, detail="No conversation_id provided")
            
        success = self.assistant.load_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        # Get conversation history for context
        history = self.assistant.get_conversation_history()
        
        return ConversationResponse(
            status="success",
            conversation_id=conversation_id,
            history=history
        )

    def set_personality(self, personality: str) -> Dict[str, str]:
        """Set the assistant's personality."""
        if not personality:
            raise HTTPException(status_code=400, detail="No personality specified")
            
        success = self.assistant.set_personality(personality)
        if not success:
            raise HTTPException(status_code=400, detail="Invalid personality")
            
        return {
            "status": "success",
            "personality": personality,
            "message": f"Personality set to {personality}"
        }

    def chat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a chat message and return the assistant's response.
        
        If no conversation_id is provided, a new conversation will be created.
        """
        try:
            message = data.get('message', '').strip()
            conversation_id = data.get('conversation_id')
            personality = data.get('personality')
            
            if not message:
                raise HTTPException(status_code=400, detail="Message cannot be empty")
                
            # Create a new conversation if no ID is provided
            if not conversation_id:
                new_conv = self.new_conversation()
                conversation_id = new_conv.conversation_id
                logger.info(f"Created new conversation: {conversation_id}")
                
            # Set personality if provided
            if personality:
                self.set_personality(personality)
            
            # Load the conversation to ensure it exists
            try:
                self.assistant.load_conversation(conversation_id)
            except Exception as e:
                logger.warning(f"Failed to load conversation {conversation_id}, creating new one")
                new_conv = self.new_conversation()
                conversation_id = new_conv.conversation_id
            
            # Get relevant context
            context = self._get_relevant_context(message, conversation_id)
            
            # Process the input using the assistant
            result = self.assistant.process_input(
                user_input=message,
                conversation_id=conversation_id,
                use_memory=bool(context)  # Use memory if context was provided
            )
            
            # Extract the response from the result
            if not result or 'response' not in result:
                raise HTTPException(status_code=500, detail="Failed to generate response")
                
            return {
                'status': 'success',
                'response': result['response'],
                'conversation_id': result.get('conversation_id', conversation_id),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    def process_message(self, request: MessageRequest) -> MessageResponse:
        """Process a user message and return the assistant's response."""
        try:
            # Set personality if provided
            if request.personality:
                self.set_personality(request.personality)
            
            # Get relevant context
            context = self._get_relevant_context(
                request.message, 
                request.conversation_id
            )
            
            # Process the message with context
            response = self.assistant.process_message(
                message=request.message,
                conversation_id=request.conversation_id,
                context=context
            )
            
            return MessageResponse(
                status="success",
                response=response,
                conversation_id=request.conversation_id,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def search(self, query: str, search_type: str = 'hybrid', 
              conversation_id: Optional[str] = None) -> SearchResponse:
        """Search for information in the knowledge base.
        
        Args:
            query: The search query
            search_type: Type of search ('hybrid', 'semantic', or 'keyword')
            conversation_id: Optional conversation ID to include in the search context
            
        Returns:
            SearchResponse with results
        """
        try:
            logger.debug(f"Starting search with query: '{query}', type: {search_type}, conversation_id: {conversation_id}")
            
            if not query:
                raise HTTPException(status_code=400, detail="No search query provided")
                
            # Validate search type
            if search_type not in ['hybrid', 'semantic', 'keyword']:
                error_msg = f"Invalid search type: {search_type}. Must be 'hybrid', 'semantic', or 'keyword'"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
            
            logger.debug(f"Searching for query: {query}")
            
            # Perform the search
            search_results = self.assistant.search_memories(
                query=query,
                conversation_id=conversation_id,
                search_type=search_type
            )
            
            logger.debug(f"Search returned {len(search_results)} results")
            
            # Convert to SearchResult objects
            results = [
                SearchResult(
                    text=result.get('text', ''),
                    metadata=result.get('metadata', {}),
                    score=result.get('score', 0.0),
                    search_type=result.get('search_type', search_type)
                )
                for result in search_results
            ]
            
            logger.debug(f"Formatted {len(results)} search results")
            
            response = SearchResponse(
                status="success",
                results=results,
                query=query,
                search_type=search_type
            )
            
            logger.debug(f"Returning search response with {len(results)} results")
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
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
            logger.error(f"Error getting context: {str(e)}")
            return ""
