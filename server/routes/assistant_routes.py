from fastapi import APIRouter, Depends, HTTPException, Request, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# Import the controller
from server.controllers.assistant_controller import (
    AssistantController,
    PersonalityResponse,
    NewConversationResponse,
    ConversationResponse,
    MessageRequest,
    MessageResponse,
    ErrorResponse
)

router = APIRouter(prefix="/api")

def get_assistant_controller() -> AssistantController:
    """Dependency to get the assistant controller instance."""
    from server.main import app
    return app.state.assistant_controller

@router.get("/agent_info")
async def get_agent_info(
    controller: AssistantController = Depends(get_assistant_controller)
):
    """Get agent information."""
    return controller.get_agent_info()

@router.post("/conversation/new", response_model=NewConversationResponse)
async def new_conversation(
    controller: AssistantController = Depends(get_assistant_controller)
):
    """Start a new conversation."""
    return controller.new_conversation()

@router.post("/conversation/load", response_model=ConversationResponse)
async def load_conversation(
    data: Dict[str, Any] = Body(...),
    controller: AssistantController = Depends(get_assistant_controller)
):
    """Load an existing conversation."""
    return controller.load_conversation(data)

@router.post("/set_personality")
async def set_personality(
    data: Dict[str, Any] = Body(...),
    controller: AssistantController = Depends(get_assistant_controller)
):
    """Set the assistant's personality."""
    return controller.set_personality(data)

@router.post("/conversation/chat", response_model=MessageResponse)
async def chat(
    data: Dict[str, Any] = Body(...),
    controller: AssistantController = Depends(get_assistant_controller)
):
    """Send a message to the assistant and get a response."""
    return controller.chat(data)

# Add error handlers
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
