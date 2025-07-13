from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import json
import os
from typing import Dict, Any

# Import existing application components
from server.assistant import AIAssistant
from server.services.vector_store import VectorMemoryStore
from server.services.document_processor import DocumentProcessor

# Import FastAPI controllers
from server.controllers.assistant_controller import AssistantController
from server.controllers.document_controller import DocumentController

# Import FastAPI routes
from server.routes.assistant_routes import router as assistant_router
from server.routes.document_routes import router as document_router

def create_app():
    """Create and configure the FastAPI application."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    with open(config_path) as config_file:
        config = json.load(config_file)

    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    # Create FastAPI app
    app = FastAPI(
        title="HEDES API",
        description="HEDES AI Assistant API",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize core components
    vector_store = VectorMemoryStore(
        persist_directory=str(data_dir / 'document_store'),
        model_name=config.get('embedding_model', 'all-MiniLM-L6-v2')
    )

    assistant = AIAssistant(
        agent_name=config['agent_name'],
        ollama_base_url=config['ollama_base_url'],
        personalities_config=config['personalities'],
        conversation_config=config['conversation_history']
    )

    document_processor = DocumentProcessor(vector_store)

    # Initialize controllers
    app.state.assistant_controller = AssistantController(assistant)
    app.state.document_controller = DocumentController(
        document_processor=document_processor,
        upload_folder=str(data_dir / 'uploads')
    )

    # Include routers
    app.include_router(assistant_router)
    app.include_router(document_router)

    # Health check endpoint
    @app.get("/api/health")
    async def health_check():
        return {"status": "healthy"}

    # Exception handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "message": exc.detail}
        )
    
    return app, config

# Create the app instance
app, config = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Bind to all available network interfaces
        port=3009,
        reload=True,
        log_level="debug"
    )
