from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any
import os

# Import the document controller
from server.controllers.document_controller import DocumentController

router = APIRouter(prefix="/api")

def get_document_controller() -> DocumentController:
    """Dependency to get the document controller instance."""
    from server.main import app
    return app.state.document_controller

def _handle_controller_result(result):
    """Handle controller results that might be tuples of (response, status_code)."""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return JSONResponse(content=result[0], status_code=result[1])
    return result

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    controller: DocumentController = Depends(get_document_controller)
):
    """Handle file uploads."""
    try:
        # Save the uploaded file temporarily
        file_path = os.path.join(controller.upload_folder, file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Process the file
            result = await controller.upload_file(file_path)
            return _handle_controller_result(result)
        finally:
            # Clean up the temporary file
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/process-url")
async def process_url(
    data: Dict[str, Any] = Body(...),
    controller: DocumentController = Depends(get_document_controller)
):
    """Process a web page URL."""
    try:
        result = controller.process_url(data)
        return _handle_controller_result(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/process-directory")
async def process_directory(
    data: Dict[str, Any] = Body(...),
    controller: DocumentController = Depends(get_document_controller)
):
    """Process all supported documents in a directory."""
    try:
        result = controller.process_directory(data)
        return _handle_controller_result(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/supported-formats")
async def get_supported_formats():
    """Get the list of supported document formats."""
    return {
        'success': True,
        'formats': [
            {
                'type': 'PDF',
                'extensions': ['.pdf'],
                'description': 'Portable Document Format'
            },
            {
                'type': 'Text',
                'extensions': ['.txt', '.md'],
                'description': 'Plain text and markdown files'
            },
            {
                'type': 'JSON',
                'extensions': ['.json'],
                'description': 'JavaScript Object Notation'
            },
            {
                'type': 'Web Page',
                'extensions': ['http://', 'https://'],
                'description': 'Web page URLs'
            }
        ]
    }
