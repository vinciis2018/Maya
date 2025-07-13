import requests
from typing import Dict, Any, Optional, Union, BinaryIO
from pathlib import Path
from ..core.config import Config

class APIClient:
    """Handles all API communication with the FastAPI server"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the API client.
        
        Args:
            config: Configuration object. If None, a new one will be created.
        """
        self.config = config or Config()
        self.base_url = self.config.server_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the server.
        
        Args:
            method: HTTP method (get, post, etc.)
            endpoint: API endpoint (without leading slash)
            **kwargs: Additional arguments to pass to requests.request
            
        Returns:
            JSON response as a dictionary
            
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method=method.upper(),
                url=url,
                **kwargs
            )
            response.raise_for_status()
            return response.json() if response.text else {}
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code} Error: {e.response.text}"
            raise requests.exceptions.RequestException(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"Failed to {method.upper()} {endpoint}: {str(e)}"
            ) from e
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent."""
        return self._request('GET', '/agent_info')
    
    def new_conversation(self) -> Dict[str, Any]:
        """Start a new conversation."""
        return self._request('POST', '/conversation/new')
    
    def load_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Load a previous conversation."""
        return self._request('POST', '/conversation/load', json={"conversation_id": conversation_id})
    
    def set_personality(self, personality: str) -> Dict[str, Any]:
        """Set the agent's personality."""
        return self._request('POST', '/set_personality', json={"personality": personality})
    
    def chat(self, message: str, conversation_id: Optional[str] = None, personality: Optional[str] = None) -> Dict[str, Any]:
        """Send a chat message to the agent.
        
        Args:
            message: The message to send to the agent
            conversation_id: Optional conversation ID. If not provided, a new conversation will be started.
            personality: Optional personality to set before sending the message.
            
        Returns:
            Response from the server with the agent's reply
        """
        payload = {"message": message}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if personality:
            payload["personality"] = personality
            
        return self._request('POST', '/conversation/chat', json=payload)
    
    def upload_document(self, file_path: Union[str, Path], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Upload a document to the server.
        
        Args:
            file_path: Path to the file to upload
            metadata: Optional metadata to include with the file
            
        Returns:
            Response from the server
        """
        url = f"{self.base_url}/documents/upload"
        file_path = Path(file_path)
        
        with open(file_path, 'rb') as f:
            files = {
                'file': (file_path.name, f, 'application/octet-stream')
            }
            data = {'metadata': str(metadata or {})}
            
            try:
                response = self.session.post(url, files=files, data=data)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                raise requests.exceptions.RequestException(
                    f"Failed to upload document: {str(e)}"
                ) from e
    
    def process_url(self, url: str) -> Dict[str, Any]:
        """Process a URL and extract content.
        
        Args:
            url: URL to process
            
        Returns:
            Response from the server
        """
        return self._request('POST', '/documents/process-url', json={"url": url})
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """Get a list of supported document formats."""
        return self._request('GET', '/documents/supported-formats')
