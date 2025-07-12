import requests
from typing import Dict, Any, Optional
from ..core.config import Config

class APIClient:
    """Handles all API communication with the server"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the API client.
        
        Args:
            config: Configuration object. If None, a new one will be created.
        """
        self.config = config or Config()
        self.base_url = self.config.server_url
    
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
        headers = kwargs.pop('headers', {})
        headers.setdefault('Content-Type', 'application/json')
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                **kwargs
            )
            response.raise_for_status()
            return response.json() if response.text else {}
        except requests.exceptions.RequestException as e:
            # Re-raise with more context
            raise requests.exceptions.RequestException(
                f"Failed to {method.upper()} {endpoint}: {str(e)}"
            ) from e
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent."""
        return self._request('get', '/agent_info')
    
    def new_conversation(self) -> bool:
        """Start a new conversation."""
        response = self._request('post', '/conversation/new')
        return response.get('status') == 'success'
    
    def load_conversation(self, conversation_id: str) -> bool:
        """Load a previous conversation."""
        response = self._request('post', '/conversation/load', json={'conversation_id': conversation_id})
        return response.get('status') == 'success'
    
    def set_personality(self, personality: str) -> bool:
        """Set the agent's personality."""
        response = self._request('post', '/set_personality', json={'personality': personality})
        return response.get('status') == 'success'
    
    def chat(self, message: str) -> Dict[str, Any]:
        """Send a chat message to the agent."""
        return self._request('post', '/chat', json={'message': message})
