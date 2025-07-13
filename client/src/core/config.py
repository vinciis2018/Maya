import json
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Handles application configuration"""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to the config file. If None, looks for 'config.json' in the project root.
        """
        if config_path is None:
            # Look for config.json in the project root
            project_root = Path(__file__).resolve().parent.parent.parent.parent  # Go up to project root
            config_path = project_root / 'config.json'
            
        self.config_path = str(config_path)
        print(f"Loading config from: {self.config_path}")
        self._config = self._load_config()
        print(f"Server port from config: {self._config.get('server_port')}")
        print(f"Server URL: {self.server_url}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self._config.get(key, default)
    
    @property
    def server_url(self) -> str:
        """Get the server URL from config.
        
        Returns:
            str: The base URL for the FastAPI server with /api prefix (e.g., 'http://localhost:3009/api')
        """
        # Use 127.0.0.1 instead of localhost for more reliable connections
        host = self.get('server_host', '127.0.0.1')
        port = self.get('server_port', 3009)  # Default FastAPI port is 3009
        return f"http://{host}:{port}/api"  # Add /api prefix to match FastAPI routes
    
    @property
    def voice_output_enabled(self) -> bool:
        """Check if voice output is enabled."""
        return self._config.get('voice_output', {}).get('enabled', False)
    
    @property
    def voice_settings(self) -> Dict[str, Any]:
        """Get voice output settings."""
        return self._config.get('voice_output', {})
    
    @property
    def agent_name(self) -> str:
        """Get the agent name from config."""
        return self._config.get('agent_name', 'AI Assistant')
    
    @property
    def personalities(self) -> Dict[str, Any]:
        """Get personality configurations."""
        return self._config.get('personalities', {})
