import subprocess
import threading
from typing import Optional
from ..core.config import Config

class TextToSpeech:
    """Handles text-to-speech functionality using macOS's `say` command."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the TTS service.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
    
    def speak(self, text: str) -> bool:
        """Speak the given text.
        
        Args:
            text: Text to speak
            
        Returns:
            True if speech was started successfully, False otherwise
        """
        if not self.config.voice_output_enabled or not text.strip():
            return False
            
        # Stop any currently playing speech
        self.stop()
        
        # Get voice settings from config
        voice_settings = self.config.voice_settings
        voice_name = voice_settings.get('voice_name', 'Samantha')
        rate = voice_settings.get('rate', 175)
        
        # Build the command
        command = ['say', '-v', voice_name, '-r', str(rate)]
        
        # Add pitch if specified
        if 'pitch' in voice_settings:
            command.extend(['-f', str(voice_settings['pitch'])])
            
        command.append(text)
        
        try:
            with self._lock:
                self._process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            return True
        except Exception as e:
            print(f"Error in TTS: {e}", file=sys.stderr)
            return False
    
    def stop(self) -> None:
        """Stop any currently playing speech."""
        with self._lock:
            if self._process and self._process.poll() is None:
                try:
                    # Try graceful termination first
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        # If process didn't terminate, force kill it
                        try:
                            self._process.kill()
                            self._process.wait()
                        except (ProcessLookupError, subprocess.TimeoutExpired):
                            pass
                except ProcessLookupError:
                    # Process already terminated
                    pass
                except Exception as e:
                    print(f"Error stopping TTS: {e}", file=sys.stderr)
                finally:
                    self._process = None
    
    def is_speaking(self) -> bool:
        """Check if speech is currently in progress."""
        with self._lock:
            return self._process is not None and self._process.poll() is None
