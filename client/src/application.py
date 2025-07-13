import os
import signal
import sys
from typing import Dict, Any, Optional
from pathlib import Path

from .core.config import Config
from .services.api_client import APIClient
from .services.voice_trainer import VoiceTrainer
from .services.tts import TextToSpeech
from .ui.console_ui import ConsoleUI

class Application:
    """Main application class for the HEDES client."""
    
    def __init__(self, config_path: str = None):
        """Initialize the application.
        
        Args:
            config_path: Optional path to config file
        """
        # Initialize components
        self.config = Config(config_path)
        self.api_client = APIClient(self.config)
        self.voice_trainer = VoiceTrainer("voices", self.config)
        self.tts = TextToSpeech(self.config)
        self.ui = ConsoleUI()
        
        # State
        self.running = True
        self.current_conversation_id: Optional[str] = None
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """Handle termination signals."""
        self.ui.print_message("\nShutting down...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up resources."""
        self.tts.stop()
    
    def run(self):
        """Run the main application loop."""
        self._show_welcome()
        self._main_loop()
    
    def _show_help(self):
        """Display the help menu with available commands."""
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
        
        # Create a table for commands
        commands_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        commands_table.add_column("Command", style="cyan", width=20)
        commands_table.add_column("Description", style="green")
        
        # Add commands to the table
        commands = [
            ("helpcmd", "Show this help menu"),
            ("new", "Start a new conversation"),
            ("load <id>", "Load a previous conversation by ID"),
            ("personality", "Change AI personality"),
            ("voice", "Toggle voice output on/off"),
            ("voicecmd", "Record and process voice command"),
            ("voicetrain", "Train the voice recognition system"),
            ("stop", "Stop current speech output"),
            ("exit", "Exit the application")
        ]
        
        for cmd, desc in commands:
            commands_table.add_row(cmd, desc)
        
        # Print the help panel
        self.ui.console.print(Panel(
            "Available Commands:",
            title="❓ Help",
            border_style="blue",
            padding=(1, 2)
        ))
        
        self.ui.console.print(commands_table)
        self.ui.console.print("\n")  # Add some space after the help
    
    def _show_welcome(self):
        """Display the welcome message and initial help."""
        from rich.panel import Panel
        from rich.text import Text
        
        # Welcome message
        welcome_msg = Text("\nWelcome to ", style="bold")
        welcome_msg.append("Maya", style="bold blue")
        welcome_msg.append(", your personalized AI assistant!")
        
        # Memory status
        memory_status = "[green]ENABLED" if self.config.get('memory', {}).get('enabled', True) else "[red]DISABLED"
        memory_msg = f"Memory Feature: {memory_status}"
        
        # Print the welcome panel
        self.ui.console.print(Panel(
            welcome_msg,
            title="✨ Maya ✨",
            border_style="blue",
            padding=(1, 2)
        ))
        
        # Print status panel
        self.ui.console.print(Panel(
            memory_msg,
            title="ℹ️  Status",
            border_style="green",
            padding=(1, 2)
        ))
        
        # Show help command hint
        self.ui.console.print("\nType 'helpcmd' to see available commands.\n")
                
    def _main_loop(self):
        """Main application loop."""
        while self.running:
            try:
                user_input = self.ui.get_user_input("> ").strip()
                
                if not user_input:
                    continue
                    
                # Handle commands (start with !)
                if user_input.startswith('!'):
                    cmd = user_input[1:].lower()
                    self._process_command(cmd)
                    continue
                
                # If we get here, it's a regular chat message
                self._handle_chat_message(user_input)
                    
            except KeyboardInterrupt:
                self.ui.print_message("\nType 'exit' to quit or press Ctrl+C again to force quit.")
            except Exception as e:
                self.ui.print_error(f"An error occurred: {str(e)}")
    
    def _handle_new_conversation(self):
        """Handle starting a new conversation."""
        if self.api_client.new_conversation():
            self.ui.print_success("Started a new conversation")
        else:
            self.ui.print_error("Failed to start a new conversation")
    
    def _handle_load_conversation(self, conversation_id: str):
        """Handle loading a previous conversation."""
        if not conversation_id:
            self.ui.print_error("Please specify a conversation ID")
            return
            
        if self.api_client.load_conversation(conversation_id):
            self.current_conversation_id = conversation_id
            self.ui.print_success(f"Loaded conversation {conversation_id}")
        else:
            self.ui.print_error(f"Failed to load conversation {conversation_id}")
    
    def _handle_switch_personality(self):
        """Handle switching the AI's personality."""
        personalities = list(self.config.personalities.keys())
        if not personalities:
            self.ui.print_error("No personalities configured")
            return
            
        # Let user select a personality
        self.ui.print_message("\nAvailable personalities:")
        for i, personality in enumerate(personalities, 1):
            self.ui.print_message(f"{i}. {personality}")
            
        try:
            choice = int(self.ui.get_user_input("\nSelect personality (number): ")) - 1
            if 0 <= choice < len(personalities):
                personality = personalities[choice]
                if self.api_client.set_personality(personality):
                    self.ui.print_success(f"Switched to {personality} personality")
                else:
                    self.ui.print_error("Failed to switch personality")
            else:
                self.ui.print_error("Invalid selection")
        except ValueError:
            self.ui.print_error("Please enter a valid number")
    
    def _toggle_voice(self):
        """Toggle voice output on/off."""
        current = self.config.get('voice_output', {}).get('enabled', False)
        self.config._config.setdefault('voice_output', {})['enabled'] = not current
        status = "enabled" if not current else "disabled"
        self.ui.print_success(f"Voice output {status}")
    
    def _process_command(self, cmd):
        """Process a command without showing it in chat or sending to AI."""
        if cmd == 'exit':
            self.running = False
        elif cmd == 'new':
            self._handle_new_conversation()
        elif cmd.startswith('load '):
            conv_id = cmd[5:].strip()
            self._handle_load_conversation(conv_id)
        elif cmd == 'personality':
            self._handle_switch_personality()
        elif cmd == 'voice':
            self._toggle_voice()
        elif cmd == 'voicecmd':
            self._handle_voice_command()
        elif cmd == 'voicetrain':
            self._handle_voice_training()
        elif cmd == 'helpcmd':
            self._show_help()
        else:
            self.ui.print_error(f"Unknown command: {cmd}")

    def _handle_voice_command(self):
        """Handle voice command input."""
        self.ui.print_message("Listening... (press Ctrl+C to cancel)")
        
        try:
            # Record audio using VoiceRecorder
            from src.core.voice import VoiceRecorder
            
            # Record for 5 seconds
            audio, sample_rate = VoiceRecorder.record_audio(duration=5)
            
            # Save to a temporary file
            temp_file = "temp_voice_command.wav"
            VoiceRecorder.save_audio(audio, sample_rate, temp_file)
            
            # Transcribe using Whisper
            transcription = self.voice_trainer.transcribe_audio(temp_file)
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            if transcription:
                self.ui.print_message(f"You said: {transcription}")
                
                # Check if the transcription is a command (starts with !)
                if transcription.startswith('!'):
                    # Process as a command without showing in chat
                    cmd = transcription[1:].lower()
                    self._process_command(cmd)
                else:
                    # Process as a regular chat message
                    self._handle_chat_message(transcription)
            else:
                self.ui.print_error("Could not transcribe audio")
                
        except KeyboardInterrupt:
            self.ui.print_message("\nVoice input cancelled")
        except Exception as e:
            self.ui.print_error(f"Error processing voice command: {str(e)}")
    
    def _handle_voice_training(self):
        """Handle voice training flow."""
        self.ui.print_message("\nVoice Training")
        self.ui.print_message("-------------")
        
        # Get or create speaker ID
        speaker_id = self.ui.get_user_input("Enter your name (for voice recognition): ").strip()
        if not speaker_id:
            self.ui.print_error("Name cannot be empty")
            return
            
        # Sample text for training
        sample_text = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump!"
        )
        
        self.ui.print_message("\nPlease read the following text when prompted:")
        self.ui.print_message(f"\n{sample_text}")
        
        if not self.ui.confirm("\nReady to begin?"):
            return
            
        try:
            if self.voice_trainer.train_speaker(speaker_id, sample_text):
                self.ui.print_success("Voice training completed successfully!")
            else:
                self.ui.print_error("Voice training failed")
        except Exception as e:
            self.ui.print_error(f"Error during voice training: {str(e)}")
    
    def _handle_voice_training(self):
        """Handle voice training flow."""
        self.ui.print_message("\nVoice Training")
        self.ui.print_message("-------------")
        
        # Get or create speaker ID
        speaker_id = self.ui.get_user_input("Enter your name (for voice recognition): ").strip()
        if not speaker_id:
            self.ui.print_error("Name cannot be empty")
            return
            
        # Sample text for training
        sample_text = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump!"
        )
        
        self.ui.print_message("\nPlease read the following text when prompted:")
        self.ui.print_message(f"\n{sample_text}")
        
        if not self.ui.confirm("\nReady to begin?"):
            return
            
        try:
            if self.voice_trainer.train_speaker(speaker_id, sample_text):
                self.ui.print_success("Voice training completed successfully!")
            else:
                self.ui.print_error("Voice training failed")
        except Exception as e:
            self.ui.print_error(f"Error during voice training: {str(e)}")
    
    def _handle_chat_message(self, message: str):
        """Handle sending a chat message to the AI."""
        if not message.strip():
            return
            
        try:
            # Check for stop command
            if message.strip().lower() == 'stop':
                self.tts.stop()
                self.ui.print_message("\nStopping speech...")
                return
                
            # Show user message
            self.ui.print_message(f"\nYou: {message}")
            # Show AI prefix without newline
            self.ui.print_message("Maya: ", end="")
            
            # Send to server
            response = self.api_client.chat(message)
            
            # Update conversation ID if needed
            if 'conversation_id' in response:
                self.current_conversation_id = response['conversation_id']
                
            # Extract the response text
            if isinstance(response, dict):
                # If the response has a 'response' key, use that
                if 'response' in response:
                    response_text = response['response']
                    # If the response is a string, clean it up
                    if isinstance(response_text, str):
                        # Remove any leading 'Maya:' or 'AI:' prefix if present
                        message_content = response_text.split(':', 1)[-1].strip()
                    else:
                        message_content = str(response_text)
                else:
                    # If no 'response' key, convert the whole response to string
                    message_content = str(response)
            else:
                # If response is not a dict, convert to string
                message_content = str(response)
                
            self.ui.print_message(f"{message_content}")
            
            # Speak the response if voice is enabled
            if self.config.voice_output_enabled:
                self.tts.speak(message_content)
                
        except Exception as e:
            self.ui.print_error(f"Error communicating with server: {str(e)}")

def main():
    """Entry point for the application."""
    try:
        app = Application()
        app.run()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Ensure cleanup happens
        if 'app' in locals():
            app.cleanup()

if __name__ == "__main__":
    main()
