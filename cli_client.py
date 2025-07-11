import requests
import json
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress
from rich.layout import Layout
from rich.box import ROUNDED
import os
import subprocess
from threading import Thread
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import signal
import sys
import threading
from voice_trainer import VoiceTrainer, list_speakers, view_samples

# Global variables to track program state
program_running = True
current_speaker_process = None

console = Console()

# Function to record audio
def record_audio(duration=5, sample_rate=16000):
    console.print(f"\n[bold]Speak now[/bold] (recording for {duration} seconds)")
    audio = sd.rec(int(duration * sample_rate), 
                  samplerate=sample_rate, 
                  channels=1,
                  dtype='float32')
    sd.wait()  # Wait until recording is finished
    return audio.flatten(), sample_rate

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

SERVER_URL = f"http://localhost:{config['server_port']}"

# Initialize voice trainer
voice_trainer = VoiceTrainer()

# Function to speak text using macOS say command with configurable voice
def speak(text):
    global current_speaker_process
    
    if not config['voice_output']['enabled'] or not program_running:
        return None
    
    def speak_thread():
        global current_speaker_process
        
        # Clean the text for the say command
        clean_text = text.replace('"', '"')
        
        # Get voice settings from config
        voice_name = config['voice_output'].get('voice_name', 'Samantha')
        rate = config['voice_output'].get('rate', 175)
        pitch = config['voice_output'].get('pitch', 0.8)
        
        # Build the say command with all parameters
        command = ['say', '-v', voice_name, '-r', str(rate)]
        if pitch != 1.0:
            command.extend(['-f', str(pitch)])
        command.append(clean_text)
        
        try:
            # Start the process and store it
            current_speaker_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            current_speaker_process.wait()
        except Exception as e:
            print(f"Error in voice output: {e}")
        finally:
            current_speaker_process = None
            
    # Create thread and store it for cleanup
    thread = Thread(target=speak_thread, daemon=True)
    thread.start()
    return thread

# Function to get voice input
def get_voice_input():
    temp_file = None
    try:
        # Record audio with silence detection
        # Record and save temporary file
        audio_data, sample_rate = record_audio(duration=5)
        
        # Check if we got any audio above silence threshold
        if len(audio_data) < 1024:  # Less than ~50ms of audio
            console.print("[yellow]No speech detected, please try again[/yellow]")
            return None
            
        temp_file = os.path.join('voices', 'temp_command.wav')
        os.makedirs('voices', exist_ok=True)
        wav.write(temp_file, sample_rate, audio_data)
        
        # First check if we can get a clean transcription
        transcription = voice_trainer.transcribe_audio(temp_file)
        if not transcription or len(transcription.strip()) < 3:  # At least 3 characters
            console.print("[yellow]Could not transcribe any speech, please try again[/yellow]")
            return None
        
        # Only then check speaker recognition
        speaker, confidence = voice_trainer.recognize_speaker(temp_file)
        
        if speaker and confidence >= 0.7:  # Minimum confidence threshold
            if config['voice_output']['enabled']:
                speak(f"I recognize {speaker}")
            
            if transcription:
                return transcription
        else:
            # Still allow the command if we have a good transcription
            if transcription:
                return transcription
                
        console.print("[red]Could not process voice command[/red]")
        return None
        
    except Exception as e:
        console.print(f"[red]Error in voice command: {e}[/red]")
        return None
        
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not remove temp file: {e}[/yellow]")

def display_welcome():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main")
    )
    
    header_text = Text(f"✨ {config['agent_name']} AI Assistant ✨", style="bold blue", justify="center")
    layout["header"].update(Panel(header_text, box=ROUNDED))
    
    personality_text = "\n".join(
        f"  [bold magenta]{name}[/bold magenta]: {details['description']}"
        for name, details in config['personalities'].items()
    )
    
    memory_status = "ON" if config['conversation_history']['enabled'] else "OFF"
    voice_status = "ON" if config['voice_output']['enabled'] else "OFF"
    
    welcome_msg = Text(f"""
Welcome to {config['agent_name']}, your personalized AI assistant!
    
Memory Feature: {memory_status}
Voice Output: {voice_status}
    
Available Personalities:
{personality_text}

Commands:
  new - Start a new conversation
  load <id> - Load previous conversation
  switch - Change personality
  voice - Toggle voice output
  voicecmd - Use voice command instead of typing
  voicetrain - Train voice recognition
  exit - End the session
""")
    
    layout["main"].update(Panel(welcome_msg))
    console.print(layout)
    if config['voice_output']['enabled']:
        speak("Welcome to " + config['agent_name'] + ", your personalized AI assistant!")

def set_personality(personality):
    try:
        response = requests.post(
            f"{SERVER_URL}/set_personality",
            json={'personality': personality},
            headers={'Content-Type': 'application/json'}
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def new_conversation():
    try:
        response = requests.post(f"{SERVER_URL}/conversation/new")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def load_conversation(conversation_id):
    try:
        response = requests.post(
            f"{SERVER_URL}/conversation/load",
            json={'conversation_id': conversation_id},
            headers={'Content-Type': 'application/json'}
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def list_conversations():
    if not os.path.exists('conversations'):
        return []
    
    conv_files = [f for f in os.listdir('conversations') if f.endswith('.json')]
    conversations = []
    
    for f in conv_files:
        try:
            with open(f"conversations/{f}", 'r') as cf:
                data = json.load(cf)
                conversations.append({
                    'id': data['id'],
                    'personality': data['personality'],
                    'last_updated': data['last_updated'],
                    'message_count': len(data['messages'])
                })
        except:
            continue
    
    return sorted(conversations, key=lambda x: x['last_updated'], reverse=True)

def voice_trainer_menu():
    """Handle the voice trainer menu"""
    while True:
        console.print("\n[bold]Voice Trainer Menu[/bold]")
        console.print("1. Train with transcript")
        console.print("2. Test recognition")
        console.print("3. List speakers")
        console.print("4. View speaker samples")
        console.print("5. Back to main menu")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            speaker_id = input("\nEnter speaker ID: ").strip()
            if not speaker_id:
                console.print("[red]Speaker ID cannot be empty[/red]")
                continue
                
            transcript = input("\nEnter transcript: ").strip()
            if not transcript:
                console.print("[red]Transcript cannot be empty[/red]")
                continue
                
            with console.status("[cyan]Training with transcript..."):
                success = voice_trainer.train_with_transcript(speaker_id, transcript)
                if success:
                    console.print(f"\n[green]Successfully trained with transcript for {speaker_id}![/green]")
                    if config['voice_output']['enabled']:
                        speak(f"Successfully trained with transcript for {speaker_id}")
                else:
                    console.print("[red]Failed to train with transcript[/red]")
                    
        elif choice == '2':
            speaker, confidence = voice_trainer.test_recognition()
            if speaker:
                console.print(f"\n[green]Recognized speaker: {speaker} (confidence: {confidence:.2f})[/green]")
                if config['voice_output']['enabled']:
                    speak(f"I recognize {speaker} with {confidence:.0%} confidence")
            else:
                console.print(f"\n[red]Speaker not recognized (confidence: {confidence:.2f})[/red]")
                if config['voice_output']['enabled']:
                    speak("I couldn't recognize the speaker with sufficient confidence")
        
        elif choice == '3':
            speakers = list_speakers(voice_trainer)
            if not speakers:
                console.print("\n[yellow]No speakers found in the training data.[/yellow]")
                if config['voice_output']['enabled']:
                    speak("No speakers found in the training data")
            
        elif choice == '4':
            speaker_id = input("\nEnter speaker ID to view samples: ").strip()
            if speaker_id:
                view_samples(voice_trainer, speaker_id)
            else:
                console.print("[red]Please enter a speaker ID[/red]")
                
        elif choice == '5':
            console.print("\nReturning to main menu...")
            break
            
        else:
            console.print("[red]Invalid choice. Please enter a number between 1 and 5.[/red]")

def chat_loop():
    # Start a new conversation by default
    with console.status("[cyan]Initializing conversation..."):
        new_conversation()
    
    # Import required modules at function level
    import select
    import sys
    import time
    
    # Function to get input with timeout
    def get_input(timeout=0.1):
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.readline().strip()
        return None
    
    # Clear any existing input
    def clear_input_buffer():
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
    
    while True:
        # Check if we're currently speaking
        is_speaking = current_speaker_process and current_speaker_process.poll() is None
        
        if is_speaking:
            # Show the stop prompt once when starting to speak
            console.print("\n[bold red]Press Enter to interrupt[/bold red] (or type 'stop' and press Enter)")
            
            # While speaking, check for stop command
            while current_speaker_process and current_speaker_process.poll() is None:
                user_input = get_input(0.1)  # Check for input with 100ms timeout
                if user_input and user_input.lower() in ['stop', '']:
                    stop_voice()
                    clear_input_buffer()
                    break
            
            # Clear any remaining input and continue to normal input
            clear_input_buffer()
            continue
        
        # Normal input when not speaking
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
            if not user_input:  # Skip empty input
                continue
                
            if user_input.lower() == 'stop':
                stop_voice()
                continue
                
        except (KeyboardInterrupt, EOFError):
            console.print("\nType 'exit' to quit or 'new' to start a new conversation")
            continue
        
        if user_input.lower() in ['exit', 'quit']:
            break
            
        if user_input.lower() == 'new':
            with console.status("[cyan]Starting new conversation..."):
                if new_conversation():
                    console.print("[green]Started a new conversation![/green]")
                else:
                    console.print("[red]Failed to start new conversation.[/red]")
            continue
            
        if user_input.lower().startswith('load '):
            conv_id = user_input[5:].strip()
            with console.status("[cyan]Loading conversation..."):
                if load_conversation(conv_id):
                    console.print(f"[green]Loaded conversation {conv_id}[/green]")
                else:
                    console.print("[red]Failed to load conversation.[/red]")
            continue
            
        if user_input.lower() == 'list':
            conversations = list_conversations()
            if conversations:
                console.print("\n[bold]Recent Conversations:[/bold]")
                for conv in conversations[:5]:  # Show last 5 conversations
                    console.print(f"  [yellow]{conv['id']}[/yellow] - {conv['personality']} ({conv['message_count']} messages)")
            else:
                console.print("[yellow]No previous conversations found.[/yellow]")
            continue
            
        if user_input.lower() == 'voicecmd':
            # Stop any current speech before starting voice command
            stop_voice()
            
            # Get voice input without showing too many system messages
            voice_text = get_voice_input()
            
            # Process the recognized text
            if voice_text and voice_text.strip():
                # Remove any system prompts that might have been recognized
                prompts = ["please speak your command now", "listening", "speak now"]
                for prompt in prompts:
                    if voice_text.lower().startswith(prompt):
                        voice_text = voice_text[len(prompt):].strip()
                
                if voice_text:  # Only if there's actual user input left
                    console.print(f"\n[bold]You said:[/bold] {voice_text}")
                    user_input = voice_text
                else:
                    console.print("[red]No valid input detected[/red]")
                    continue
            else:
                console.print("[red]Failed to get voice input[/red]")
                continue
                
        if user_input.lower() == 'voicetrain':
            console.print("\n[bold]Voice Recognition Training[/bold]")
            console.print("This will help me recognize your voice. You'll need to read some text samples.")
            if config['voice_output']['enabled']:
                speak("Let's train voice recognition. You'll need to read some text samples.")
            
            # Go to voice trainer menu
            voice_trainer_menu()
            continue
            
        if user_input.lower() == 'voice':
            config['voice_output']['enabled'] = not config['voice_output']['enabled']
            status = "ON" if config['voice_output']['enabled'] else "OFF"
            console.print(f"\nVoice output is now: [bold]{status}[/bold]")
            if config['voice_output']['enabled']:
                speak("Voice output is now enabled")
            else:
                speak("Voice output is now disabled")
            continue
            
        if user_input.lower() == 'switch':
            agent_info = get_agent_info()
            if agent_info:
                console.print("\n[bold]Available Personalities:[/bold]")
                for name, desc in agent_info['personalities'].items():
                    console.print(f"  [bold magenta]{name}[/bold magenta]: {desc}")
                
                choice = console.input("\n[bold]Enter personality to switch to:[/bold] ").lower()
                if choice in agent_info['personalities']:
                    with console.status("[cyan]Switching personality..."):
                        if set_personality(choice):
                            console.print(f"\n[green]Switched to {choice} personality![/green]")
                        else:
                            console.print("[red]Failed to switch personality.[/red]")
                else:
                    console.print("[red]Invalid personality choice.[/red]")
            continue
            
        try:
            with console.status("[cyan]Thinking..."):
                response = requests.post(
                    f"{SERVER_URL}/chat",
                    json={'message': user_input},
                    headers={'Content-Type': 'application/json'}
                )
                
            if response.status_code == 200:
                data = response.json()
                if config['voice_output']['enabled']:
                    # Only speak the actual response text, not the prefix
                    speak(data['response'].split(':', 1)[-1].strip())
                
                console.print(f"\n[bold]{config['agent_name']}:[/bold]")
                console.print(data['response'])
                console.print(f"\nConversation ID: {data['conversation_id']}")
                console.print(f"[dim]Conversation ID: {data['conversation_id']}[/dim]")
            else:
                console.print(f"[red]Error:[/red] Received status code {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error connecting to server:[/red] {e}")

def cleanup():
    """Cleanup function to stop voice output and threads"""
    global program_running, current_speaker_process
    program_running = False
    
    # Stop any running voice output
    stop_voice()
    
    # Wait for any running threads to finish
    for thread in threading.enumerate():
        if thread != threading.current_thread() and thread.is_alive():
            thread.join(timeout=1.0)

def stop_voice():
    """Stop any currently playing voice output"""
    global current_speaker_process
    if current_speaker_process:
        try:
            # Kill the say process
            subprocess.run(['pkill', '-f', 'say'])
            current_speaker_process = None
        except Exception as e:
            print(f"Error stopping voice: {e}")

def signal_handler(sig, frame):
    """Handle signals to ensure proper cleanup"""
    print("\nCleaning up and exiting...")
    cleanup()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, signal_handler) # Handle termination

def main():
    try:
        display_welcome()
        chat_loop()
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"\nError: {e}")
        cleanup()
    finally:
        cleanup()
    console.print(f"\n[bold magenta]Goodbye from {config['agent_name']}![/bold magenta]")

if __name__ == '__main__':
    main()