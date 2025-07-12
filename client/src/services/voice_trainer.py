import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import whisper
from ..core.voice import VoiceRecorder
from ..core.config import Config

class VoiceTrainer:
    """Handles voice training and recognition functionality."""
    
    def __init__(self, voice_dir: str = "voices", config: Optional[Config] = None):
        """Initialize the voice trainer.
        
        Args:
            voice_dir: Directory to store voice samples
            config: Configuration object
        """
        self.config = config or Config()
        self.voice_dir = voice_dir
        self.samples_dir = os.path.join(voice_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)
        
        # Initialize Whisper model
        self.model = whisper.load_model("tiny.en")
        
        # Load existing transcripts
        self.transcripts = self._load_transcripts()
        
    def _load_transcripts(self) -> List[Dict[str, Any]]:
        """Load existing transcripts from file."""
        transcript_file = os.path.join(self.voice_dir, "transcripts.json")
        if not os.path.exists(transcript_file):
            return []
            
        try:
            with open(transcript_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load transcripts: {e}")
            return []
    
    def _save_transcripts(self) -> bool:
        """Save transcripts to file."""
        transcript_file = os.path.join(self.voice_dir, "transcripts.json")
        try:
            with open(transcript_file, 'w') as f:
                json.dump(self.transcripts, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving transcripts: {e}")
            return False
    
    def record_sample(self, speaker_id: str, duration: int = 5) -> Optional[str]:
        """Record a voice sample for a speaker.
        
        Args:
            speaker_id: Unique identifier for the speaker
            duration: Duration of the recording in seconds
            
        Returns:
            Path to the saved sample, or None if recording failed
        """
        try:
            audio, sample_rate = VoiceRecorder.record_audio(duration=duration)
            speaker_dir = os.path.join(self.samples_dir, speaker_id)
            os.makedirs(speaker_dir, exist_ok=True)
            
            # Save the raw audio
            filename = VoiceRecorder.save_audio(
                audio, 
                sample_rate, 
                speaker_dir,
                prefix=f"sample_{len(os.listdir(speaker_dir)) + 1}"
            )
            
            return filename
        except Exception as e:
            print(f"Error recording sample: {e}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            result = self.model.transcribe(audio_path, fp16=False, language='en')
            return result["text"].strip()
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""
    
    def train_speaker(self, speaker_id: str, sample_text: str) -> bool:
        """Train the system with a speaker's voice sample.
        
        Args:
            speaker_id: Unique identifier for the speaker
            sample_text: Text that the speaker will read
            
        Returns:
            True if training was successful, False otherwise
        """
        print(f"\nPlease read the following text:\n\n{sample_text}\n")
        
        # Record the sample
        sample_path = self.record_sample(speaker_id)
        if not sample_path:
            return False
        
        # Transcribe and verify
        transcription = self.transcribe_audio(sample_path)
        if not transcription:
            print("Error: Could not transcribe the audio.")
            return False
            
        # Calculate similarity (simple word overlap)
        similarity = self._calculate_similarity(sample_text, transcription)
        
        if similarity < 0.7:  # Threshold for acceptable match
            print(f"Warning: Transcription doesn't match expected text (similarity: {similarity:.2f})")
            print(f"Expected: {sample_text}")
            print(f"Got: {transcription}")
            return False
        
        # Save the sample information
        self._save_sample_info(speaker_id, sample_path, sample_text)
        return True
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (0.0 to 1.0)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _save_sample_info(self, speaker_id: str, sample_path: str, text: str) -> None:
        """Save information about a voice sample."""
        # Find or create speaker entry
        speaker_data = next(
            (s for s in self.transcripts if s.get('id') == speaker_id),
            None
        )
        
        if not speaker_data:
            speaker_data = {
                'id': speaker_id,
                'samples': []
            }
            self.transcripts.append(speaker_data)
        
        # Add the new sample
        sample_info = {
            'path': sample_path,
            'text': text,
            'timestamp': str(datetime.now())
        }
        speaker_data['samples'].append(sample_info)
        
        # Save the updated transcripts
        self._save_transcripts()
    
    def list_speakers(self) -> List[Dict[str, Any]]:
        """List all registered speakers and their sample counts."""
        return [
            {
                'id': s['id'],
                'sample_count': len(s.get('samples', [])),
                'last_trained': max(
                    (sample.get('timestamp', '') for sample in s.get('samples', [])),
                    default='Never'
                )
            }
            for s in self.transcripts
        ]
    
    def recognize_speaker(self, audio_path: str) -> Tuple[Optional[str], float]:
        """Try to recognize the speaker from an audio sample.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (speaker_id, confidence) or (None, 0.0) if no match
        """
        # This is a simplified implementation
        # In a real application, you'd use a proper speaker recognition model
        
        # For now, just return the first speaker if we have any
        if not self.transcripts:
            return None, 0.0
            
        return self.transcripts[0]['id'], 0.8  # Fixed confidence for demo
