#!/usr/bin/env python3
"""
Voice Trainer Module

This module provides voice training and recognition functionality.
It's now a thin wrapper around the main implementation in src/services/voice_trainer.py
"""
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_dir = str(Path(__file__).parent.parent / 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    # Import the main implementation
    from src.services.voice_trainer import VoiceTrainer as CoreVoiceTrainer
    
    class VoiceTrainer(CoreVoiceTrainer):
        """Wrapper for backward compatibility"""
        def __init__(self, voice_dir="voices", transcript_file=None):
            # Ignore transcript_file parameter as it's handled by the core implementation
            super().__init__(voice_dir=voice_dir)
        
        def train_with_transcript(self, speaker_id, transcript):
            """Train with a specific transcript (for backward compatibility)"""
            return self.train_speaker(speaker_id, transcript)
        
        def test_recognition(self):
            """Test speaker recognition (for backward compatibility)"""
            return self.recognize_speaker()
    
    def list_speakers(trainer):
        """List all trained speakers and their sample counts"""
        speakers_info = trainer.list_speakers()
        return [
            f"{s['id']} - {s['sample_count']} samples (last: {s['last_trained']})"
            for s in speakers_info
        ]
    
    def view_samples(trainer, speaker_id):
        """View samples for a specific speaker"""
        samples = trainer.get_speaker_samples(speaker_id)
        return [
            f"{s['filename']} - {s['transcript']} ({s['timestamp']})"
            for s in samples
        ]

except ImportError as e:
    print(f"Error: Could not import core voice trainer: {e}")
    print("Please make sure the src directory is in your Python path.")
    
    # Provide a dummy implementation to prevent import errors
    class VoiceTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("Core voice trainer not available")
    
    def list_speakers(trainer):
        return []
    
    def view_samples(trainer, speaker_id):
        return []

# For direct script execution
if __name__ == "__main__":
    print("This module is part of the HEDES client and is not meant to be run directly.")
    print("Please use the main client application instead.")
