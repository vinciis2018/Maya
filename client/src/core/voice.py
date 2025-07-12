import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from datetime import datetime
import uuid
from typing import Optional, Tuple

class VoiceRecorder:
    """Handles voice recording functionality"""
    
    @staticmethod
    def record_audio(duration: int = 5, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
        """Record audio for the specified duration.
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        return audio.flatten(), sample_rate
    
    @staticmethod
    def save_audio(
        audio: np.ndarray,
        sample_rate: int,
        output_dir: str,
        prefix: str = "recording"
    ) -> str:
        """Save audio data to a WAV file.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate in Hz
            output_dir: Directory to save the file
            prefix: Prefix for the filename
            
        Returns:
            Path to the saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"{prefix}_{timestamp}.wav")
        wav.write(filename, sample_rate, audio)
        return filename
