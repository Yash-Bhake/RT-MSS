"""
Circular buffer for real-time audio processing
Handles STFT framing and overlap-add reconstruction
"""

import torch
import numpy as np
from typing import Optional, Tuple
from collections import deque


class CircularAudioBuffer:
    """
    Circular buffer for managing audio chunks in real-time processing
    Handles:
    - Input buffering for STFT
    - Overlap-add for output reconstruction
    - Zero-latency operation with proper state management
    """
    
    def __init__(self, n_fft: int, hop_length: int, n_channels: int = 2):
        """
        Args:
            n_fft: FFT window size
            hop_length: Hop size between frames
            n_channels: Number of audio channels
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_channels = n_channels
        
        # Input buffer: stores samples waiting to be processed
        # Need at least n_fft samples to compute one STFT frame
        self.input_buffer = torch.zeros(n_channels, n_fft)
        self.input_pos = 0
        
        # Output buffer: stores overlapping output frames for overlap-add
        self.output_buffer = torch.zeros(n_channels, n_fft)
        self.output_pos = 0
        
        # Window for STFT/iSTFT
        self.window = torch.hann_window(n_fft)
        
    def push_audio(self, audio_chunk: torch.Tensor) -> bool:
        """
        Add new audio samples to input buffer
        
        Args:
            audio_chunk: (n_channels, n_samples)
        Returns:
            True if buffer has enough samples for processing
        """
        n_samples = audio_chunk.size(1)
        
        # Shift buffer if needed
        if self.input_pos + n_samples > self.n_fft:
            # Shift left by hop_length
            self.input_buffer[:, :-self.hop_length] = self.input_buffer[:, self.hop_length:].clone()
            self.input_pos -= self.hop_length
        
        # Add new samples
        end_pos = min(self.input_pos + n_samples, self.n_fft)
        samples_to_add = end_pos - self.input_pos
        self.input_buffer[:, self.input_pos:end_pos] = audio_chunk[:, :samples_to_add]
        self.input_pos = end_pos
        
        # Check if we have enough for one frame
        return self.input_pos >= self.n_fft
    
    def get_stft_frame(self) -> Optional[torch.Tensor]:
        """
        Get samples for STFT computation (left-aligned window)
        
        Returns:
            Audio frame (n_channels, n_fft) or None if not enough samples
        """
        if self.input_pos < self.n_fft:
            return None
        
        # Return the frame (left-aligned, no centering for causality)
        return self.input_buffer.clone()
    
    def add_istft_frame(self, frame: torch.Tensor):
        """
        Add reconstructed frame to output buffer with overlap-add
        
        Args:
            frame: (n_channels, n_fft)
        """
        # Apply window
        windowed = frame * self.window.unsqueeze(0)
        
        # Overlap-add
        self.output_buffer += windowed
    
    def pop_audio(self, n_samples: int) -> torch.Tensor:
        """
        Extract processed audio samples from output buffer
        
        Args:
            n_samples: Number of samples to extract (typically hop_length)
        Returns:
            Audio samples (n_channels, n_samples)
        """
        output = self.output_buffer[:, :n_samples].clone()
        
        # Shift buffer
        self.output_buffer[:, :-n_samples] = self.output_buffer[:, n_samples:].clone()
        self.output_buffer[:, -n_samples:] = 0
        
        return output
    
    def reset(self):
        """Reset all buffers"""
        self.input_buffer.zero_()
        self.output_buffer.zero_()
        self.input_pos = 0
        self.output_pos = 0


class STFTProcessor:
    """
    STFT/iSTFT processor with circular buffering for real-time operation
    """
    
    def __init__(self, n_fft: int, hop_length: int, 
                 sample_rate: int, n_channels: int = 2):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        
        self.window = torch.hann_window(n_fft)
        self.buffer = CircularAudioBuffer(n_fft, hop_length, n_channels)
        
    def process_chunk(self, audio_chunk: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Process audio chunk through STFT
        
        Args:
            audio_chunk: (n_channels, n_samples)
        Returns:
            Complex spectrogram (n_channels, n_freq, n_frames) or None
        """
        # Add to buffer
        ready = self.buffer.push_audio(audio_chunk)
        
        if not ready:
            return None
        
        # Get frame for STFT
        frame = self.buffer.get_stft_frame()
        
        if frame is None:
            return None
        
        # Compute STFT (left-aligned)
        spec = torch.stft(
            frame,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,  # Left-aligned for causality
            return_complex=True
        )
        
        return spec
    
    def inverse_chunk(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Inverse STFT and return audio chunk
        
        Args:
            spec: Complex spectrogram (n_channels, n_freq, n_frames)
        Returns:
            Audio samples (n_channels, hop_length)
        """
        # Inverse STFT
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=False
        )
        
        # Add to output buffer
        if audio.size(1) >= self.n_fft:
            frame = audio[:, :self.n_fft]
            self.buffer.add_istft_frame(frame)
        
        # Extract hop_length samples
        output = self.buffer.pop_audio(self.hop_length)
        
        return output
    
    def reset(self):
        """Reset processor state"""
        self.buffer.reset()


class StreamingSTFT:
    """
    Streaming STFT processor with latency tracking
    """
    
    def __init__(self, n_fft: int, hop_length: int, 
                 sample_rate: int, n_channels: int = 2):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        
        self.window = torch.hann_window(n_fft)
        
        # Input queue for buffering
        self.input_queue = deque()
        self.input_buffer = torch.zeros(n_channels, n_fft)
        self.buffer_pos = 0
        
        # Output queue for overlap-add
        self.output_buffer = torch.zeros(n_channels, n_fft * 2)
        self.output_pos = 0
        
        # Latency tracking
        self.algorithmic_latency = n_fft - hop_length
        self.processing_latency = 0
        
    def get_latency_ms(self) -> float:
        """Get total latency in milliseconds"""
        samples = self.algorithmic_latency + self.processing_latency
        return (samples / self.sample_rate) * 1000
    
    def process(self, audio: torch.Tensor) -> Tuple[Optional[torch.Tensor], int]:
        """
        Process audio and return STFT when ready
        
        Args:
            audio: (n_channels, n_samples)
        Returns:
            spec: Complex spectrogram or None
            n_frames: Number of frames in spectrogram
        """
        n_samples = audio.size(1)
        
        # Add to buffer
        remaining_space = self.n_fft - self.buffer_pos
        samples_to_add = min(n_samples, remaining_space)
        
        self.input_buffer[:, self.buffer_pos:self.buffer_pos + samples_to_add] = \
            audio[:, :samples_to_add]
        self.buffer_pos += samples_to_add
        
        # Check if we can compute STFT
        if self.buffer_pos < self.n_fft:
            return None, 0
        
        # Compute STFT
        spec = torch.stft(
            self.input_buffer,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True
        )
        
        # Shift buffer
        self.input_buffer[:, :-self.hop_length] = \
            self.input_buffer[:, self.hop_length:].clone()
        self.input_buffer[:, -self.hop_length:] = 0
        self.buffer_pos -= self.hop_length
        
        # Add remaining samples if any
        if samples_to_add < n_samples:
            remaining = n_samples - samples_to_add
            to_add = min(remaining, self.n_fft - self.buffer_pos)
            self.input_buffer[:, self.buffer_pos:self.buffer_pos + to_add] = \
                audio[:, samples_to_add:samples_to_add + to_add]
            self.buffer_pos += to_add
        
        return spec, spec.size(-1)
    
    def inverse(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Inverse STFT with overlap-add
        
        Args:
            spec: Complex spectrogram (n_channels, n_freq, n_frames)
        Returns:
            Audio (n_channels, hop_length * n_frames)
        """
        # Compute iSTFT
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=False
        )
        
        return audio
    
    def reset(self):
        """Reset all buffers"""
        self.input_buffer.zero_()
        self.output_buffer.zero_()
        self.buffer_pos = 0
        self.output_pos = 0