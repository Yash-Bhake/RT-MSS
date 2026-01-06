"""
Data augmentation for music source separation
Includes pitch shifting and gain scaling
"""

import torch
import torchaudio
import random
import numpy as np
from typing import Tuple, List


class PitchShift:
    """
    Pitch shift augmentation using torchaudio
    """
    
    def __init__(self, sample_rate: int, shift_range: List[int] = [-2, -1, 1, 2]):
        """
        Args:
            sample_rate: Sample rate in Hz
            shift_range: List of semitone shifts (e.g., [-2, -1, 1, 2])
        """
        self.sample_rate = sample_rate
        self.shift_range = shift_range
        
    def __call__(self, audio: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:
        """
        Apply pitch shift
        
        Args:
            audio: (n_channels, n_samples)
            n_steps: Number of semitones to shift (random if None)
        Returns:
            Pitch-shifted audio
        """
        if n_steps is None:
            n_steps = random.choice(self.shift_range)
        
        if n_steps == 0:
            return audio
        
        # Apply pitch shift per channel
        shifted_channels = []
        for ch in range(audio.size(0)):
            # Create pitch shift effect
            effects = [
                ["pitch", str(n_steps * 100)],  # Convert semitones to cents
                ["rate", str(self.sample_rate)]
            ]
            
            shifted, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio[ch:ch+1], self.sample_rate, effects
            )
            shifted_channels.append(shifted)
        
        return torch.cat(shifted_channels, dim=0)


class GainScale:
    """
    Gain scaling augmentation
    """
    
    def __init__(self, gain_range: Tuple[float, float] = (-10, 10)):
        """
        Args:
            gain_range: Range of gain in dB (min, max)
        """
        self.gain_range = gain_range
        
    def __call__(self, audio: torch.Tensor, gain_db: Optional[float] = None) -> torch.Tensor:
        """
        Apply gain scaling
        
        Args:
            audio: (n_channels, n_samples)
            gain_db: Gain in dB (random if None)
        Returns:
            Scaled audio
        """
        if gain_db is None:
            gain_db = random.uniform(self.gain_range[0], self.gain_range[1])
        
        # Convert dB to linear scale
        gain_linear = 10 ** (gain_db / 20)
        
        return audio * gain_linear


class SourceActivityDetection:
    """
    Simple energy-based source activity detector
    Removes silent segments from training data
    """
    
    def __init__(self, threshold_db: float = -40, 
                 chunk_ratio: float = 0.5,
                 quantile: float = 0.15):
        """
        Args:
            threshold_db: Energy threshold in dB
            chunk_ratio: Minimum ratio of active chunks required
            quantile: Quantile for threshold calculation
        """
        self.threshold_db = threshold_db
        self.chunk_ratio = chunk_ratio
        self.quantile = quantile
        
    def is_active(self, audio: torch.Tensor, segment_length: float = 6.0,
                  sample_rate: int = 44100) -> bool:
        """
        Check if audio segment is active (non-silent)
        
        Args:
            audio: (n_channels, n_samples)
            segment_length: Length of segment in seconds
            sample_rate: Sample rate
        Returns:
            True if segment is active
        """
        n_samples = audio.size(1)
        segment_samples = int(segment_length * sample_rate)
        
        # Split into chunks
        n_chunks = 10
        chunk_size = segment_samples // n_chunks
        
        energies = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n_samples)
            chunk = audio[:, start:end]
            
            # Calculate energy (RMS)
            energy = torch.sqrt(torch.mean(chunk ** 2))
            
            # Convert to dB
            energy_db = 20 * torch.log10(energy + 1e-8)
            energies.append(energy_db.item())
        
        # Calculate threshold
        energies = np.array(energies)
        threshold = max(np.quantile(energies, self.quantile), self.threshold_db)
        
        # Check if enough chunks are above threshold
        active_chunks = np.sum(energies > threshold)
        active_ratio = active_chunks / n_chunks
        
        return active_ratio >= self.chunk_ratio


class MixAugmentation:
    """
    Mix augmentation: randomly mix stems from different songs
    with optional pitch shift and gain scaling
    """
    
    def __init__(self, config: dict, sample_rate: int = 44100):
        """
        Args:
            config: Augmentation configuration
            sample_rate: Sample rate
        """
        self.sample_rate = sample_rate
        self.config = config
        
        # Initialize augmentations
        if config.get('pitch_shift', False):
            self.pitch_shift = PitchShift(
                sample_rate, 
                config.get('pitch_shift_range', [-2, -1, 1, 2])
            )
        else:
            self.pitch_shift = None
        
        if config.get('gain_scale', False):
            self.gain_scale = GainScale(
                tuple(config.get('gain_range', [-10, 10]))
            )
        else:
            self.gain_scale = None
        
        self.mix_prob = config.get('mix_prob', 0.5)
        
    def __call__(self, stems: Dict[str, torch.Tensor], 
                 apply_augmentation: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create augmented mixture
        
        Args:
            stems: Dictionary of source stems {'vocals': tensor, 'drums': tensor, ...}
            apply_augmentation: Whether to apply augmentation
        Returns:
            mixture: Mixed audio
            target: Target source
        """
        # Randomly decide whether to augment
        if not apply_augmentation or random.random() > self.mix_prob:
            # No augmentation, return original
            target = stems['target']
            mixture = sum(stems.values())
            return mixture, target
        
        # Apply augmentation to each stem
        augmented_stems = {}
        
        for stem_name, stem_audio in stems.items():
            # Pitch shift with some probability
            if self.pitch_shift is not None and random.random() < 0.5:
                stem_audio = self.pitch_shift(stem_audio)
            
            # Gain scaling with some probability
            if self.gain_scale is not None and random.random() < 0.5:
                stem_audio = self.gain_scale(stem_audio)
            
            augmented_stems[stem_name] = stem_audio
        
        # Mix all stems
        mixture = sum(augmented_stems.values())
        target = augmented_stems['target']
        
        return mixture, target


class RandomChunkSelector:
    """
    Randomly select chunks from audio tracks
    """
    
    def __init__(self, chunk_length: float, sample_rate: int = 44100):
        """
        Args:
            chunk_length: Length of chunk in seconds
            sample_rate: Sample rate
        """
        self.chunk_length = chunk_length
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_length * sample_rate)
        
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Select random chunk from audio
        
        Args:
            audio: (n_channels, n_samples)
        Returns:
            Chunk (n_channels, chunk_samples)
        """
        n_samples = audio.size(1)
        
        if n_samples <= self.chunk_samples:
            # Pad if too short
            padding = self.chunk_samples - n_samples
            audio = torch.nn.functional.pad(audio, (0, padding))
            return audio
        
        # Random start position
        max_start = n_samples - self.chunk_samples
        start = random.randint(0, max_start)
        
        return audio[:, start:start + self.chunk_samples]


class EnergyNormalization:
    """
    Normalize energy of mixed audio
    """
    
    def __init__(self, target_lufs: float = -23.0):
        """
        Args:
            target_lufs: Target loudness in LUFS
        """
        self.target_lufs = target_lufs
        
    def __call__(self, mixture: torch.Tensor, target: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize energy
        
        Args:
            mixture: Mixed audio (n_channels, n_samples)
            target: Target audio (n_channels, n_samples)
        Returns:
            Normalized mixture and target
        """
        # Calculate max absolute value
        max_val = max(
            torch.abs(mixture).max().item(),
            torch.abs(target).max().item()
        )
        
        if max_val > 0:
            # Scale to prevent clipping
            scale = 0.95 / max_val
            mixture = mixture * scale
            target = target * scale
        
        return mixture, target