import os
import random
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class MUSDB18Dataset(Dataset):
    """MUSDB18 dataset for vocal separation."""
    
    def __init__(self, root_dir, split='train', segment_length=3.0, 
                 sample_rate=44100, use_augmentation=False, augmentation_config=None):
        """
        Args:
            root_dir: Path to MUSDB18 dataset
            split: 'train' or 'test'
            segment_length: Length of audio segments in seconds
            sample_rate: Audio sample rate
            use_augmentation: Whether to use data augmentation
            augmentation_config: Dict with augmentation settings
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.use_augmentation = use_augmentation
        self.augmentation_config = augmentation_config or {}
        
        # Find all songs
        split_dir = self.root_dir / split
        self.songs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        print(f"Found {len(self.songs)} songs in {split} split")
    
    def __len__(self):
        # Each epoch will sample different segments
        return len(self.songs) * 100 if self.split == 'train' else len(self.songs)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            # Random song selection for training
            song_idx = idx % len(self.songs)
            return self._load_training_sample(song_idx)
        else:
            # Fixed song for validation/test
            return self._load_test_sample(idx)
    
    def _load_training_sample(self, song_idx):
        """Load a random segment from a song with augmentation."""
        song_dir = self.songs[song_idx]
        
        # Load stems
        vocals, sr = torchaudio.load(song_dir / 'vocals.wav')
        bass, _ = torchaudio.load(song_dir / 'bass.wav')
        drums, _ = torchaudio.load(song_dir / 'drums.wav')
        other, _ = torchaudio.load(song_dir / 'other.wav')
        
        assert sr == self.sample_rate, f"Sample rate mismatch: {sr} vs {self.sample_rate}"
        
        # Convert to mono if stereo (average channels)
        if vocals.size(0) == 2:
            vocals = vocals.mean(dim=0, keepdim=True)
            bass = bass.mean(dim=0, keepdim=True)
            drums = drums.mean(dim=0, keepdim=True)
            other = other.mean(dim=0, keepdim=True)
        
        # Random segment extraction with energy thresholding
        max_start = max(0, vocals.size(1) - self.segment_samples)
        
        # Try to find a segment with vocals
        for _ in range(10):
            start = random.randint(0, max_start) if max_start > 0 else 0
            end = start + self.segment_samples
            
            vocal_seg = vocals[:, start:end]
            
            # Check if vocals have sufficient energy
            if vocal_seg.abs().mean() > 0.01:
                break
        
        # Pad if necessary
        if vocal_seg.size(1) < self.segment_samples:
            pad_length = self.segment_samples - vocal_seg.size(1)
            vocal_seg = torch.nn.functional.pad(vocal_seg, (0, pad_length))
        
        # Extract same segment from other stems
        bass_seg = bass[:, start:end]
        drums_seg = drums[:, start:end]
        other_seg = other[:, start:end]
        
        if bass_seg.size(1) < self.segment_samples:
            pad_length = self.segment_samples - bass_seg.size(1)
            bass_seg = torch.nn.functional.pad(bass_seg, (0, pad_length))
            drums_seg = torch.nn.functional.pad(drums_seg, (0, pad_length))
            other_seg = torch.nn.functional.pad(other_seg, (0, pad_length))
        
        # Apply augmentation if enabled
        if self.use_augmentation:
            from .augmentation import apply_augmentation
            vocal_seg, bass_seg, drums_seg, other_seg = apply_augmentation(
                vocal_seg, bass_seg, drums_seg, other_seg, self.augmentation_config
            )
        
        # Random gain scaling [-10, 10] dB
        for stem in [vocal_seg, bass_seg, drums_seg, other_seg]:
            gain_db = random.uniform(-10, 10)
            gain = 10 ** (gain_db / 20)
            stem *= gain
        
        # Mix
        mixture = vocal_seg + bass_seg + drums_seg + other_seg
        
        # Normalize to prevent clipping
        max_val = max(mixture.abs().max(), vocal_seg.abs().max())
        if max_val > 0.95:
            scale = 0.95 / max_val
            mixture *= scale
            vocal_seg *= scale
        
        return mixture.squeeze(0), vocal_seg.squeeze(0)
    
    def _load_test_sample(self, idx):
        """Load full song for testing."""
        song_dir = self.songs[idx]
        
        vocals, sr = torchaudio.load(song_dir / 'vocals.wav')
        mixture, _ = torchaudio.load(song_dir / 'mixture.wav')
        
        # Convert to mono
        if vocals.size(0) == 2:
            vocals = vocals.mean(dim=0, keepdim=True)
            mixture = mixture.mean(dim=0, keepdim=True)
        
        return mixture.squeeze(0), vocals.squeeze(0)


def collate_fn(batch):
    """Collate function for DataLoader."""
    mixtures, targets = zip(*batch)
    
    # Stack
    mixtures = torch.stack([m for m in mixtures], dim=0)
    targets = torch.stack([t for t in targets], dim=0)
    
    return mixtures, targets