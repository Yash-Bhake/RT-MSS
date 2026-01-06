"""
Dataset for MUSDB18 with augmentation
"""

import torch
import torchaudio
from torch.utils.data import Dataset
import musdb
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .augmentation import (
    MixAugmentation, RandomChunkSelector, 
    SourceActivityDetection, EnergyNormalization
)


class MUSDB18Dataset(Dataset):
    """
    MUSDB18 dataset with on-the-fly mixing and augmentation
    """
    
    def __init__(self, root: str, subset: str = 'train', 
                 target: str = 'vocals', config: dict = None,
                 segment_length: float = 3.0, sample_rate: int = 44100):
        """
        Args:
            root: Path to MUSDB18 directory
            subset: 'train' or 'test'
            target: Target source ('vocals', 'drums', 'bass', 'other')
            config: Configuration dictionary
            segment_length: Length of training segments in seconds
            sample_rate: Sample rate (MUSDB18 is 44.1kHz)
        """
        self.root = Path(root)
        self.subset = subset
        self.target = target
        self.config = config
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        
        # Load MUSDB18
        self.musdb = musdb.DB(root=str(self.root), subsets=[subset])
        self.tracks = self.musdb.tracks
        
        print(f"Loaded {len(self.tracks)} tracks from MUSDB18 {subset} set")
        
        # Source activity detector for preprocessing
        self.sad = SourceActivityDetection()
        
        # Augmentation
        if config and subset == 'train':
            aug_config = config.get('training', {}).get('augmentation', {})
            self.augmentation = MixAugmentation(aug_config, sample_rate)
            self.chunk_selector = RandomChunkSelector(segment_length, sample_rate)
            self.energy_norm = EnergyNormalization()
        else:
            self.augmentation = None
            self.chunk_selector = None
            self.energy_norm = None
        
        # Preprocess tracks to extract salient segments
        if subset == 'train':
            self.segments = self._extract_salient_segments()
            print(f"Extracted {len(self.segments)} salient segments")
        else:
            self.segments = [(i, None) for i in range(len(self.tracks))]
    
    def _extract_salient_segments(self) -> List[Tuple[int, int]]:
        """
        Extract salient (non-silent) segments from tracks
        Returns list of (track_idx, segment_start_sample)
        """
        segments = []
        segment_samples = int(self.segment_length * self.sample_rate)
        overlap_ratio = 0.5
        hop_samples = int(segment_samples * (1 - overlap_ratio))
        
        for track_idx, track in enumerate(self.tracks):
            # Load target stem
            target_audio = track.targets[self.target].audio.T  # (2, n_samples)
            target_tensor = torch.from_numpy(target_audio).float()
            
            n_samples = target_tensor.size(1)
            
            # Extract segments
            for start in range(0, n_samples - segment_samples, hop_samples):
                segment = target_tensor[:, start:start + segment_samples]
                
                # Check if segment is active
                if self.sad.is_active(segment, self.segment_length, self.sample_rate):
                    segments.append((track_idx, start))
        
        return segments
    
    def __len__(self) -> int:
        if self.subset == 'train':
            # For training, return large number for epoch-based training
            return 10000
        else:
            return len(self.tracks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get mixture and target
        
        Returns:
            mixture: (n_channels, n_samples) stereo mixture
            target: (n_channels, n_samples) target source
        """
        if self.subset == 'train':
            return self._get_train_item()
        else:
            return self._get_test_item(idx)
    
    def _get_train_item(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get training sample with augmentation"""
        
        # Randomly select a salient segment
        track_idx, seg_start = random.choice(self.segments)
        track = self.tracks[track_idx]
        
        segment_samples = int(self.segment_length * self.sample_rate)
        
        # Load all stems for this segment
        stems = {}
        for stem_name in ['vocals', 'drums', 'bass', 'other']:
            stem_audio = track.targets[stem_name].audio.T  # (2, n_samples)
            stem_tensor = torch.from_numpy(stem_audio).float()
            
            # Extract segment
            if seg_start is not None:
                stem_segment = stem_tensor[:, seg_start:seg_start + segment_samples]
            else:
                # Random chunk if no specific start
                stem_segment = self.chunk_selector(stem_tensor)
            
            stems[stem_name] = stem_segment
        
        # Mark target
        stems['target'] = stems[self.target]
        
        # Apply augmentation with mixing
        if self.augmentation is not None:
            # Randomly decide whether to remix stems from different tracks
            if random.random() < 0.5:
                # Remix: replace some stems with stems from other tracks
                other_track_idx = random.choice(range(len(self.tracks)))
                if other_track_idx != track_idx:
                    other_track = self.tracks[other_track_idx]
                    
                    # Randomly choose which stems to replace
                    stems_to_replace = random.sample(
                        ['vocals', 'drums', 'bass', 'other'],
                        k=random.randint(1, 3)
                    )
                    
                    for stem_name in stems_to_replace:
                        if stem_name != self.target:  # Don't replace target
                            other_audio = other_track.targets[stem_name].audio.T
                            other_tensor = torch.from_numpy(other_audio).float()
                            stems[stem_name] = self.chunk_selector(other_tensor)
            
            # Apply pitch shift and gain augmentation
            mixture, target = self.augmentation(stems, apply_augmentation=True)
        else:
            # No augmentation
            target = stems[self.target]
            mixture = sum([stems[s] for s in ['vocals', 'drums', 'bass', 'other']])
        
        # Energy normalization
        if self.energy_norm is not None:
            mixture, target = self.energy_norm(mixture, target)
        
        return mixture, target
    
    def _get_test_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get test sample (full track or long segment)"""
        track = self.tracks[idx]
        
        # Load mixture
        mixture_audio = track.audio.T  # (2, n_samples)
        mixture = torch.from_numpy(mixture_audio).float()
        
        # Load target
        target_audio = track.targets[self.target].audio.T
        target = torch.from_numpy(target_audio).float()
        
        # For testing, we might want to use full tracks or long segments
        # Here we'll use 10-second segments for evaluation
        if self.config and 'evaluation' in self.config:
            test_length = self.config['evaluation'].get('test_segment_length', 10.0)
            test_samples = int(test_length * self.sample_rate)
            
            if mixture.size(1) > test_samples:
                # Take first segment
                mixture = mixture[:, :test_samples]
                target = target[:, :test_samples]
        
        return mixture, target


class MUSDB18Collator:
    """
    Custom collator for MUSDB18 dataset
    Handles variable length sequences and STFT computation
    """
    
    def __init__(self, n_fft: int = 1408, hop_length: int = 704,
                 sample_rate: int = 44100):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.window = torch.hann_window(n_fft)
    
    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate batch and compute STFT
        
        Args:
            batch: List of (mixture, target) pairs
        Returns:
            mixture_specs: (B, 2, F, T) complex spectrograms
            target_specs: (B, 2, F, T) complex spectrograms
        """
        mixtures, targets = zip(*batch)
        
        # Stack
        mixtures = torch.stack(mixtures, dim=0)  # (B, 2, n_samples)
        targets = torch.stack(targets, dim=0)    # (B, 2, n_samples)
        
        # Compute STFT (left-aligned for causality)
        mixture_specs = []
        target_specs = []
        
        for i in range(mixtures.size(0)):
            # Compute STFT for each channel
            mix_spec = torch.stft(
                mixtures[i],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=False,  # Left-aligned for causality
                return_complex=True
            )
            
            tgt_spec = torch.stft(
                targets[i],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=False,
                return_complex=True
            )
            
            mixture_specs.append(mix_spec)
            target_specs.append(tgt_spec)
        
        mixture_specs = torch.stack(mixture_specs, dim=0)  # (B, 2, F, T)
        target_specs = torch.stack(target_specs, dim=0)    # (B, 2, F, T)
        
        return mixture_specs, target_specs