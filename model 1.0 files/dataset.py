import os
import random
import math
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from .augmentation import apply_augmentation_pipeline
import warnings
warnings.filterwarnings("ignore")

class MUSDB18Dataset(Dataset):
    def __init__(self, root_dir, split='train', config=None, total_epochs=50):
        """
        Args:
            root_dir: Dataset root
            split: 'train' or 'test'
            config: Full config dictionary
            total_epochs: For augmentation scheduling
        """
        self.root_dir = Path(root_dir) / split
        self.split = split
        
        self.config = config['data']
        self.aug_config = config['data'] 

        self.sr = self.config['sample_rate']
        self.seg_len_samples = int(self.config['segment_length'] * self.sr)
        self.overlap = self.config.get('overlap_percentage', 0.15)
        self.use_same_data = self.config.get('use_same_data', False)
        
        # Augmentation State
        self.current_epoch = 0
        self.total_epochs = total_epochs
        
        # 1. Load Songs List
        self.songs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        if not self.songs:
            raise RuntimeError(f"No songs found in {self.root_dir}")
            
        # 2. Setup Cache Folder
        # We create a dedicated folder for cache and debug samples
        vad_thresh = self.config.get('vocal_threshold', 0.005)
        cache_folder_name = f"cache_sr{self.sr}_len{self.config['segment_length']}_ov{self.overlap}_vad{vad_thresh}"
        self.cache_dir = self.root_dir / cache_folder_name
        self.cache_dir.mkdir(exist_ok=True)
        
        self.cache_path = self.cache_dir / "dataset_registry.pt"
        self.segment_registry = [] # List of tuples: (song_index, start_sample)

        # 3. Generate or Load Segments
        if self.use_same_data and self.cache_path.exists():
            print(f"Loading cached segments from {self.cache_path}...")
            self.segment_registry = torch.load(self.cache_path)
        else:
            print(f"Generating segments (Sliding Window, Overlap: {self.overlap})...")
            self._generate_segments_sliding_window()
            
            # Randomize the order to ensure no dependence between adjacent chunks
            if self.split == 'train':
                print("Shuffling segments for independence...")
                random.shuffle(self.segment_registry)
            
            if self.use_same_data:
                print(f"Saving segments to {self.cache_path}...")
                torch.save(self.segment_registry, self.cache_path)

        # 4. Save Initial Debug Samples
        if self.split == 'train':
            self._save_debug_samples("init")

        self._print_stats()

    def set_epoch(self, epoch):
        """
        Update epoch for augmentation scheduling and save debug samples.
        """
        self.current_epoch = epoch
        # Save debug samples to verify dynamic mixing for this epoch
        if self.split == 'train':
            self._save_debug_samples(f"epoch_{epoch}")

    def _save_debug_samples(self, prefix):
        """Save 3 random samples to the cache folder to verify data/augmentation."""
        try:
            indices = random.sample(range(len(self)), min(3, len(self)))
            print(f"Saving debug samples for '{prefix}' to {self.cache_dir}...")
            
            for i, idx in enumerate(indices):
                mix, target = self.__getitem__(idx)
                # Save mixture
                out_path = self.cache_dir / f"debug_{prefix}_sample_{i}.wav"
                torchaudio.save(out_path, mix, self.sr)
        except Exception as e:
            print(f"Warning: Could not save debug samples: {e}")

    def _generate_segments_sliding_window(self):
        """
        Strict Sliding Window Strategy:
        1. Stride = Segment_Length * (1 - Overlap)
        2. Check VAD for every window.
        3. Keep if vocal activity > threshold.
        """
        vocal_threshold = self.config.get('vocal_threshold', 0.005)
        stride = int(self.seg_len_samples * (1 - self.overlap))
        
        for song_idx, song_path in enumerate(tqdm(self.songs, desc="Scanning Songs")):
            vocals_path = song_path / 'vocals.wav'
            if not vocals_path.exists(): continue
            
            try:
                # Load full vocals to memory for fast VAD check
                # (Reading once is faster than seeking 1000 times)
                info = sf.info(str(vocals_path))
                y_voc, _ = sf.read(str(vocals_path), dtype='float32', always_2d=True)
                y_voc = torch.from_numpy(y_voc.T).mean(dim=0) # Mono
                
                total_samples = y_voc.size(0)
                
                # Slide window
                # We stop when start + seg_len > total
                for start in range(0, total_samples - self.seg_len_samples + 1, stride):
                    chunk = y_voc[start : start + self.seg_len_samples]
                    
                    # VAD Check (RMS)
                    rms = torch.sqrt(torch.mean(chunk**2))
                    
                    if rms > vocal_threshold:
                        self.segment_registry.append((song_idx, start))
                        
            except Exception as e:
                print(f"Error processing {song_path.name}: {e}")
                continue

    def _print_stats(self):
        num_segments = len(self.segment_registry)
        num_songs = len(self.songs)
        
        if num_songs == 0:
            print("WARNING: No songs loaded!")
            return

        total_duration_sec = num_segments * self.config['segment_length']
        total_hours = total_duration_sec / 3600.0
        
        print("\n" + "="*40)
        print("  DATASET STATISTICS (Sliding Window)")
        print("="*40)
        print(f"  Total Songs:       {num_songs}")
        print(f"  Total Segments:    {num_segments}")
        print(f"  Avg Segments/Song: {num_segments / num_songs:.1f}")
        print(f"  Total Duration:    {total_hours:.2f} hours")
        print(f"  Overlap:           {self.overlap*100}%")
        print(f"  Cache Location:    {self.cache_dir}")
        print("="*40 + "\n")

    def __len__(self):
        return len(self.segment_registry)

    def _load_audio_chunk(self, path, start, samples):
        """Helper to read specific chunk."""
        y, _ = sf.read(str(path), start=start, stop=start+samples, dtype='float32', always_2d=True)
        return torch.from_numpy(y.T) # (C, T)

    def __getitem__(self, idx):
        # 1. Retrieve Metadata
        song_idx, start_sample = self.segment_registry[idx]
        song_path = self.songs[song_idx]
        
        try:
            # Load chunks (Mono conversion happens here)
            v_chunk = self._load_audio_chunk(song_path / 'vocals.wav', start_sample, self.seg_len_samples).mean(dim=0, keepdim=True)
            b_chunk = self._load_audio_chunk(song_path / 'bass.wav', start_sample, self.seg_len_samples).mean(dim=0, keepdim=True)
            d_chunk = self._load_audio_chunk(song_path / 'drums.wav', start_sample, self.seg_len_samples).mean(dim=0, keepdim=True)
            o_chunk = self._load_audio_chunk(song_path / 'other.wav', start_sample, self.seg_len_samples).mean(dim=0, keepdim=True)
            
            # 3. Apply Augmentation (Training Only)
            if self.split == 'train' and self.config.get('use_augmentation', False):
                mixture, target = apply_augmentation_pipeline(
                    v_chunk, b_chunk, d_chunk, o_chunk,
                    self.sr, 
                    self.aug_config, 
                    self.current_epoch,
                    self.total_epochs
                )
            else:
                # Validation / No Augmentation
                mixture = v_chunk + b_chunk + d_chunk + o_chunk
                target = v_chunk
            
            return mixture, target

        except Exception as e:
            print(f"Error loading segment {idx}: {e}")
            silent = torch.zeros(1, self.seg_len_samples)
            return silent, silent

def collate_fn(batch):
    mixtures, targets = zip(*batch)
    mixtures = torch.stack(mixtures).squeeze(1) 
    targets = torch.stack(targets).squeeze(1)
    return mixtures, targets