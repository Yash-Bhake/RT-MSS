#!/usr/bin/env python3
"""Evaluation script for Causal BSRNN."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torchaudio
import argparse
import time
from pathlib import Path
from tqdm import tqdm

from src.model.causal_bsrnn import CausalBSRNN
from src.utils.metrics import MetricsCalculator


def evaluate_model(model, test_dir, device, chunk_size=44100):
    """
    Evaluate model on test set.
    
    Args:
        model: CausalBSRNN model
        test_dir: Path to test directory
        device: torch device
        chunk_size: Audio chunk size for processing
    
    Returns:
        metrics: Dictionary of average metrics
    """
    model.eval()
    metrics_calc = MetricsCalculator()
    
    test_songs = sorted([d for d in Path(test_dir).iterdir() if d.is_dir()])
    
    print(f"Evaluating on {len(test_songs)} songs...")
    
    with torch.no_grad():
        for song_dir in tqdm(test_songs):
            # Load audio
            mixture, sr = torchaudio.load(song_dir / 'mixture.wav')
            vocals, _ = torchaudio.load(song_dir / 'vocals.wav')
            
            # Convert to mono
            if mixture.size(0) == 2:
                mixture = mixture.mean(dim=0, keepdim=True)
                vocals = vocals.mean(dim=0, keepdim=True)
            
            # Move to device
            mixture = mixture.to(device)
            vocals = vocals.to(device)
            
            # Process
            start_time = time.time()
            separated = process_audio(model, mixture, chunk_size, device)
            processing_time = time.time() - start_time
            
            # Trim to target length
            separated = separated[..., :vocals.size(-1)]
            
            # Compute metrics
            audio_length = vocals.size(-1) / sr
            metrics_calc.update(separated, vocals, processing_time, audio_length)
    
    return metrics_calc.get_average()


def process_audio(model, audio, chunk_size, device):
    """Process audio in chunks with state management."""
    n_fft = model.n_fft
    hop_length = model.hop_length
    
    # Compute STFT
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=model.window.to(device),
        center=False,
        return_complex=False,
        return_complex=True
    )
    
    # Convert to (B, F, T, 2)
    spec = spec.permute(0, 1, 3, 2)
    
    # Process in chunks
    num_frames = spec.size(2)
    chunk_hop = chunk_size // 2
    
    output_spec = torch.zeros_like(spec)
    states = None
    
    for i in range(0, num_frames, chunk_hop):
        end = min(i + chunk_size, num_frames)
        chunk_spec = spec[:, :, i:end, :]
        
        # Pad if needed
        if chunk_spec.size(2) < chunk_size:
            pad_len = chunk_size - chunk_spec.size(2)
            chunk_spec = torch.nn.functional.pad(chunk_spec, (0, 0, 0, pad_len))
        
        # Process
        separated_chunk, states = model(chunk_spec, states)
        
        # Trim padding
        separated_chunk = separated_chunk[:, :, :end-i, :]
        
        # Overlap-add with windowing
        if i > 0:
            fade_len = min(chunk_hop, separated_chunk.size(2))
            fade = torch.linspace(0, 1, fade_len, device=device)
            
            # Crossfade
            output_spec[:, :, i:i+fade_len, :] *= (1 - fade).view(1, 1, -1, 1)
            separated_chunk[:, :, :fade_len, :] *= fade.view(1, 1, -1, 1)
        
        output_spec[:, :, i:end, :] += separated_chunk
    
    # Convert back to audio
    output_spec = output_spec.permute(0, 1, 3, 2)
    spec_complex = torch.view_as_complex(output_spec.contiguous())
    
    separated_audio = torch.istft(
        spec_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        window=model.window.to(device),
        center=False
    )
    
    return separated_audio


def main():
    parser = argparse.ArgumentParser(description='Evaluate Causal BSRNN')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--chunk_size', type=int, default=256,
                        help='Chunk size for processing (in frames)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    model = CausalBSRNN(
        n_fft=model_config.get('n_fft', 1024),
        hop_length=model_config.get('hop_length', 512),
        feature_dim=model_config.get('feature_dim', 128),
        num_repeat=model_config.get('num_repeat', 12),
        hidden_dim=model_config.get('hidden_dim', 256)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Evaluate
    metrics = evaluate_model(model, args.test_dir, device, args.chunk_size)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    for metric_name, value in metrics.items():
        if metric_name == 'RTF':
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {value:.3f} dB")
    print("="*50)


if __name__ == '__main__':
    main()