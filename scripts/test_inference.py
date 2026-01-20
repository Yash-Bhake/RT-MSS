#!/usr/bin/env python3
"""Simple test inference on a single audio file."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torchaudio
import argparse
from pathlib import Path
import time

from src.model.causal_bsrnn import CausalBSRNN
from src.utils.metrics import si_snr, sdr


def separate_audio(model, audio, device, chunk_size=256):
    """
    Separate vocals from mixture.
    
    Args:
        model: CausalBSRNN model
        audio: (1, T) audio tensor
        device: torch device
        chunk_size: processing chunk size in frames
    
    Returns:
        separated: (1, T) separated audio
    """
    model.eval()
    
    # Compute STFT
    spec = torch.stft(
        audio,
        n_fft=model.n_fft,
        hop_length=model.hop_length,
        window=model.window.to(device),
        center=False,
        return_complex=True
    )
    
    # Convert to (B, F, T, 2)
    spec = spec.permute(0, 1, 3, 2)
    
    print(f"Spectrogram shape: {spec.shape}")
    
    # Process in chunks with state management
    num_frames = spec.size(2)
    output_spec = torch.zeros_like(spec)
    states = None
    
    num_chunks = (num_frames + chunk_size - 1) // chunk_size
    print(f"Processing {num_chunks} chunks...")
    
    with torch.no_grad():
        for i in range(0, num_frames, chunk_size):
            end = min(i + chunk_size, num_frames)
            chunk_spec = spec[:, :, i:end, :]
            
            # Pad if needed
            actual_size = chunk_spec.size(2)
            if actual_size < chunk_size:
                pad_len = chunk_size - actual_size
                chunk_spec = torch.nn.functional.pad(chunk_spec, (0, 0, 0, pad_len))
            
            # Process
            separated_chunk, states = model(chunk_spec, states)
            
            # Remove padding
            separated_chunk = separated_chunk[:, :, :actual_size, :]
            
            # Store
            output_spec[:, :, i:end, :] = separated_chunk
            
            if (i // chunk_size + 1) % 10 == 0:
                print(f"  Processed {i // chunk_size + 1}/{num_chunks} chunks")
    
    # Convert back to audio
    output_spec = output_spec.permute(0, 1, 3, 2)
    spec_complex = torch.view_as_complex(output_spec.contiguous())
    
    separated = torch.istft(
        spec_complex,
        n_fft=model.n_fft,
        hop_length=model.hop_length,
        window=model.window.to(device),
        center=False
    )
    
    return separated


def main():
    parser = argparse.ArgumentParser(description='Test inference on single file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input mixture audio file')
    parser.add_argument('--output', type=str, default='separated_vocals.wav',
                        help='Output separated vocals file')
    parser.add_argument('--reference', type=str, default=None,
                        help='Reference vocals for metric calculation (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--chunk_size', type=int, default=256,
                        help='Processing chunk size in frames')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    model = CausalBSRNN(
        n_fft=model_config.get('n_fft', 2048),
        hop_length=model_config.get('hop_length', 512),
        feature_dim=model_config.get('feature_dim', 128),
        num_repeat=model_config.get('num_repeat', 12),
        hidden_dim=model_config.get('hidden_dim', 256)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {params:.2f}M parameters\n")
    
    # Load audio
    print(f"Loading audio: {args.input}")
    mixture, sr = torchaudio.load(args.input)
    
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {mixture.size(1) / sr:.2f} seconds")
    print(f"Channels: {mixture.size(0)}")
    
    # Convert to mono if stereo
    if mixture.size(0) == 2:
        mixture = mixture.mean(dim=0, keepdim=True)
        print("Converted to mono")
    
    # Resample if needed
    if sr != 44100:
        print(f"Resampling from {sr} Hz to 44100 Hz...")
        resampler = torchaudio.transforms.Resample(sr, 44100)
        mixture = resampler(mixture)
        sr = 44100
    
    mixture = mixture.to(device)
    print(f"Input shape: {mixture.shape}\n")
    
    # Separate
    print("Separating vocals...")
    start_time = time.time()
    
    separated = separate_audio(model, mixture, device, args.chunk_size)
    
    processing_time = time.time() - start_time
    audio_length = mixture.size(1) / sr
    rtf = processing_time / audio_length
    
    print(f"\n✓ Separation complete!")
    print(f"  Processing time: {processing_time:.2f} seconds")
    print(f"  Audio length: {audio_length:.2f} seconds")
    print(f"  RTF: {rtf:.4f}")
    print(f"  Speed: {1/rtf:.2f}x realtime\n")
    
    # Trim to input length
    separated = separated[:, :mixture.size(1)]
    
    # Calculate metrics if reference provided
    if args.reference:
        print("Calculating metrics...")
        reference, ref_sr = torchaudio.load(args.reference)
        
        if reference.size(0) == 2:
            reference = reference.mean(dim=0, keepdim=True)
        
        if ref_sr != 44100:
            resampler = torchaudio.transforms.Resample(ref_sr, 44100)
            reference = resampler(reference)
        
        reference = reference.to(device)
        
        # Match lengths
        min_len = min(separated.size(1), reference.size(1))
        separated_crop = separated[:, :min_len]
        reference_crop = reference[:, :min_len]
        
        si_snr_val = si_snr(separated_crop, reference_crop)
        sdr_val = sdr(separated_crop, reference_crop)
        
        print(f"  SI-SNR: {si_snr_val:.3f} dB")
        print(f"  SDR: {sdr_val:.3f} dB\n")
    
    # Save output
    print(f"Saving output to: {args.output}")
    separated_cpu = separated.cpu()
    torchaudio.save(args.output, separated_cpu, 44100)
    
    print("✓ Done!\n")


if __name__ == '__main__':
    main()