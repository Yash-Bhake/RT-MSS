#!/usr/bin/env python3
"""
Updated Inference Script for Causal BSRNN
- Fixes 'window overlap add min: 1' error using Clamped Window Trick
- Maintains strict causality (center=False)
- Handles chunked processing and OLA reconstruction
"""

import sys
import os
import torch
import torchaudio
import argparse
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model.causal_bsrnn import CausalBSRNN

def calculate_metrics(separated, reference):
    """
    Calculate SI-SNR and SDR metrics.
    """
    def to_numpy(x): return x.detach().cpu().numpy().flatten()
    s = to_numpy(separated)
    r = to_numpy(reference)
    
    # Ensure same length
    length = min(len(s), len(r))
    s, r = s[:length], r[:length]
    
    # SI-SNR
    target = np.dot(s, r) * r / (np.linalg.norm(r)**2 + 1e-8)
    noise = s - target
    si_snr = 10 * np.log10(np.linalg.norm(target)**2 / (np.linalg.norm(noise)**2 + 1e-8))
    
    # SDR (Signal-to-Distortion Ratio)
    sdr = 10 * np.log10(np.linalg.norm(r)**2 / (np.linalg.norm(r - s)**2 + 1e-8))
    
    return si_snr, sdr

def separate_audio(model, audio, device, chunk_size=256):
    """
    Separate vocals using causal stateful processing.
    """
    model.eval()
    audio = audio.to(device)
    T_orig = audio.size(-1)
    
    hop = model.hop_length
    n_fft = model.n_fft
    
    # --- 1. Robust Padding ---
    # Left Pad: Pushes the start of the signal into the non-zero region of the window.
    left_pad = n_fft  
    
    remainder = (T_orig + left_pad) % hop
    if remainder == 0:
        right_pad = 0
    else:
        right_pad = hop - remainder
    
    # Add extra safety padding for the right side
    right_pad += n_fft 
    
    audio_padded = torch.nn.functional.pad(audio, (left_pad, right_pad))

    with torch.no_grad():
        # 2. Compute STFT (center=False for causality)
        window = model.window.to(device)
        spec = torch.stft(
            audio_padded,
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            center=False,
            normalized=True,
            return_complex=True
        )
        spec = torch.view_as_real(spec).contiguous() 
        
        num_frames = spec.size(2)
        output_spec = torch.zeros_like(spec)
        
        # 3. Initialize states for streaming
        B = spec.size(0)
        states = model._init_states(B, device)

        # 4. Chunked Processing
        for i in tqdm(range(0, num_frames, chunk_size), desc="Separating"):
            end = min(i + chunk_size, num_frames)
            chunk_spec = spec[:, :, i:end, :]

            # Handle last chunk padding if needed
            actual_size = chunk_spec.size(2)
            if actual_size < chunk_size:
                pad_len = chunk_size - actual_size
                chunk_spec = torch.nn.functional.pad(chunk_spec, (0, 0, 0, pad_len))

            # Streaming forward pass
            separated_chunk, states = model(chunk_spec, states=states)

            # Trim padding if it was the last chunk
            separated_chunk = separated_chunk[:, :, :actual_size, :]
            output_spec[:, :, i:end, :] = separated_chunk

        # 5. Inverse STFT with "Clamped Window" Trick
        spec_complex = torch.view_as_complex(output_spec.contiguous())

        # FIX: Create a temporary window that forces non-zero edges.
        # This satisfies the NOLA check at index 0 and index -1.
        rec_window = window.clone()
        epsilon = 1e-4
        rec_window[0] = epsilon
        rec_window[-1] = epsilon

        separated_padded = torch.istft(
            spec_complex,
            n_fft=n_fft,
            hop_length=hop,
            window=rec_window, # Use the modified window
            center=False,
            normalized=True
        )
        
        # 6. Crop back to original length
        # We discard the first 'left_pad' samples (which contain the garbage from our epsilon hack)
        if separated_padded.size(-1) >= left_pad + T_orig:
            separated = separated_padded[..., left_pad : left_pad + T_orig]
        else:
            # Fallback (unlikely)
            separated = torch.nn.functional.pad(separated_padded, (0, T_orig))
            separated = separated[..., :T_orig]

    return separated

def main():
    parser = argparse.ArgumentParser(description='Causal Inference and Metrics')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input mixture wav')
    parser.add_argument('--output', type=str, default='vocals_pred.wav', help='Path to save output')
    parser.add_argument('--chunk_size', type=int, default=256, help='STFT frames per chunk')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Handle device selection
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")

    # Load checkpoint
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint {args.checkpoint} not found")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    m_conf = checkpoint['config']['model']

    # Initialize Model
    model = CausalBSRNN(
        n_fft=m_conf['n_fft'],
        hop_length=m_conf['hop_length'],
        feature_dim=m_conf['feature_dim'],
        num_repeat=m_conf['num_repeat'],
        hidden_dim=m_conf['hidden_dim']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load mixture audio
    try:
        mixture, sr = torchaudio.load(args.input)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    # Preprocessing (Mono + Resample)
    if mixture.size(0) > 1: 
        mixture = mixture.mean(0, keepdim=True)
    if sr != 44100: 
        mixture = torchaudio.transforms.Resample(sr, 44100)(mixture)

    # Run Separation
    print(f"Separating {args.input}...")
    start = time.time()
    separated = separate_audio(model, mixture, device, args.chunk_size)
    dt = time.time() - start
    
    duration = mixture.size(1) / 44100
    print(f"Separation finished in {dt:.2f}s (RTF: {dt/duration:.3f})")
    
    # Save output
    torchaudio.save(args.output, separated.cpu(), 44100)
    print(f"Saved prediction to {args.output}")

    # Calculate Metrics (if ground truth exists)
    ref_path = Path(args.input).parent / 'vocals.wav'
    if ref_path.exists():
        try:
            reference, rsr = torchaudio.load(ref_path)
            if reference.size(0) > 1: reference = reference.mean(0, keepdim=True)
            if rsr != 44100: reference = torchaudio.transforms.Resample(rsr, 44100)(reference)
            
            # Move to same device for metric calc
            si_snr, sdr = calculate_metrics(separated, reference.to(separated.device))
            
            print(f"\n--- Metrics (Reference: {ref_path.name}) ---")
            print(f"SI-SNR: {si_snr:.2f} dB")
            print(f"SDR:    {sdr:.2f} dB")
        except Exception as e:
            print(f"Error calculating metrics: {e}")
    else:
        print(f"\nReference 'vocals.wav' not found at {ref_path}. Skipping metrics.")

if __name__ == '__main__':
    main()