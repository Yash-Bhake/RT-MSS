"""
Evaluation script with SI-SNR, SDR, and RTF metrics
"""

import torch
import torchaudio
import numpy as np
import argparse
import yaml
import time
from pathlib import Path
from tqdm import tqdm
import musdb
import museval

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model.causal_bsrnn import MultiTargetCausalBSRNN
from src.inference.realtime_processor import RealtimeProcessor


def si_snr(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute Scale-Invariant Signal-to-Noise Ratio
    
    Args:
        pred: Predicted audio (n_samples,) or (n_channels, n_samples)
        target: Target audio
    Returns:
        SI-SNR in dB
    """
    # Ensure same shape
    pred = pred.flatten()
    target = target.flatten()
    
    # Zero-mean
    target = target - np.mean(target)
    pred = pred - np.mean(pred)
    
    # Scaling factor
    s_target = np.sum(pred * target) / (np.sum(target ** 2) + eps)
    s_target = s_target * target
    
    # SI-SNR
    si_snr_value = 10 * np.log10(
        (np.sum(s_target ** 2) + eps) / (np.sum((pred - s_target) ** 2) + eps)
    )
    
    return si_snr_value


def compute_sdr(pred: np.ndarray, target: np.ndarray, 
                sample_rate: int = 44100) -> dict:
    """
    Compute SDR using museval
    
    Args:
        pred: Predicted audio (n_channels, n_samples)
        target: Target audio (n_channels, n_samples)
        sample_rate: Sample rate
    Returns:
        Dictionary with SDR metrics
    """
    # Transpose for museval (expects (n_samples, n_channels))
    pred = pred.T
    target = target.T
    
    # Compute SDR
    sdr, isr, sir, sar = museval.evaluate(target, pred, win=sample_rate, hop=sample_rate)
    
    # Return median values
    return {
        'sdr': np.median(sdr),
        'isr': np.median(isr),
        'sir': np.median(sir),
        'sar': np.median(sar)
    }


def evaluate_track(model: MultiTargetCausalBSRNN, track: musdb.Track,
                   target: str, config: dict, device: str = 'cpu') -> dict:
    """
    Evaluate model on a single track
    
    Args:
        model: Causal BSRNN model
        track: MUSDB track
        target: Target source
        config: Configuration
        device: Device for inference
    Returns:
        Dictionary with metrics
    """
    # Load audio
    mixture = torch.from_numpy(track.audio.T).float()  # (2, n_samples)
    target_audio = torch.from_numpy(track.targets[target].audio.T).float()
    
    # Create processor
    processor = RealtimeProcessor(model, config, target, device)
    
    # Process
    start_time = time.time()
    separated = processor.process_stream(mixture)
    processing_time = time.time() - start_time
    
    # Compute RTF
    audio_duration = mixture.size(1) / config['audio']['sample_rate']
    rtf = processing_time / audio_duration
    
    # Convert to numpy
    separated_np = separated.numpy()
    target_np = target_audio.numpy()
    
    # Compute metrics
    si_snr_value = si_snr(separated_np, target_np)
    
    # Compute SDR (may take time)
    try:
        sdr_metrics = compute_sdr(separated_np, target_np, config['audio']['sample_rate'])
    except:
        sdr_metrics = {'sdr': 0.0, 'isr': 0.0, 'sir': 0.0, 'sar': 0.0}
    
    # Get processor statistics
    stats = processor.get_statistics()
    
    return {
        'si_snr': si_snr_value,
        'sdr': sdr_metrics['sdr'],
        'isr': sdr_metrics['isr'],
        'sir': sdr_metrics['sir'],
        'sar': sdr_metrics['sar'],
        'rtf': rtf,
        'processing_time_ms': processing_time * 1000,
        'mean_chunk_time_ms': stats.get('mean_processing_time_ms', 0),
        'max_chunk_time_ms': stats.get('max_processing_time_ms', 0),
        'latency_ms': processor.get_latency_ms()
    }


def evaluate(args):
    """Main evaluation function"""
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load MUSDB
    musdb_test = musdb.DB(root=args.data_path, subsets=['test'])
    
    # Load model
    print("Loading model...")
    model = MultiTargetCausalBSRNN(config)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    model.eval()
    
    # Evaluate each target
    targets = args.targets.split(',') if args.targets else ['vocals', 'drums', 'bass', 'other']
    
    results = {target: [] for target in targets}
    
    print(f"\nEvaluating on {len(musdb_test.tracks)} tracks...")
    
    for target in targets:
        print(f"\n{'='*50}")
        print(f"Target: {target}")
        print(f"{'='*50}")
        
        for track in tqdm(musdb_test.tracks, desc=f"Evaluating {target}"):
            metrics = evaluate_track(model, track, target, config, args.device)
            results[target].append(metrics)
            
            # Print track result
            print(f"\n{track.name}:")
            print(f"  SI-SNR: {metrics['si_snr']:.2f} dB")
            print(f"  SDR: {metrics['sdr']:.2f} dB")
            print(f"  RTF: {metrics['rtf']:.3f}")
            print(f"  Latency: {metrics['latency_ms']:.1f} ms")
            print(f"  Processing time: {metrics['processing_time_ms']:.1f} ms")
            print(f"  Mean chunk time: {metrics['mean_chunk_time_ms']:.1f} ms")
            print(f"  Max chunk time: {metrics['max_chunk_time_ms']:.1f} ms")
    
    # Compute averages
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}\n")
    
    for target in targets:
        metrics_list = results[target]
        
        avg_si_snr = np.mean([m['si_snr'] for m in metrics_list])
        avg_sdr = np.mean([m['sdr'] for m in metrics_list])
        avg_rtf = np.mean([m['rtf'] for m in metrics_list])
        avg_latency = np.mean([m['latency_ms'] for m in metrics_list])
        avg_chunk_time = np.mean([m['mean_chunk_time_ms'] for m in metrics_list])
        max_chunk_time = max([m['max_chunk_time_ms'] for m in metrics_list])
        
        print(f"{target.capitalize()}:")
        print(f"  Average SI-SNR: {avg_si_snr:.2f} dB")
        print(f"  Average SDR: {avg_sdr:.2f} dB")
        print(f"  Average RTF: {avg_rtf:.3f} {'(REAL-TIME!)' if avg_rtf < 1.0 else '(NOT REAL-TIME)'}")
        print(f"  Average Latency: {avg_latency:.1f} ms")
        print(f"  Average Chunk Processing: {avg_chunk_time:.1f} ms")
        print(f"  Max Chunk Processing: {max_chunk_time:.1f} ms")
        print(f"  Real-time capable: {'YES ✓' if avg_rtf < 1.0 and max_chunk_time < 16 else 'NO ✗'}")
        print()
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Causal BSRNN')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to MUSDB18 directory')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to config file')
    parser.add_argument('--targets', type=str, default='vocals,drums,bass,other',
                       help='Comma-separated list of targets to evaluate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for inference')
    parser.add_argument('--output', type=str, default='results/evaluation_results.json',
                       help='Path to save results')
    
    args = parser.parse_args()
    evaluate(args)