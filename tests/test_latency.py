"""
Latency testing and benchmarking for real-time performance
Tests that processing time < 16ms for real-time operation
"""

import torch
import time
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model.causal_bsrnn import MultiTargetCausalBSRNN
from src.inference.realtime_processor import RealtimeProcessor


def test_chunk_latency(config_path: str = 'config/model_config.yaml',
                       n_iterations: int = 100):
    """
    Test latency of processing single chunks
    Target: <16ms per chunk for real-time operation
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    print("Creating model...")
    model = MultiTargetCausalBSRNN(config)
    model.eval()
    
    # Create processor
    processor = RealtimeProcessor(model, config, target='vocals', device='cpu')
    
    # Audio parameters
    sample_rate = config['audio']['sample_rate']
    hop_length = config['audio']['hop_length']
    
    # Create dummy audio chunk (hop_length samples)
    chunk_duration_ms = (hop_length / sample_rate) * 1000
    print(f"\nChunk duration: {chunk_duration_ms:.2f} ms")
    print(f"Target processing time: <{chunk_duration_ms:.2f} ms for real-time\n")
    
    # Warm-up
    print("Warming up...")
    dummy_chunk = torch.randn(2, hop_length)
    for _ in range(10):
        _ = processor.process_chunk(dummy_chunk)
    
    processor.reset()
    
    # Benchmark
    print(f"Benchmarking {n_iterations} iterations...")
    processing_times = []
    
    for i in range(n_iterations):
        chunk = torch.randn(2, hop_length)
        
        start_time = time.time()
        output = processor.process_chunk(chunk)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        if output is not None:
            processing_times.append(processing_time)
    
    # Statistics
    processing_times = np.array(processing_times)
    
    print(f"\n{'='*60}")
    print("LATENCY TEST RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Number of chunks processed: {len(processing_times)}")
    print(f"\nProcessing Time (ms):")
    print(f"  Mean: {np.mean(processing_times):.2f}")
    print(f"  Median: {np.median(processing_times):.2f}")
    print(f"  Std: {np.std(processing_times):.2f}")
    print(f"  Min: {np.min(processing_times):.2f}")
    print(f"  Max: {np.max(processing_times):.2f}")
    print(f"  95th percentile: {np.percentile(processing_times, 95):.2f}")
    print(f"  99th percentile: {np.percentile(processing_times, 99):.2f}")
    
    # Real-time factor
    rtf = np.mean(processing_times) / chunk_duration_ms
    print(f"\nReal-Time Factor (RTF): {rtf:.3f}")
    
    # Check if real-time capable
    max_time = np.max(processing_times)
    target_time = chunk_duration_ms
    
    print(f"\nTarget: {target_time:.2f} ms")
    print(f"Max observed: {max_time:.2f} ms")
    
    if max_time < target_time:
        print(f"\n✓ REAL-TIME CAPABLE")
        print(f"  Headroom: {target_time - max_time:.2f} ms ({((target_time - max_time) / target_time * 100):.1f}%)")
    else:
        print(f"\n✗ NOT REAL-TIME")
        print(f"  Overrun: {max_time - target_time:.2f} ms")
    
    # Total latency
    algorithmic_latency = processor.get_latency_ms()
    total_latency = algorithmic_latency + np.mean(processing_times)
    
    print(f"\nTotal Latency:")
    print(f"  Algorithmic: {algorithmic_latency:.2f} ms")
    print(f"  Processing: {np.mean(processing_times):.2f} ms")
    print(f"  Total: {total_latency:.2f} ms")
    
    target_total_latency = 40  # From config
    if total_latency < target_total_latency:
        print(f"  ✓ Under {target_total_latency}ms target")
    else:
        print(f"  ✗ Over {target_total_latency}ms target")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(processing_times, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(target_time, color='r', linestyle='--', 
               label=f'Target: {target_time:.2f}ms')
    plt.axvline(np.mean(processing_times), color='g', linestyle='--',
               label=f'Mean: {np.mean(processing_times):.2f}ms')
    plt.xlabel('Processing Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Processing Time Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'latency_histogram.png', dpi=150)
    print(f"\nHistogram saved to {output_dir / 'latency_histogram.png'}")
    
    return processing_times


def test_streaming_latency(config_path: str = 'config/model_config.yaml',
                           duration_seconds: float = 10.0):
    """
    Test latency during continuous streaming
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = MultiTargetCausalBSRNN(config)
    model.eval()
    
    # Create processor
    processor = RealtimeProcessor(model, config, target='vocals', device='cpu')
    
    # Audio parameters
    sample_rate = config['audio']['sample_rate']
    hop_length = config['audio']['hop_length']
    
    # Generate test audio
    n_samples = int(duration_seconds * sample_rate)
    test_audio = torch.randn(2, n_samples)
    
    print(f"\n{'='*60}")
    print("STREAMING TEST")
    print(f"{'='*60}\n")
    print(f"Test duration: {duration_seconds:.1f} seconds")
    print(f"Processing...")
    
    # Process stream
    start_time = time.time()
    output = processor.process_stream(test_audio, chunk_size=hop_length)
    total_time = time.time() - start_time
    
    # Get statistics
    stats = processor.get_statistics()
    
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print(f"Audio duration: {duration_seconds:.2f} seconds")
    print(f"Real-Time Factor: {total_time / duration_seconds:.3f}")
    
    print(f"\nPer-chunk statistics:")
    print(f"  Mean: {stats['mean_processing_time_ms']:.2f} ms")
    print(f"  Max: {stats['max_processing_time_ms']:.2f} ms")
    print(f"  Chunks processed: {stats['n_chunks_processed']}")
    
    print(f"\nLatency:")
    print(f"  Algorithmic: {stats['algorithmic_latency_ms']:.2f} ms")
    print(f"  Total: {processor.get_latency_ms():.2f} ms")


def compare_original_vs_causal():
    """
    Compare original BSRNN vs Causal BSRNN
    """
    print(f"\n{'='*60}")
    print("ARCHITECTURE COMPARISON")
    print(f"{'='*60}\n")
    
    print("Original BSRNN:")
    print("  - Bidirectional LSTM (non-causal)")
    print("  - Standard layer normalization")
    print("  - Centered STFT windows")
    print("  - Looks at future frames")
    print("  - Typical latency: ~100-200ms")
    print("  - Better separation quality")
    print("  - NOT suitable for real-time\n")
    
    print("Causal BSRNN (This Implementation):")
    print("  - Unidirectional LSTM (causal)")
    print("  - Cumulative layer normalization")
    print("  - Left-aligned STFT windows")
    print("  - Only uses past frames")
    print("  - Target latency: <40ms")
    print("  - 8-bit quantization support")
    print("  - Suitable for real-time")
    print("  - Slightly lower quality vs original")


if __name__ == '__main__':
    # Compare architectures
    compare_original_vs_causal()
    
    # Test chunk latency
    processing_times = test_chunk_latency(n_iterations=200)
    
    # Test streaming
    test_streaming_latency(duration_seconds=5.0)
    
    print(f"\n{'='*60}")
    print("All tests complete!")
    print(f"{'='*60}\n")