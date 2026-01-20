#!/usr/bin/env python3
"""Setup validation and pre-training checks."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
from pathlib import Path
import time
import torchaudio

from src.model.causal_bsrnn import CausalBSRNN
from src.data.dataset import MUSDB18Dataset


def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def check_dependencies():
    """Check if all dependencies are installed."""
    print_section("1. Checking Dependencies")
    
    deps = {
        'torch': torch.__version__,
        'torchaudio': torchaudio.__version__,
    }
    
    for name, version in deps.items():
        print(f"✓ {name}: {version}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("⚠ CUDA not available - will use CPU (slower)")
    
    return True


def check_dataset(data_dir):
    """Check if dataset is properly formatted."""
    print_section("2. Checking Dataset")
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"✗ Dataset not found at: {data_dir}")
        return False
    
    # Check for train and test splits
    train_dir = data_path / 'train'
    test_dir = data_path / 'test'
    
    if not train_dir.exists():
        print(f"✗ Train directory not found: {train_dir}")
        return False
    
    if not test_dir.exists():
        print(f"✗ Test directory not found: {test_dir}")
        return False
    
    # Count songs
    train_songs = list(train_dir.iterdir())
    test_songs = list(test_dir.iterdir())
    
    print(f"✓ Train songs: {len(train_songs)}")
    print(f"✓ Test songs: {len(test_songs)}")
    
    # Check a sample song structure
    if train_songs:
        sample_song = train_songs[0]
        required_files = ['vocals.wav', 'bass.wav', 'drums.wav', 'other.wav', 'mixture.wav']
        
        print(f"\nChecking sample song: {sample_song.name}")
        for fname in required_files:
            fpath = sample_song / fname
            if fpath.exists():
                print(f"  ✓ {fname}")
            else:
                print(f"  ✗ {fname} - MISSING!")
                return False
    
    return True


def test_data_loading(data_dir):
    """Test dataset loading."""
    print_section("3. Testing Data Loading")
    
    try:
        dataset = MUSDB18Dataset(
            root_dir=data_dir,
            split='train',
            segment_length=3.0,
            sample_rate=44100,
            use_augmentation=False
        )
        
        print(f"✓ Dataset created: {len(dataset)} samples")
        
        # Load one sample
        print("\nLoading sample...")
        mixture, vocals = dataset[0]
        
        print(f"✓ Mixture shape: {mixture.shape}")
        print(f"✓ Vocals shape: {vocals.shape}")
        print(f"✓ Sample rate: 44100 Hz")
        print(f"✓ Duration: {mixture.shape[0] / 44100:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False


def test_model_creation(config_path):
    """Test model creation and forward pass."""
    print_section("4. Testing Model Creation")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    print(f"Configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    try:
        model = CausalBSRNN(
            n_fft=model_config['n_fft'],
            hop_length=model_config['hop_length'],
            feature_dim=model_config['feature_dim'],
            num_repeat=model_config['num_repeat'],
            hidden_dim=model_config['hidden_dim']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n✓ Model created successfully")
        print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return model, config
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return None, None


def test_forward_pass(model, config):
    """Test forward pass with dummy data."""
    print_section("5. Testing Forward Pass")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy input
    batch_size = 1
    num_freq_bins = model.n_fft // 2 + 1
    num_frames = 10  # Small number for testing
    
    dummy_input = torch.randn(batch_size, num_freq_bins, num_frames, 2).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Device: {device}")
    
    try:
        start_time = time.time()
        
        with torch.no_grad():
            output, states = model(dummy_input)
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Time: {elapsed*1000:.2f} ms")
        print(f"  States: {len(states)} state tensors")
        
        # Memory usage
        if device.type == 'cuda':
            memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  GPU memory: {memory:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def estimate_training_time(model, config):
    """Estimate training time per epoch."""
    print_section("6. Estimating Training Time")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Simulate one batch
    batch_size = config['training']['batch_size']
    segment_length = config['data']['segment_length']
    sample_rate = config['data']['sample_rate']
    
    num_samples = int(segment_length * sample_rate)
    
    # Create dummy batch
    dummy_mixture = torch.randn(batch_size, num_samples).to(device)
    dummy_target = torch.randn(batch_size, num_samples).to(device)
    
    # Compute STFT
    spec = torch.stft(
        dummy_mixture,
        n_fft=model.n_fft,
        hop_length=model.hop_length,
        window=model.window.to(device),
        center=False,
        return_complex=True
    )

    spec = torch.view_as_real(spec)   # (B, F, T, 2)
    spec = spec.contiguous()

    print(f"Batch shape: {spec.shape}")

    
    # Time forward+backward
    num_iterations = 5
    times = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for i in range(num_iterations):
        start = time.time()
        
        output, _ = model(spec)
        loss = torch.mean((output - spec) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        times.append(time.time() - start)
    
    avg_time = sum(times[1:]) / (num_iterations - 1)  # Skip first (warmup)
    
    print(f"\n✓ Average time per batch: {avg_time:.3f} seconds")
    
    # Estimate epoch time
    batches_per_epoch = config['training'].get('batches_per_epoch', 10000)
    epoch_time = avg_time * batches_per_epoch
    
    print(f"\nEstimated times:")
    print(f"  Per epoch ({batches_per_epoch} batches): {epoch_time/60:.1f} minutes")
    print(f"  For {config['training']['num_epochs']} epochs: {epoch_time*config['training']['num_epochs']/3600:.1f} hours")
    
    if device.type == 'cpu':
        print(f"\n⚠ WARNING: Training on CPU will be slow!")
        print(f"  Consider reducing num_repeat or using GPU")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup validation')
    parser.add_argument('--config', type=str, default='config/vocal_config.yaml')
    parser.add_argument('--data_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    print("\n" + "🎵"*30)
    print("  CAUSAL BSRNN - SETUP VALIDATION")
    print("🎵"*30)
    
    # Run checks
    checks = [
        ("Dependencies", check_dependencies, []),
        ("Dataset", check_dataset, [args.data_dir]),
        ("Data Loading", test_data_loading, [args.data_dir]),
        ("Model Creation", test_model_creation, [args.config]),
    ]
    
    model = None
    config = None
    
    for name, func, func_args in checks:
        result = func(*func_args)
        
        if name == "Model Creation":
            model, config = result
            if model is None:
                print("\n❌ Setup validation FAILED")
                return
        elif not result:
            print(f"\n❌ {name} check FAILED")
            return
    
    # Additional tests with model
    if model and config:
        if not test_forward_pass(model, config):
            print("\n❌ Setup validation FAILED")
            return
        
        estimate_training_time(model, config)
    
    print_section("SUMMARY")
    print("✓ All checks passed!")
    print("\nYou can now start training with:")
    print(f"  python scripts/train.py --config {args.config} --data_dir {args.data_dir}")
    print("\n" + "🎵"*30 + "\n")


if __name__ == '__main__':
    main()