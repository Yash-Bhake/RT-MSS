#!/usr/bin/env python3
"""
Training script for Causal BSRNN vocal separation - FIXED VERSION
"""

import sys
import os
import torch
import argparse
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import from src directory structure
from src.model.causal_bsrnn import CausalBSRNN
from src.training.trainer import Trainer
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='Train Causal BSRNN')
    parser.add_argument('--config', type=str, default='config/vocal_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to MUSDB18 dataset root')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load Configuration
    print("Loading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Inject data_dir into config
    if 'data' not in config:
        config['data'] = {}
    config['data']['root_dir'] = args.data_dir
    
    print("Configuration loaded successfully.")
    print(f"Dataset: {args.data_dir}")
    
    # Initialize Model
    model_conf = config['model']
    model = CausalBSRNN(
        n_fft=model_conf['n_fft'],
        hop_length=model_conf['hop_length'],
        feature_dim=model_conf['feature_dim'],
        num_repeat=model_conf['num_repeat'],
        hidden_dim=model_conf['hidden_dim']
    )
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel created with {num_params:.2f}M parameters")
    print(f"  n_fft: {model_conf['n_fft']}")
    print(f"  hop_length: {model_conf['hop_length']}")
    print(f"  feature_dim: {model_conf['feature_dim']}")
    print(f"  num_repeat: {model_conf['num_repeat']}")
    print(f"  hidden_dim: {model_conf['hidden_dim']}")
    
    # Initialize Trainer
    trainer = Trainer(model, config)
    
    # Resume if checkpoint provided
    if args.checkpoint:
        print(f"\nResuming from checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Start Training
    print("\n" + "="*60)
    print("  STARTING TRAINING")
    print("="*60)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer._save_checkpoint("interrupted_model.pt")
        print("Checkpoint saved: interrupted_model.pt")
        
    print("\n" + "="*60)
    print("  TRAINING COMPLETED")
    print("="*60)
    print(f"Best model saved at: {trainer.checkpoint_dir / 'best_model.pt'}")
    print(f"Last model saved at: {trainer.checkpoint_dir / 'last_model.pt'}")
    print(f"TensorBoard logs: logs/{config['training']['name']}")
    print("\nView training progress with:")
    print(f"  tensorboard --logdir logs/{config['training']['name']}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()