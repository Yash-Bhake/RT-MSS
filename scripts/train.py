#!/usr/bin/env python3
"""Training script for Causal BSRNN vocal separation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import yaml
from pathlib import Path

from src.model.causal_bsrnn import CausalBSRNN
from src.data.dataset import MUSDB18Dataset, collate_fn
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train Causal BSRNN')
    parser.add_argument('--config', type=str, default='config/vocal_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, default ='src/data/musdb18_sampled',
                        help='Path to MUSDB18 dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Create model
    model = CausalBSRNN(
        n_fft=config['model']['n_fft'],
        hop_length=config['model']['hop_length'],
        feature_dim=config['model']['feature_dim'],
        num_repeat=config['model']['num_repeat'],
        hidden_dim=config['model']['hidden_dim']
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Create datasets
    train_dataset = MUSDB18Dataset(
        root_dir=args.data_dir,
        split='train',
        segment_length=config['data']['segment_length'],
        sample_rate=config['data']['sample_rate'],
        use_augmentation=config['data'].get('use_augmentation', False),
        augmentation_config=config['data'].get('augmentation', {})
    )
    
    val_dataset = MUSDB18Dataset(
        root_dir=args.data_dir,
        split='test',
        segment_length=config['data']['segment_length'],
        sample_rate=config['data']['sample_rate'],
        use_augmentation=False
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create trainer
    trainer = Trainer(model, train_dataset, val_dataset, config['training'])
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    print("Training completed!")


if __name__ == '__main__':
    main()