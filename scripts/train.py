"""
Training script for Causal Band-Split RNN
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model.causal_bsrnn import MultiTargetCausalBSRNN
from src.data.dataset import MUSDB18Dataset, MUSDB18Collator
from src.training.loss import CombinedLoss


def train_epoch(model: nn.Module, dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                device: torch.device, target: str) -> dict:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    freq_loss_sum = 0
    time_loss_sum = 0
    
    pbar = tqdm(dataloader, desc='Training')
    
    for batch_idx, (mixture_specs, target_specs) in enumerate(pbar):
        # Move to device
        mixture_specs = mixture_specs.to(device)  # (B, 2, F, T)
        target_specs = target_specs.to(device)
        
        optimizer.zero_grad()
        
        # Process each channel separately
        batch_losses = []
        
        for ch in range(2):  # Stereo
            mix_ch = mixture_specs[:, ch, :, :]  # (B, F, T)
            tgt_ch = target_specs[:, ch, :, :]
            
            # Forward pass (no states needed during training)
            pred_ch, _ = model(mix_ch, target, states=None)
            
            # Compute loss
            loss_dict = criterion(pred_ch, tgt_ch)
            batch_losses.append(loss_dict['total'])
            
            freq_loss_sum += loss_dict['freq']
            time_loss_sum += loss_dict['time']
        
        # Average loss across channels
        loss = torch.mean(torch.stack(batch_losses))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'avg_loss': total_loss / (batch_idx + 1)
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'freq_loss': freq_loss_sum / (len(dataloader) * 2),
        'time_loss': time_loss_sum / (len(dataloader) * 2)
    }


def validate(model: nn.Module, dataloader: DataLoader, 
            criterion: nn.Module, device: torch.device, target: str) -> dict:
    """Validate model"""
    model.eval()
    
    total_loss = 0
    
    with torch.no_grad():
        for mixture_specs, target_specs in tqdm(dataloader, desc='Validation'):
            mixture_specs = mixture_specs.to(device)
            target_specs = target_specs.to(device)
            
            batch_losses = []
            
            for ch in range(2):
                mix_ch = mixture_specs[:, ch, :, :]
                tgt_ch = target_specs[:, ch, :, :]
                
                pred_ch, _ = model(mix_ch, target, states=None)
                loss_dict = criterion(pred_ch, tgt_ch)
                batch_losses.append(loss_dict['total'])
            
            loss = torch.mean(torch.stack(batch_losses))
            total_loss += loss.item()
    
    return {'loss': total_loss / len(dataloader)}


def train(args):
    """Main training function"""
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = MUSDB18Dataset(
        root=args.data_path,
        subset='train',
        target=args.target,
        config=config,
        segment_length=config['training']['segment_length']
    )
    
    # For validation, we use a small subset of training set
    val_dataset = MUSDB18Dataset(
        root=args.data_path,
        subset='train',
        target=args.target,
        config=None,  # No augmentation
        segment_length=config['training']['segment_length']
    )
    
    # Dataloaders
    collator = MUSDB18Collator(
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        sample_rate=config['audio']['sample_rate']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collator
    )
    
    # Create model
    print("Creating model...")
    model = MultiTargetCausalBSRNN(config)
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(
        freq_weight=config['training']['loss']['freq_weight'],
        time_weight=config['training']['loss']['time_weight'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length']
    )
    
    optimizer = Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    scheduler = StepLR(
        optimizer,
        step_size=config['training']['lr_decay_step'],
        gamma=config['training']['lr_decay']
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    checkpoint_dir = Path('checkpoints') / args.target
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTraining {args.target} separation model...")
    print(f"{'='*60}\n")
    
    for epoch in range(config['training']['num_epochs']):
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, args.target
        )
        
        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_metrics = validate(
                model, val_loader, criterion, device, args.target
            )
            
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                checkpoint_path = checkpoint_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': config
                }, checkpoint_path)
                
                print(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= config['training']['early_stop_patience']:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        else:
            print(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # Step scheduler
        scheduler.step()
        
        print()
    
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Causal BSRNN')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to MUSDB18 directory')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to config file')
    parser.add_argument('--target', type=str, default='vocals',
                       choices=['vocals', 'drums', 'bass', 'other'],
                       help='Target source to train')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for training')
    
    args = parser.parse_args()
    train(args)