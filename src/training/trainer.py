import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json


class Trainer:
    """Trainer for Causal BSRNN."""
    
    def __init__(self, model, train_dataset, val_dataset, config):
        """
        Args:
            model: CausalBSRNN model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration dict
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.model.to(self.device)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 2),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2
        )
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = ExponentialLR(
            self.optimizer,
            gamma=config.get('lr_decay', 0.98)
        )
        
        # Loss function
        from .loss import MultiDomainLoss
        self.criterion = MultiDomainLoss(
            n_fft=model.n_fft,
            hop_length=model.hop_length,
            device=self.device
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for mixture, target in pbar:
            mixture = mixture.to(self.device)
            target = target.to(self.device)
            
            # Compute STFT
            mixture_spec = self._compute_stft(mixture)
            target_spec = self._compute_stft(target)
            
            # Forward pass
            separated_spec, _ = self.model(mixture_spec)
            
            # Compute loss
            loss = self.criterion(separated_spec, target_spec, mixture, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('max_grad_norm', 5.0)
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for mixture, target in tqdm(self.val_loader, desc="Validation"):
                mixture = mixture.to(self.device)
                target = target.to(self.device)
                
                # Process in chunks for long audio
                chunk_size = self.config.get('val_chunk_size', 44100 * 30)  # 30 seconds
                
                if mixture.size(-1) > chunk_size:
                    separated = self._process_long_audio(mixture, chunk_size)
                else:
                    mixture_spec = self._compute_stft(mixture)
                    separated_spec, _ = self.model(mixture_spec)
                    separated = self._compute_istft(separated_spec)
                
                # Trim to target length
                separated = separated[..., :target.size(-1)]
                
                # Compute loss
                separated_spec = self._compute_stft(separated)
                target_spec = self._compute_stft(target)
                loss = self.criterion(separated_spec, target_spec, separated, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _process_long_audio(self, mixture, chunk_size):
        """Process long audio in chunks with overlap-add."""
        hop_size = chunk_size // 2
        num_chunks = (mixture.size(-1) - chunk_size) // hop_size + 1
        
        output = torch.zeros_like(mixture)
        fade_window = torch.hann_window(chunk_size, device=self.device)
        
        states = None
        
        for i in range(num_chunks + 1):
            start = i * hop_size
            end = min(start + chunk_size, mixture.size(-1))
            
            if end - start < chunk_size:
                # Last chunk - pad
                chunk = mixture[..., start:end]
                pad_len = chunk_size - (end - start)
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))
            else:
                chunk = mixture[..., start:end]
            
            # Process chunk
            chunk_spec = self._compute_stft(chunk)
            separated_spec, states = self.model(chunk_spec, states)
            separated_chunk = self._compute_istft(separated_spec)
            
            # Apply fade window
            if end - start < chunk_size:
                separated_chunk = separated_chunk[..., :end-start]
                fade = fade_window[:end-start]
            else:
                fade = fade_window
            
            separated_chunk = separated_chunk * fade
            
            # Overlap-add
            output[..., start:end] += separated_chunk
        
        return output
    
    def _compute_stft(self, audio):
        """Compute STFT with left-aligned windows for causality."""
        # Use left-aligned window (center=False for causal)
        spec = torch.stft(
            audio,
            n_fft=self.model.n_fft,
            hop_length=self.model.hop_length,
            window=self.model.window,
            center=False,  # Causal
            return_complex=True
        )
        
        # Convert to (B, F, T, 2) format
        spec = spec.permute(0, 1, 3, 2)  # (B, F, 2, T) -> (B, F, T, 2)
        
        return spec
    
    def _compute_istft(self, spec):
        """Compute inverse STFT."""
        # Convert from (B, F, T, 2) to (B, F, 2, T)
        spec = spec.permute(0, 1, 3, 2)
        
        # Convert to complex
        spec_complex = torch.view_as_complex(spec.contiguous())
        
        # ISTFT
        audio = torch.istft(
            spec_complex,
            n_fft=self.model.n_fft,
            hop_length=self.model.hop_length,
            window=self.model.window,
            center=False  # Causal
        )
        
        return audio
    
    def train(self, num_epochs):
        """Train for multiple epochs."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
            
            # Learning rate decay every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.scheduler.step()
                print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint('best_model.pt')
                print(f"New best model saved with val loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.get('early_stopping_patience', 10):
                print(f"Early stopping after {epoch + 1} epochs")
                break
    
    def _save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")