import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

from src.data.dataset import MUSDB18Dataset, collate_fn
from .loss import HybridLoss, MultiDomainLoss
import torch.nn.functional as F

class Trainer:
    """
    Trainer for Causal BSRNN
    
    1. Gradient monitoring in TensorBoard
    2. Simplified STFT handling (center=False everywhere)
    3. No global normalization (model handles this internally via cumulative norm)
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(
            config.get('training', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        self.model.to(self.device)
        
        # TensorBoard Logger
        log_dir = Path("logs") / config.get('training', {}).get('name', 'experiment')
        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard logs: {log_dir}")
        
        # Data loaders
        train_config = config['training']
        data_config = config['data']
        
        self.train_dataset = MUSDB18Dataset(
            root_dir=data_config['root_dir'],
            split='train',
            config=config,
            total_epochs=train_config['num_epochs']
        )
        
        self.val_dataset = MUSDB18Dataset(
            root_dir=data_config['root_dir'],
            split='test',
            config=config,
            total_epochs=train_config['num_epochs']
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config.get('batch_size', 8),
            shuffle=True,
            num_workers=train_config.get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True  # Ensure consistent batch sizes
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=train_config.get('num_workers', 4),
            collate_fn=collate_fn
        )
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=train_config.get('learning_rate', 1e-3),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss Function
        self.loss_type = train_config.get('loss_type', 'hybrid').lower()
        print(f"Using Loss Function: {self.loss_type}")
        
        if self.loss_type == 'hybrid':
            self.criterion = HybridLoss(
                n_fft=model.n_fft,
                hop_length=model.hop_length,
                device=self.device,
                alpha_time=train_config.get('alpha_time', 1.0),
                alpha_freq=train_config.get('alpha_freq', 1.0)
            ).to(self.device)
        else:
            # Legacy multi-domain loss
            self.criterion = MultiDomainLoss(
                n_fft=model.n_fft,
                hop_length=model.hop_length,
                device=self.device
            ).to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.global_step = 0
        
        self.checkpoint_dir = Path(train_config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.train_losses = []
        self.val_losses = []

    def _compute_stft(self, audio):
        """
        Compute STFT with proactive padding to satisfy the iSTFT OLA condition.
        """
        T = audio.size(-1)
        hop = self.model.hop_length
        n_fft = self.model.n_fft
        
        # Pad audio to ensure (num_frames - 1) * hop + n_fft >= T
        # Adding one extra hop_length is usually sufficient.
        pad_amount = (hop - (T % hop)) % hop
        audio = torch.nn.functional.pad(audio, (0, pad_amount + hop))

        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop,
            window=self.model.window,
            center=False, 
            return_complex=True,
            normalized=True  
        )
        return torch.view_as_real(spec).contiguous()
        
    def _compute_istft(self, spec, length=None):
        """
        Compute iSTFT with consistent parameters.
        
        Args:
            spec: (B, F, T, 2) complex spectrogram as real tensor
            length: Optional target length
        
        Returns:
            audio: (B, T) waveform
        """
        spec_complex = torch.view_as_complex(spec.contiguous())
        audio = torch.istft(
            spec_complex,
            n_fft=self.model.n_fft,
            hop_length=self.model.hop_length,
            window=self.model.window,
            center=False,  # CRITICAL for causality
            normalized=True,  # CRITICAL for stability
            length=length
        )
        return audio
    
    def _log_gradients(self, step):
        """
        Log gradient statistics to TensorBoard.
        
        This helps monitor:
        - Gradient flow (are gradients reaching all layers?)
        - Gradient magnitude (too small = vanishing, too large = exploding)
        - Layer-wise gradient distribution
        """
        total_norm = 0.0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
                
                # Log individual layer gradients
                self.writer.add_scalar(f'Gradients/{name}', param_norm, step)
        
        total_norm = total_norm ** 0.5
        
        # Log overall gradient norm
        self.writer.add_scalar('Gradients/total_norm', total_norm, step)
        self.writer.add_scalar('Gradients/avg_norm', total_norm / max(param_count, 1), step)
        
        return total_norm
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Update dataset epoch for augmentation
        self.train_dataset.set_epoch(self.current_epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (mixture, target) in enumerate(pbar):
            mixture = mixture.to(self.device)
            target = target.to(self.device)
            target_len = target.size(-1)
            
            # 1. Compute Input STFT
            mixture_spec = self._compute_stft(mixture)
            
            # 2. Forward Pass (Model handles normalization internally)
            separated_spec, _ = self.model(mixture_spec)
            
            # 3. Loss Calculation
            if self.loss_type == 'multidomain':
                target_spec = self._compute_stft(target)
                loss = self.criterion(separated_spec, target_spec, target, target_len)
            else:
                # Hybrid loss (recommended)
                loss = self.criterion(separated_spec, target, target_len)
            
            # 4. Backward Pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # 5. Gradient Clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training'].get('max_grad_norm', 1.0)
            )
            
            # 6. Optimizer Step
            self.optimizer.step()
            
            # 7. Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'grad': f"{grad_norm:.4f}"
            })
            
            # Log to TensorBoard every N steps
            if self.global_step % 10 == 0:
                self.writer.add_scalar("Loss/Train_Step", loss.item(), self.global_step)
                self.writer.add_scalar("Gradients/Norm_Step", grad_norm, self.global_step)
            
            # Detailed gradient logging every 100 steps
            if self.global_step % 100 == 0:
                self._log_gradients(self.global_step)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def validate(self):
        """Validate on test set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        subset_size = self.config['training'].get('val_subset_size', -1)
        
        with torch.no_grad():
            for i, (mixture, target) in enumerate(tqdm(self.val_loader, desc="Validating")):
                if subset_size > 0 and i >= subset_size:
                    break
                
                mixture = mixture.to(self.device)
                target = target.to(self.device)
                target_len = target.size(-1)
                
                # Compute STFT
                mixture_spec = self._compute_stft(mixture)
                
                # Forward pass
                separated_spec, _ = self.model(mixture_spec)
                
                # Loss calculation
                if self.loss_type == 'multidomain':
                    target_spec = self._compute_stft(target)
                    loss = self.criterion(separated_spec, target_spec, target, target_len)
                else:
                    loss = self.criterion(separated_spec, target, target_len)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']
        val_freq = self.config['training'].get('val_frequency', 1)
        
        print(f"\nTraining Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch Size: {self.config['training']['batch_size']}")
        print(f"  Learning Rate: {self.config['training']['learning_rate']}")
        print(f"  Loss Type: {self.loss_type}")
        print(f"  Model Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        print()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            self.writer.add_scalar("Loss/Train_Epoch", train_loss, epoch)
            print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validate
            if (epoch + 1) % val_freq == 0 or epoch == num_epochs - 1:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                self.writer.add_scalar("Loss/Validation", val_loss, epoch)
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
                
                # Update scheduler
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self._save_checkpoint('best_model.pt')
                    print(f"  ✓ New best model saved! Val Loss: {val_loss:.4f}")
                else:
                    self.epochs_without_improvement += 1
            else:
                self.val_losses.append(self.val_losses[-1] if self.val_losses else 0)
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("LR", current_lr, epoch)
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            self._save_checkpoint('last_model.pt')
            self._save_history()
            
            # Early stopping
            patience = self.config['training'].get('early_stopping_patience', 10)
            if self.epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
                break
        
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*50)
        
        self.writer.close()

    def _save_checkpoint(self, filename):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
        
    def load_checkpoint(self, filename):
        """Load training checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        
    def _save_history(self):
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'current_epoch': self.current_epoch
        }
        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=4)