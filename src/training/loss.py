import torch
import torch.nn as nn


class MultiDomainLoss(nn.Module):
    """Multi-domain loss combining frequency and time domain MAE."""
    
    def __init__(self, n_fft=1024, hop_length=512, device='cuda'):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        
        # Register window
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def forward(self, pred_spec, target_spec, pred_audio=None, target_audio=None):
        """
        Compute multi-domain loss.
        
        Args:
            pred_spec: (B, F, T, 2) predicted spectrogram
            target_spec: (B, F, T, 2) target spectrogram
            pred_audio: (B, T) predicted audio (optional)
            target_audio: (B, T) target audio (optional)
        
        Returns:
            total_loss: Combined loss
        """
        # Frequency domain loss (MAE on real and imaginary parts)
        freq_loss_real = torch.mean(torch.abs(pred_spec[..., 0] - target_spec[..., 0]))
        freq_loss_imag = torch.mean(torch.abs(pred_spec[..., 1] - target_spec[..., 1]))
        freq_loss = freq_loss_real + freq_loss_imag
        
        # Time domain loss
        if pred_audio is not None and target_audio is not None:
            time_loss = torch.mean(torch.abs(pred_audio - target_audio))
        else:
            # Convert spectrograms to audio
            pred_audio = self._spec_to_audio(pred_spec)
            target_audio = self._spec_to_audio(target_spec)
            
            # Ensure same length
            min_len = min(pred_audio.size(-1), target_audio.size(-1))
            pred_audio = pred_audio[..., :min_len]
            target_audio = target_audio[..., :min_len]
            
            time_loss = torch.mean(torch.abs(pred_audio - target_audio))
        
        # Combined loss
        total_loss = freq_loss + time_loss
        
        return total_loss
    
    def _spec_to_audio(self, spec):
        """Convert spectrogram to audio."""
        # spec: (B, F, T, 2)
        spec = spec.permute(0, 1, 3, 2)  # (B, F, 2, T)
        spec_complex = torch.view_as_complex(spec.contiguous())
        
        audio = torch.istft(
            spec_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=False
        )
        
        return audio