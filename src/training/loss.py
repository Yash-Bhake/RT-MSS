"""
Loss functions for music source separation
"""

import torch
import torch.nn as nn


class FrequencyDomainLoss(nn.Module):
    """
    Frequency domain loss (MAE on real and imaginary parts)
    """
    
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted complex spectrogram (B, C, F, T)
            target: Target complex spectrogram (B, C, F, T)
        Returns:
            Loss value
        """
        # Separate real and imaginary parts
        pred_real = pred.real
        pred_imag = pred.imag
        target_real = target.real
        target_imag = target.imag
        
        # MAE on both parts
        loss_real = self.mae(pred_real, target_real)
        loss_imag = self.mae(pred_imag, target_imag)
        
        return loss_real + loss_imag


class TimeDomainLoss(nn.Module):
    """
    Time domain loss (MAE on waveform)
    """
    
    def __init__(self, n_fft: int = 1408, hop_length: int = 704):
        super().__init__()
        self.mae = nn.L1Loss()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)
    
    def forward(self, pred_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_spec: Predicted complex spectrogram (B, C, F, T)
            target_spec: Target complex spectrogram (B, C, F, T)
        Returns:
            Loss value
        """
        B, C = pred_spec.size(0), pred_spec.size(1)
        
        # Inverse STFT
        pred_audio = []
        target_audio = []
        
        window = self.window.to(pred_spec.device)
        
        for b in range(B):
            # Inverse STFT for each channel
            pred_wav = torch.istft(
                pred_spec[b],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                center=False,
                return_complex=False
            )
            
            target_wav = torch.istft(
                target_spec[b],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                center=False,
                return_complex=False
            )
            
            pred_audio.append(pred_wav)
            target_audio.append(target_wav)
        
        pred_audio = torch.stack(pred_audio, dim=0)
        target_audio = torch.stack(target_audio, dim=0)
        
        # MAE in time domain
        return self.mae(pred_audio, target_audio)


class CombinedLoss(nn.Module):
    """
    Combined frequency and time domain loss
    """
    
    def __init__(self, freq_weight: float = 1.0, time_weight: float = 1.0,
                 n_fft: int = 1408, hop_length: int = 704):
        super().__init__()
        self.freq_weight = freq_weight
        self.time_weight = time_weight
        
        self.freq_loss = FrequencyDomainLoss()
        self.time_loss = TimeDomainLoss(n_fft, hop_length)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Args:
            pred: Predicted complex spectrogram (B, C, F, T)
            target: Target complex spectrogram (B, C, F, T)
        Returns:
            Dictionary with total loss and individual losses
        """
        freq_loss = self.freq_loss(pred, target)
        time_loss = self.time_loss(pred, target)
        
        total_loss = self.freq_weight * freq_loss + self.time_weight * time_loss
        
        return {
            'total': total_loss,
            'freq': freq_loss.item(),
            'time': time_loss.item()
        }


def si_snr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Noise Ratio loss
    
    Args:
        pred: Predicted waveform (B, C, T)
        target: Target waveform (B, C, T)
        eps: Small constant for numerical stability
    Returns:
        Negative SI-SNR (lower is better)
    """
    # Zero-mean
    target = target - torch.mean(target, dim=-1, keepdim=True)
    pred = pred - torch.mean(pred, dim=-1, keepdim=True)
    
    # Compute scaling factor
    s_target = torch.sum(pred * target, dim=-1, keepdim=True) / \
               (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    
    # Scaled target
    s_target = s_target * target
    
    # Compute SI-SNR
    si_snr = 10 * torch.log10(
        (torch.sum(s_target ** 2, dim=-1) + eps) /
        (torch.sum((pred - s_target) ** 2, dim=-1) + eps)
    )
    
    # Return negative for minimization
    return -torch.mean(si_snr)


class SISDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio loss (same as SI-SNR)
    """
    
    def __init__(self, n_fft: int = 1408, hop_length: int = 704):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)
    
    def forward(self, pred_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_spec: Predicted complex spectrogram (B, C, F, T)
            target_spec: Target complex spectrogram (B, C, F, T)
        Returns:
            Negative SI-SDR
        """
        B, C = pred_spec.size(0), pred_spec.size(1)
        
        # Convert to time domain
        pred_audio = []
        target_audio = []
        
        window = self.window.to(pred_spec.device)
        
        for b in range(B):
            pred_wav = torch.istft(
                pred_spec[b],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                center=False,
                return_complex=False
            )
            
            target_wav = torch.istft(
                target_spec[b],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                center=False,
                return_complex=False
            )
            
            pred_audio.append(pred_wav)
            target_audio.append(target_wav)
        
        pred_audio = torch.stack(pred_audio, dim=0)
        target_audio = torch.stack(target_audio, dim=0)
        
        return si_snr_loss(pred_audio, target_audio)