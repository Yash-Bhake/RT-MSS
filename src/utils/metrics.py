import torch
import numpy as np
from typing import Tuple


def si_snr(estimate: torch.Tensor, reference: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR).
    
    Args:
        estimate: (B, T) or (T,) estimated signal
        reference: (B, T) or (T,) reference signal
        eps: small constant for numerical stability
    
    Returns:
        SI-SNR value in dB
    """
    if estimate.dim() == 1:
        estimate = estimate.unsqueeze(0)
        reference = reference.unsqueeze(0)
    
    # Zero-mean normalization
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    
    # Compute scaling factor
    reference_energy = torch.sum(reference ** 2, dim=-1, keepdim=True)
    scale = torch.sum(estimate * reference, dim=-1, keepdim=True) / (reference_energy + eps)
    
    # Compute SI-SNR
    scaled_reference = scale * reference
    noise = estimate - scaled_reference
    
    si_snr_value = 10 * torch.log10(
        torch.sum(scaled_reference ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps) + eps
    )
    
    return si_snr_value.mean()


def sdr(estimate: torch.Tensor, reference: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Signal-to-Distortion Ratio (SDR).
    
    Args:
        estimate: (B, T) or (T,) estimated signal
        reference: (B, T) or (T,) reference signal
        eps: small constant for numerical stability
    
    Returns:
        SDR value in dB
    """
    if estimate.dim() == 1:
        estimate = estimate.unsqueeze(0)
        reference = reference.unsqueeze(0)
    
    # Compute SDR
    reference_energy = torch.sum(reference ** 2, dim=-1)
    error = estimate - reference
    error_energy = torch.sum(error ** 2, dim=-1)
    
    sdr_value = 10 * torch.log10(reference_energy / (error_energy + eps) + eps)
    
    return sdr_value.mean()


def compute_rtf(processing_time: float, audio_length: float) -> float:
    """
    Compute Real-Time Factor (RTF).
    
    Args:
        processing_time: Time taken to process audio (seconds)
        audio_length: Length of audio processed (seconds)
    
    Returns:
        RTF: processing_time / audio_length
    """
    return processing_time / audio_length


class MetricsCalculator:
    """Helper class to calculate and accumulate metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.si_snr_values = []
        self.sdr_values = []
        self.rtf_values = []
    
    def update(self, estimate: torch.Tensor, reference: torch.Tensor, 
               processing_time: float = None, audio_length: float = None):
        """
        Update metrics with new values.
        
        Args:
            estimate: estimated signal
            reference: reference signal
            processing_time: time taken to process (optional)
            audio_length: length of audio (optional)
        """
        # Move to CPU for metric computation
        estimate = estimate.detach().cpu()
        reference = reference.detach().cpu()
        
        # Compute metrics
        si_snr_val = si_snr(estimate, reference)
        sdr_val = sdr(estimate, reference)
        
        self.si_snr_values.append(si_snr_val.item())
        self.sdr_values.append(sdr_val.item())
        
        if processing_time is not None and audio_length is not None:
            rtf = compute_rtf(processing_time, audio_length)
            self.rtf_values.append(rtf)
    
    def get_average(self) -> dict:
        """Get average of all accumulated metrics."""
        metrics = {
            'SI-SNR': np.mean(self.si_snr_values) if self.si_snr_values else 0.0,
            'SDR': np.mean(self.sdr_values) if self.sdr_values else 0.0,
        }
        
        if self.rtf_values:
            metrics['RTF'] = np.mean(self.rtf_values)
        
        return metrics
    
    def get_summary(self) -> str:
        """Get a formatted summary of metrics."""
        metrics = self.get_average()
        
        summary = "Metrics Summary:\n"
        summary += f"  SI-SNR: {metrics['SI-SNR']:.3f} dB\n"
        summary += f"  SDR: {metrics['SDR']:.3f} dB\n"
        
        if 'RTF' in metrics:
            summary += f"  RTF: {metrics['RTF']:.4f}\n"
        
        return summary