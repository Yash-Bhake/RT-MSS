"""
Custom layers for Causal Band-Split RNN
Includes cumulative normalization and causal convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CumulativeLayerNorm(nn.Module):
    """
    Cumulative Layer Normalization for causal processing.
    Normalizes using only past statistics (running mean/var).
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
        # Running statistics for causal normalization
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) or (B, T, C)
        Returns:
            Normalized tensor
        """
        if x.dim() == 3 and x.size(1) == self.dim:
            # (B, C, T) format
            x = x.transpose(1, 2)  # -> (B, T, C)
            transposed = True
        else:
            transposed = False
            
        if self.training:
            # Compute cumulative statistics along time dimension
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = 0.9 * self.running_mean + 0.1 * mean.mean(dim=(0, 1))
                self.running_var = 0.9 * self.running_var + 0.1 * var.mean(dim=(0, 1))
                
            out = (x - mean) / (var + self.eps).sqrt()
        else:
            # Use running statistics for inference (causal)
            out = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        
        out = out * self.gamma + self.beta
        
        if transposed:
            out = out.transpose(1, 2)
            
        return out


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution with left-padding
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=0, dilation=dilation)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        """
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class BandSplitModule(nn.Module):
    """
    Band split module that splits spectrogram into subbands
    with predefined bandwidths
    """
    def __init__(self, band_specs: list, n_features: int):
        """
        Args:
            band_specs: List of [start_freq, end_freq, bandwidth] in Hz
            n_features: Feature dimension for each subband
        """
        super().__init__()
        self.band_specs = band_specs
        self.n_features = n_features
        
        # Calculate band indices and create normalization + FC for each band
        self.band_ranges = []
        self.norms = nn.ModuleList()
        self.fcs = nn.ModuleList()
        
        for start, end, bw in band_specs:
            n_bands = int((end - start) / bw)
            for i in range(n_bands):
                band_start = start + i * bw
                band_end = min(start + (i + 1) * bw, end)
                self.band_ranges.append((band_start, band_end))
                
                # Each band gets its own norm and FC
                band_width = band_end - band_start
                self.norms.append(CumulativeLayerNorm(band_width * 2))  # *2 for real+imag
                self.fcs.append(nn.Linear(band_width * 2, n_features))
        
        self.n_bands = len(self.band_ranges)
    
    def freq_to_bin(self, freq: float, sample_rate: int, n_fft: int) -> int:
        """Convert frequency in Hz to FFT bin index"""
        return int(freq * n_fft / sample_rate)
    
    def forward(self, x: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Args:
            x: Complex spectrogram (B, F, T)
            sample_rate: Sample rate in Hz
        Returns:
            Band features (B, N, K, T) where K is number of bands
        """
        B, F, T = x.shape
        n_fft = (F - 1) * 2
        
        band_features = []
        
        for idx, (start_freq, end_freq) in enumerate(self.band_ranges):
            start_bin = self.freq_to_bin(start_freq, sample_rate, n_fft)
            end_bin = self.freq_to_bin(end_freq, sample_rate, n_fft)
            end_bin = min(end_bin, F)
            
            # Extract subband
            subband = x[:, start_bin:end_bin, :]  # (B, G, T)
            
            # Split real and imaginary parts and concatenate
            subband_real = subband.real
            subband_imag = subband.imag
            subband_cat = torch.cat([subband_real, subband_imag], dim=1)  # (B, 2G, T)
            
            # Normalize and project
            subband_cat = subband_cat.transpose(1, 2)  # (B, T, 2G)
            subband_norm = self.norms[idx](subband_cat)
            subband_feat = self.fcs[idx](subband_norm)  # (B, T, N)
            
            band_features.append(subband_feat)
        
        # Stack into (B, T, K, N) then transpose to (B, N, K, T)
        band_features = torch.stack(band_features, dim=2)  # (B, T, K, N)
        band_features = band_features.permute(0, 3, 2, 1)  # (B, N, K, T)
        
        return band_features


class MaskEstimationModule(nn.Module):
    """
    Mask estimation module that generates complex T-F masks
    """
    def __init__(self, band_specs: list, n_features: int, mlp_hidden: int):
        super().__init__()
        self.band_specs = band_specs
        
        # Calculate band ranges
        self.band_ranges = []
        for start, end, bw in band_specs:
            n_bands = int((end - start) / bw)
            for i in range(n_bands):
                band_start = start + i * bw
                band_end = min(start + (i + 1) * bw, end)
                self.band_ranges.append((band_start, band_end))
        
        self.n_bands = len(self.band_ranges)
        
        # Each band gets its own norm and MLP
        self.norms = nn.ModuleList()
        self.mlps = nn.ModuleList()
        
        for start_freq, end_freq in self.band_ranges:
            band_width = end_freq - start_freq
            self.norms.append(CumulativeLayerNorm(n_features))
            
            # MLP with one hidden layer
            self.mlps.append(nn.Sequential(
                nn.Linear(n_features, mlp_hidden),
                nn.Tanh(),
                nn.Linear(mlp_hidden, band_width * 2),  # *2 for real+imag
                nn.GLU(dim=-1)  # Reduces to band_width
            ))
    
    def freq_to_bin(self, freq: float, sample_rate: int, n_fft: int) -> int:
        return int(freq * n_fft / sample_rate)
    
    def forward(self, q: torch.Tensor, sample_rate: int, n_fft: int) -> torch.Tensor:
        """
        Args:
            q: Band features (B, N, K, T)
            sample_rate: Sample rate
            n_fft: FFT size
        Returns:
            Complex mask (B, F, T)
        """
        B, N, K, T = q.shape
        F = n_fft // 2 + 1
        
        # Initialize full mask
        mask = torch.zeros(B, F, T, dtype=torch.complex64, device=q.device)
        
        for idx in range(K):
            # Extract band feature
            band_feat = q[:, :, idx, :]  # (B, N, T)
            band_feat = band_feat.transpose(1, 2)  # (B, T, N)
            
            # Normalize and pass through MLP
            band_norm = self.norms[idx](band_feat)
            band_mask = self.mlps[idx](band_norm)  # (B, T, G)
            band_mask = band_mask.transpose(1, 2)  # (B, G, T)
            
            # Convert to complex
            start_freq, end_freq = self.band_ranges[idx]
            start_bin = self.freq_to_bin(start_freq, sample_rate, n_fft)
            end_bin = self.freq_to_bin(end_freq, sample_rate, n_fft)
            end_bin = min(end_bin, F)
            
            # Use tanh activation for mask values
            band_mask = torch.tanh(band_mask)
            mask[:, start_bin:end_bin, :] = band_mask
        
        return mask