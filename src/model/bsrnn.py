"""
Original (Non-Causal) Band-Split RNN Architecture

This is the baseline implementation from the paper:
- Bidirectional LSTM
- Standard Layer Normalization
- Centered STFT windows
"""

import torch
import torch.nn as nn
from typing import Tuple


class OriginalBandSplitModule(nn.Module):
    """Original band split with standard layer norm"""
    def __init__(self, band_specs: list, n_features: int):
        super().__init__()
        self.band_specs = band_specs
        self.n_features = n_features
        
        self.band_ranges = []
        self.norms = nn.ModuleList()
        self.fcs = nn.ModuleList()
        
        for start, end, bw in band_specs:
            n_bands = int((end - start) / bw)
            for i in range(n_bands):
                band_start = start + i * bw
                band_end = min(start + (i + 1) * bw, end)
                self.band_ranges.append((band_start, band_end))
                
                band_width = band_end - band_start
                self.norms.append(nn.LayerNorm(band_width * 2))
                self.fcs.append(nn.Linear(band_width * 2, n_features))
        
        self.n_bands = len(self.band_ranges)
    
    def freq_to_bin(self, freq: float, sample_rate: int, n_fft: int) -> int:
        return int(freq * n_fft / sample_rate)
    
    def forward(self, x: torch.Tensor, sample_rate: int) -> torch.Tensor:
        B, F, T = x.shape
        n_fft = (F - 1) * 2
        
        band_features = []
        
        for idx, (start_freq, end_freq) in enumerate(self.band_ranges):
            start_bin = self.freq_to_bin(start_freq, sample_rate, n_fft)
            end_bin = self.freq_to_bin(end_freq, sample_rate, n_fft)
            end_bin = min(end_bin, F)
            
            subband = x[:, start_bin:end_bin, :]
            subband_real = subband.real
            subband_imag = subband.imag
            subband_cat = torch.cat([subband_real, subband_imag], dim=1)
            subband_cat = subband_cat.transpose(1, 2)
            
            subband_norm = self.norms[idx](subband_cat)
            subband_feat = self.fcs[idx](subband_norm)
            band_features.append(subband_feat)
        
        band_features = torch.stack(band_features, dim=2)
        band_features = band_features.permute(0, 3, 2, 1)
        
        return band_features


class OriginalMaskEstimation(nn.Module):
    """Original mask estimation with standard layer norm"""
    def __init__(self, band_specs: list, n_features: int, mlp_hidden: int):
        super().__init__()
        self.band_specs = band_specs
        
        self.band_ranges = []
        for start, end, bw in band_specs:
            n_bands = int((end - start) / bw)
            for i in range(n_bands):
                band_start = start + i * bw
                band_end = min(start + (i + 1) * bw, end)
                self.band_ranges.append((band_start, band_end))
        
        self.n_bands = len(self.band_ranges)
        self.norms = nn.ModuleList()
        self.mlps = nn.ModuleList()
        
        for start_freq, end_freq in self.band_ranges:
            band_width = end_freq - start_freq
            self.norms.append(nn.LayerNorm(n_features))
            
            self.mlps.append(nn.Sequential(
                nn.Linear(n_features, mlp_hidden),
                nn.Tanh(),
                nn.Linear(mlp_hidden, band_width * 2),
                nn.GLU(dim=-1)
            ))
    
    def freq_to_bin(self, freq: float, sample_rate: int, n_fft: int) -> int:
        return int(freq * n_fft / sample_rate)
    
    def forward(self, q: torch.Tensor, sample_rate: int, n_fft: int) -> torch.Tensor:
        B, N, K, T = q.shape
        F = n_fft // 2 + 1
        
        mask = torch.zeros(B, F, T, dtype=torch.complex64, device=q.device)
        
        for idx in range(K):
            band_feat = q[:, :, idx, :]
            band_feat = band_feat.transpose(1, 2)
            
            band_norm = self.norms[idx](band_feat)
            band_mask = self.mlps[idx](band_norm)
            band_mask = band_mask.transpose(1, 2)
            band_mask = torch.tanh(band_mask)
            
            start_freq, end_freq = self.band_ranges[idx]
            start_bin = self.freq_to_bin(start_freq, sample_rate, n_fft)
            end_bin = self.freq_to_bin(end_freq, sample_rate, n_fft)
            end_bin = min(end_bin, F)
            
            mask[:, start_bin:end_bin, :] = band_mask
        
        return mask


class ResidualRNN(nn.Module):
    """Residual RNN block with bidirectional LSTM"""
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = True):
        super().__init__()
        self.norm = nn.GroupNorm(1, input_size)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=1, 
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, input_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) or (B, T, C)
        """
        residual = x
        
        if x.dim() == 3 and x.size(1) != x.size(2):
            # Normalize along channel dimension
            x = self.norm(x)
            x = x.transpose(1, 2)  # -> (B, T, C)
        
        x, _ = self.lstm(x)
        x = self.fc(x)
        
        if residual.size(1) != x.size(1):
            x = x.transpose(1, 2)
        
        return x + residual


class OriginalBSRNN(nn.Module):
    """
    Original Band-Split RNN (Non-Causal)
    
    Architecture:
    1. Band Split Module: Splits spectrogram into K subbands
    2. Sequence Modeling: Bidirectional LSTM across time
    3. Band Modeling: Bidirectional LSTM across bands
    4. Mask Estimation: Generates complex T-F masks per band
    
    Key differences from causal version:
    - Uses bidirectional LSTM (looks at future frames)
    - Uses standard layer normalization
    - Uses centered STFT windows
    - Higher latency but better performance
    """
    
    def __init__(self, band_specs: list, n_features: int = 128,
                 n_layers: int = 12, lstm_hidden: int = 256,
                 mlp_hidden: int = 512, sample_rate: int = 44100,
                 n_fft: int = 2048):
        super().__init__()
        
        self.band_specs = band_specs
        self.n_features = n_features
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        
        # Band split module
        self.band_split = OriginalBandSplitModule(band_specs, n_features)
        
        # Interleaved sequence and band modeling
        self.sequence_rnns = nn.ModuleList([
            ResidualRNN(n_features, lstm_hidden, bidirectional=True)
            for _ in range(n_layers // 2)
        ])
        
        self.band_rnns = nn.ModuleList([
            ResidualRNN(n_features, lstm_hidden, bidirectional=True)
            for _ in range(n_layers // 2)
        ])
        
        # Mask estimation module
        self.mask_estimation = OriginalMaskEstimation(band_specs, n_features, mlp_hidden)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex spectrogram (B, F, T)
        Returns:
            Separated source spectrogram (B, F, T)
        """
        # Band split
        z = self.band_split(x, self.sample_rate)  # (B, N, K, T)
        
        B, N, K, T = z.shape
        
        # Interleaved sequence and band modeling
        for seq_rnn, band_rnn in zip(self.sequence_rnns, self.band_rnns):
            # Sequence modeling: process each band across time
            z_seq = z.permute(0, 2, 1, 3).reshape(B * K, N, T)  # (B*K, N, T)
            z_seq = seq_rnn(z_seq)
            z = z_seq.reshape(B, K, N, T).permute(0, 2, 1, 3)  # (B, N, K, T)
            
            # Band modeling: process each frame across bands
            z_band = z.permute(0, 3, 1, 2).reshape(B * T, N, K)  # (B*T, N, K)
            z_band = band_rnn(z_band)
            z = z_band.reshape(B, T, N, K).permute(0, 2, 3, 1)  # (B, N, K, T)
        
        # Mask estimation
        mask = self.mask_estimation(z, self.sample_rate, self.n_fft)
        
        # Apply mask
        output = x * mask
        
        return output