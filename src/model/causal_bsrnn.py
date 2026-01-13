import torch
import torch.nn as nn
import numpy as np
from .normalization import CumulativeLayerNorm, GroupNorm


class CausalBandSplitModule(nn.Module):
    """Band split module that splits spectrogram into subbands."""
    
    def __init__(self, band_specs, feature_dim):
        super().__init__()
        self.band_specs = band_specs  # List of (start_freq, end_freq)
        self.num_bands = len(band_specs)
        self.feature_dim = feature_dim
        
        # Create normalization and FC layers for each band
        self.band_norms = nn.ModuleList()
        self.band_fcs = nn.ModuleList()
        
        for start_f, end_f in band_specs:
            bandwidth = end_f - start_f
            self.band_norms.append(nn.LayerNorm(bandwidth * 2))  # *2 for real and imag
            self.band_fcs.append(nn.Linear(bandwidth * 2, feature_dim))
    
    def forward(self, x):
        """
        Args:
            x: (B, F, T, 2) complex spectrogram
        Returns:
            features: (B, K, N, T) band features
        """
        B, F, T, _ = x.shape
        band_features = []
        
        for i, (start_f, end_f) in enumerate(self.band_specs):
            # Extract band
            band = x[:, start_f:end_f, :, :]  # (B, Gi, T, 2)
            band = band.reshape(B, -1, T).transpose(1, 2)  # (B, T, Gi*2)
            
            # Normalize and project
            band = self.band_norms[i](band)  # (B, T, Gi*2)
            band = self.band_fcs[i](band)  # (B, T, N)
            
            band_features.append(band.transpose(1, 2))  # (B, N, T)
        
        # Stack bands
        features = torch.stack(band_features, dim=1)  # (B, K, N, T)
        return features


class CausalBandSequenceRNN(nn.Module):
    """Causal RNN processing for band and sequence modeling."""
    
    def __init__(self, input_dim, hidden_dim, use_cumulative_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_cumulative_norm = use_cumulative_norm
        
        if use_cumulative_norm:
            self.norm = CumulativeLayerNorm(input_dim)
        else:
            self.norm = GroupNorm(1, input_dim)
        
        # Unidirectional LSTM for causal processing
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, h_state=None, c_state=None, norm_stats=None):
        """
        Args:
            x: (B, T, C) or (B, C, T)
            h_state: (1, B, H) hidden state
            c_state: (1, B, H) cell state
            norm_stats: tuple of (running_mean, running_var, running_count)
        Returns:
            output: (B, T, C) or (B, C, T)
            new_h_state: updated hidden state
            new_c_state: updated cell state
            new_norm_stats: updated normalization statistics
        """
        input_format = x.shape
        if len(x.shape) == 3 and x.size(1) == self.input_dim:
            # (B, C, T) -> (B, T, C)
            x = x.transpose(1, 2)
            transpose_back = True
        else:
            transpose_back = False
        
        residual = x
        
        # Normalization
        if self.use_cumulative_norm and norm_stats is not None:
            x, new_mean, new_var, new_count = self.norm(x, *norm_stats)
            new_norm_stats = (new_mean, new_var, new_count)
        else:
            if not self.use_cumulative_norm:
                x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
            x = self.norm(x)
            if not self.use_cumulative_norm:
                x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
            new_norm_stats = None
        
        # LSTM
        if h_state is not None and c_state is not None:
            x, (new_h_state, new_c_state) = self.lstm(x, (h_state, c_state))
        else:
            x, (new_h_state, new_c_state) = self.lstm(x)
        
        # FC
        x = self.fc(x)
        
        # Residual
        x = x + residual
        
        if transpose_back:
            x = x.transpose(1, 2)
        
        return x, new_h_state, new_c_state, new_norm_stats


class CausalBandRNN(nn.Module):
    """RNN across bands (non-causal in band dimension)."""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.norm = GroupNorm(1, input_dim)
        # Bidirectional for band dimension (non-causal)
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, K, N, T) band features
        Returns:
            output: (B, K, N, T)
        """
        B, K, N, T = x.shape
        residual = x
        
        # Normalize (B, K, N, T)
        x = self.norm(x.view(B * K, N, T)).view(B, K, N, T)
        
        # Process across bands for each time step
        x = x.permute(0, 3, 1, 2)  # (B, T, K, N)
        x = x.reshape(B * T, K, N)  # (B*T, K, N)
        
        x, _ = self.lstm(x)  # (B*T, K, H)
        x = self.fc(x)  # (B*T, K, N)
        
        x = x.view(B, T, K, N)
        x = x.permute(0, 2, 3, 1)  # (B, K, N, T)
        
        # Residual
        x = x + residual
        
        return x


class MaskEstimationModule(nn.Module):
    """Estimate complex-valued masks for each band."""
    
    def __init__(self, band_specs, feature_dim, hidden_dim):
        super().__init__()
        self.band_specs = band_specs
        self.num_bands = len(band_specs)
        self.feature_dim = feature_dim
        
        self.band_norms = nn.ModuleList()
        self.band_mlps = nn.ModuleList()
        
        for start_f, end_f in band_specs:
            bandwidth = end_f - start_f
            self.band_norms.append(nn.LayerNorm(feature_dim))
            
            # MLP with one hidden layer
            self.band_mlps.append(nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.GLU(dim=-1)  # Splits hidden_dim in half and applies gating
            ))
            
            # Output layer for complex mask (real and imag)
            self.band_mlps[-1].add_module('output', nn.Linear(hidden_dim // 2, bandwidth * 2))
    
    def forward(self, x, mixture_spec):
        """
        Args:
            x: (B, K, N, T) band features
            mixture_spec: (B, F, T, 2) complex spectrogram
        Returns:
            separated: (B, F, T, 2) separated complex spectrogram
        """
        B, K, N, T = x.shape
        F = mixture_spec.size(1)
        
        separated = torch.zeros_like(mixture_spec)
        
        for i, (start_f, end_f) in enumerate(self.band_specs):
            band_feat = x[:, i, :, :]  # (B, N, T)
            band_feat = band_feat.transpose(1, 2)  # (B, T, N)
            
            # Normalize
            band_feat = self.band_norms[i](band_feat)  # (B, T, N)
            
            # MLP to get mask
            mask = self.band_mlps[i](band_feat)  # (B, T, Gi*2)
            
            # Reshape mask
            bandwidth = end_f - start_f
            mask = mask.view(B, T, bandwidth, 2)  # (B, T, Gi, 2)
            mask = mask.transpose(1, 2)  # (B, Gi, T, 2)
            
            # Apply mask
            band_mixture = mixture_spec[:, start_f:end_f, :, :]  # (B, Gi, T, 2)
            separated[:, start_f:end_f, :, :] = band_mixture * mask
        
        return separated


class CausalBSRNN(nn.Module):
    """Causal Band-Split RNN for real-time source separation."""
    
    def __init__(self, n_fft=1024, hop_length=512, band_specs=None, 
                 feature_dim=128, num_repeat=12, hidden_dim=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_freq_bins = n_fft // 2 + 1
        self.feature_dim = feature_dim
        self.num_repeat = num_repeat
        self.hidden_dim = hidden_dim
        
        # Default band specs for vocals (from paper - V7 config)
        if band_specs is None:
            band_specs = self._create_vocal_band_specs()
        self.band_specs = band_specs
        self.num_bands = len(band_specs)
        
        # Modules
        self.band_split = CausalBandSplitModule(band_specs, feature_dim)
        
        # Interleaved sequence and band RNNs
        self.sequence_rnns = nn.ModuleList()
        self.band_rnns = nn.ModuleList()
        
        for _ in range(num_repeat):
            self.sequence_rnns.append(
                CausalBandSequenceRNN(feature_dim, hidden_dim, use_cumulative_norm=True)
            )
            self.band_rnns.append(
                CausalBandRNN(feature_dim, hidden_dim)
            )
        
        self.mask_estimation = MaskEstimationModule(band_specs, feature_dim, hidden_dim * 4)
        
        # Window for STFT
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def _create_vocal_band_specs(self):
        """Create band specifications for vocals (V7 from paper)."""
        band_specs = []
        
        # Below 1k Hz: 100 Hz bandwidth
        for i in range(0, 1000, 100):
            start_bin = int(i * self.num_freq_bins / (44100 / 2))
            end_bin = int((i + 100) * self.num_freq_bins / (44100 / 2))
            band_specs.append((start_bin, end_bin))
        
        # 1k - 4k Hz: 250 Hz bandwidth
        for i in range(1000, 4000, 250):
            start_bin = int(i * self.num_freq_bins / (44100 / 2))
            end_bin = int((i + 250) * self.num_freq_bins / (44100 / 2))
            band_specs.append((start_bin, end_bin))
        
        # 4k - 8k Hz: 500 Hz bandwidth
        for i in range(4000, 8000, 500):
            start_bin = int(i * self.num_freq_bins / (44100 / 2))
            end_bin = int((i + 500) * self.num_freq_bins / (44100 / 2))
            band_specs.append((start_bin, end_bin))
        
        # 8k - 16k Hz: 1k Hz bandwidth
        for i in range(8000, 16000, 1000):
            start_bin = int(i * self.num_freq_bins / (44100 / 2))
            end_bin = int((i + 1000) * self.num_freq_bins / (44100 / 2))
            band_specs.append((start_bin, end_bin))
        
        # 16k - 20k Hz: 2k Hz bandwidth
        start_bin = int(16000 * self.num_freq_bins / (44100 / 2))
        end_bin = int(20000 * self.num_freq_bins / (44100 / 2))
        band_specs.append((start_bin, end_bin))
        
        # Above 20k Hz: rest
        band_specs.append((end_bin, self.num_freq_bins))
        
        return band_specs
    
    def forward(self, mixture, states=None):
        """
        Args:
            mixture: (B, F, T, 2) complex spectrogram
            states: dict of RNN states for streaming
        Returns:
            separated: (B, F, T, 2) separated complex spectrogram
            new_states: updated states for streaming
        """
        # Band split
        features = self.band_split(mixture)  # (B, K, N, T)
        
        if states is None:
            states = self._init_states(mixture.size(0), mixture.device)
        
        new_states = {}
        
        # Interleaved sequence and band modeling
        for i in range(self.num_repeat):
            # Sequence RNN for each band
            B, K, N, T = features.shape
            seq_output = []
            
            for k in range(K):
                band_feat = features[:, k, :, :]  # (B, N, T)
                
                h_key = f'seq_{i}_band_{k}_h'
                c_key = f'seq_{i}_band_{k}_c'
                norm_key = f'seq_{i}_band_{k}_norm'
                
                out, new_h, new_c, new_norm = self.sequence_rnns[i](
                    band_feat,
                    states.get(h_key),
                    states.get(c_key),
                    states.get(norm_key)
                )
                
                new_states[h_key] = new_h
                new_states[c_key] = new_c
                if new_norm is not None:
                    new_states[norm_key] = new_norm
                
                seq_output.append(out)
            
            features = torch.stack(seq_output, dim=1)  # (B, K, N, T)
            
            # Band RNN
            features = self.band_rnns[i](features)  # (B, K, N, T)
        
        # Mask estimation
        separated = self.mask_estimation(features, mixture)
        
        return separated, new_states
    
    def _init_states(self, batch_size, device):
        """Initialize RNN states."""
        states = {}
        for i in range(self.num_repeat):
            for k in range(self.num_bands):
                h_key = f'seq_{i}_band_{k}_h'
                c_key = f'seq_{i}_band_{k}_c'
                norm_key = f'seq_{i}_band_{k}_norm'
                
                states[h_key] = torch.zeros(1, batch_size, self.hidden_dim, device=device)
                states[c_key] = torch.zeros(1, batch_size, self.hidden_dim, device=device)
                states[norm_key] = (
                    torch.zeros(batch_size, self.feature_dim, device=device),
                    torch.zeros(batch_size, self.feature_dim, device=device),
                    torch.zeros(batch_size, device=device)
                )
        
        return states