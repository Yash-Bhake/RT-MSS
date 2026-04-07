import torch
import torch.nn as nn
import numpy as np
from .normalization import CumulativeLayerNorm, GroupNorm

class CausalBandSplitModule(nn.Module):
    def __init__(self, band_specs, feature_dim):
        super().__init__()
        self.band_specs = band_specs
        self.num_bands = len(band_specs)
        self.feature_dim = feature_dim
        self.band_norms = nn.ModuleList()
        self.band_fcs = nn.ModuleList()
        for start_f, end_f in band_specs:
            bandwidth = end_f - start_f
            self.band_norms.append(nn.LayerNorm(bandwidth * 2))
            self.band_fcs.append(nn.Linear(bandwidth * 2, feature_dim))
    
    def forward(self, x):
        B, F, T, _ = x.shape
        band_features = []
        for i, (start_f, end_f) in enumerate(self.band_specs):
            band = x[:, start_f:end_f, :, :]
            band = band.reshape(B, -1, T).transpose(1, 2)
            band = self.band_norms[i](band)
            band = self.band_fcs[i](band)
            band_features.append(band.transpose(1, 2))
        features = torch.stack(band_features, dim=1)
        return features

class CausalBandSequenceRNN(nn.Module):
    # ... [Keep your existing implementation, it is correct] ...
    def __init__(self, input_dim, hidden_dim, use_cumulative_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_cumulative_norm = use_cumulative_norm
        if use_cumulative_norm:
            self.norm = CumulativeLayerNorm(input_dim)
        else:
            self.norm = GroupNorm(1, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        
        nn.init.xavier_uniform_(self.fc.weight, gain=1.0)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x, h_state=None, c_state=None, norm_stats=None):
        residual = x
        if self.use_cumulative_norm:
            if norm_stats is not None:
                x, new_mean, new_var, new_count = self.norm(x, *norm_stats)
                new_norm_stats = (new_mean, new_var, new_count)
            else:
                x, _, _, _ = self.norm(x)
                new_norm_stats = None
        else:
            # Fallback (mostly for debugging)
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
            new_norm_stats = None

        if h_state is not None:
            x, (new_h, new_c) = self.lstm(x, (h_state, c_state))
        else:
            x, (new_h, new_c) = self.lstm(x)
        
        x = self.fc(x)
        return x + residual, new_h, new_c, new_norm_stats

class CausalBandRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # FIX 1: Use LayerNorm instead of GroupNorm to avoid Time-axis averaging
        # We apply this per timestep, so it preserves causality.
        self.norm = nn.LayerNorm(input_dim)
        
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
        # Initialization
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.orthogonal_(self.lstm.weight_ih_l0_reverse)
        nn.init.orthogonal_(self.lstm.weight_hh_l0_reverse)
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        nn.init.zeros_(self.lstm.bias_ih_l0_reverse)
        nn.init.zeros_(self.lstm.bias_hh_l0_reverse)
        
        # FIX 2: Increase gain from 0.01 to 0.1 to allow gradient flow
        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        # x: (B, K, N, T)
        B, K, N, T = x.shape
        residual = x
        
        # Permute to (B, T, K, N) to apply LayerNorm on the last dim (N)
        x = x.permute(0, 3, 1, 2) 
        x = self.norm(x)
        
        # Reshape for Band-RNN: (B*T, K, N)
        # We process K bands for every time step independently
        x = x.reshape(B * T, K, N)
        
        x, _ = self.lstm(x)
        x = self.fc(x)
        
        # Reshape back: (B, T, K, N) -> (B, K, N, T)
        x = x.view(B, T, K, N).permute(0, 2, 3, 1)
        
        return x + residual

class MaskEstimationModule(nn.Module):
    def __init__(self, band_specs, feature_dim, hidden_dim):
        super().__init__()
        self.band_specs = band_specs
        self.num_bands = len(band_specs)
        self.band_norms = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in band_specs])
        self.band_mlps = nn.ModuleList()
        
        for start_f, end_f in band_specs:
            bandwidth = end_f - start_f
            # MLP with GLU
            mlp = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.GLU(dim=-1),
                nn.Linear(hidden_dim // 2, bandwidth * 2)
            )
            # Initialization for identity mask
            nn.init.xavier_uniform_(mlp[-1].weight, gain=1.0) #
            nn.init.constant_(mlp[-1].bias, 0.0)
            self.band_mlps.append(mlp)

    def forward(self, x, mixture_spec):
        B, K, N, T = x.shape
        separated = torch.zeros_like(mixture_spec)
        
        for i, (start_f, end_f) in enumerate(self.band_specs):
            band_feat = x[:, i, :, :].transpose(1, 2) # (B, T, N)
            band_feat = self.band_norms[i](band_feat)
            mask = self.band_mlps[i](band_feat) # (B, T, bandwidth*2)
            
            bandwidth = end_f - start_f
            mask = mask.view(B, T, bandwidth, 2).transpose(1, 2) # (B, bandwidth, T, 2)
            mask = torch.sigmoid(mask)
            
            separated[:, start_f:end_f, :, :] = mixture_spec[:, start_f:end_f, :, :] * mask
        
        return separated
    
class CausalBSRNN(nn.Module):
    """
    Causal Band-Split RNN for real-time source separation.
    """
    def __init__(self, n_fft=1024, hop_length=512, band_specs=None, 
                 feature_dim=128, num_repeat=6, hidden_dim=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_freq_bins = n_fft // 2 + 1
        self.feature_dim = feature_dim
        self.num_repeat = num_repeat
        self.hidden_dim = hidden_dim
        
        if band_specs is None:
            band_specs = self._create_vocal_band_specs()
        self.band_specs = band_specs
        self.num_bands = len(band_specs)
        
        self.band_split = CausalBandSplitModule(band_specs, feature_dim)
        
        # Sequential components
        self.sequence_rnns = nn.ModuleList([
            CausalBandSequenceRNN(feature_dim, hidden_dim) for _ in range(num_repeat)
        ])
        self.band_rnns = nn.ModuleList([
            CausalBandRNN(feature_dim, hidden_dim) for _ in range(num_repeat)
        ])
        
        self.mask_estimation = MaskEstimationModule(band_specs, feature_dim, hidden_dim * 4)
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def _create_vocal_band_specs(self):
        """Create frequency band specifications optimized for vocals."""
        band_specs = []
        # Fine resolution in vocal range (0-4000 Hz)
        for i in range(0, 1000, 100):
            start_bin = int(i * self.num_freq_bins / (44100 / 2))
            end_bin = int((i + 100) * self.num_freq_bins / (44100 / 2))
            band_specs.append((start_bin, end_bin))
        for i in range(1000, 4000, 250):
            start_bin = int(i * self.num_freq_bins / (44100 / 2))
            end_bin = int((i + 250) * self.num_freq_bins / (44100 / 2))
            band_specs.append((start_bin, end_bin))
        # Coarser resolution in higher frequencies
        for i in range(4000, 8000, 500):
            start_bin = int(i * self.num_freq_bins / (44100 / 2))
            end_bin = int((i + 500) * self.num_freq_bins / (44100 / 2))
            band_specs.append((start_bin, end_bin))
        for i in range(8000, 16000, 1000):
            start_bin = int(i * self.num_freq_bins / (44100 / 2))
            end_bin = int((i + 1000) * self.num_freq_bins / (44100 / 2))
            band_specs.append((start_bin, end_bin))
        start_bin = int(16000 * self.num_freq_bins / (44100 / 2))
        end_bin = int(20000 * self.num_freq_bins / (44100 / 2))
        band_specs.append((start_bin, end_bin))
        band_specs.append((end_bin, self.num_freq_bins))
        return band_specs
    
    def forward(self, mixture_spec, states=None):
        # mixture_spec: (B, F, T, 2)
        features = self.band_split(mixture_spec) 
        B, K, N, T = features.shape
        
        # NEW: Determine if we are in 'Sequence' mode (training) or 'Step' mode (streaming)
        # If states is None, we are training and reset states for this batch.
        is_streaming = (states is not None)
        new_states = {} if is_streaming else None

        for i in range(self.num_repeat):
            # Reshape to process all bands in parallel
            x = features.reshape(B * K, N, T).transpose(1, 2) # (B*K, T, N)
            
            if not is_streaming:
                # --- TRAINING PATH (FAST) ---
                # LSTM handles context internally across the whole T
                x, _, _, _ = self.sequence_rnns[i](x) 
            else:
                # --- STREAMING PATH (CONTEXT-AWARE) ---
                # Extract and flatten states for the vectorized batch B*K
                h_in = states[f'layer_{i}_h'] # Expected shape: (1, B*K, hidden_dim)
                c_in = states[f'layer_{i}_c']
                n_in = states[f'layer_{i}_norm'] # (mean, var, count) for B*K

                x, h_out, c_out, n_out = self.sequence_rnns[i](x, h_in, c_in, n_in)
                
                # Save updated states for the next chunk
                new_states[f'layer_{i}_h'] = h_out
                new_states[f'layer_{i}_c'] = c_out
                new_states[f'layer_{i}_norm'] = n_out

            features = x.transpose(1, 2).reshape(B, K, N, T)
            features = self.band_rnns[i](features)
            
        separated_spec = self.mask_estimation(features, mixture_spec)
        return separated_spec, new_states
    
    def _init_states(self, batch_size, device):
        """
        Initialize RNN states for vectorized streaming inference.
        Shapes are (1, B*K, hidden_dim) to match the parallelized forward pass.
        """
        states = {}
        # Total parallel sequences = batch_size * number of bands
        num_parallel = batch_size * self.num_bands
        
        for i in range(self.num_repeat):
            # State keys for the Sequence RNNs
            states[f'layer_{i}_h'] = torch.zeros(1, num_parallel, self.hidden_dim, device=device)
            states[f'layer_{i}_c'] = torch.zeros(1, num_parallel, self.hidden_dim, device=device)
            
            # Stats for the Cumulative Normalization
            # Needs to be (B*K, feature_dim)
            states[f'layer_{i}_norm'] = (
                torch.zeros(num_parallel, self.feature_dim, device=device), # mean
                torch.zeros(num_parallel, self.feature_dim, device=device), # var
                torch.zeros(num_parallel, 1, device=device)                # count
            )
        return states