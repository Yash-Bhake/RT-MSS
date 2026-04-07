import torch
import torch.nn as nn
import numpy as np
from .normalization import CumulativeLayerNorm, GroupNorm

# ... (Previous classes: CausalBandSplitModule, CausalBandSequenceRNN, CausalBandRNN, MaskEstimationModule remain unchanged) ...
# I will output the Full CausalBSRNN class to ensure you have the exact placement.

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
    def forward(self, x, h_state=None, c_state=None, norm_stats=None):
        input_format = x.shape
        if len(x.shape) == 3 and x.size(1) == self.input_dim:
            x = x.transpose(1, 2)
            transpose_back = True
        else:
            transpose_back = False
        residual = x
        if self.use_cumulative_norm and norm_stats is not None:
            x, new_mean, new_var, new_count = self.norm(x, *norm_stats)
            new_norm_stats = (new_mean, new_var, new_count)
        else:
            if not self.use_cumulative_norm: x = x.transpose(1, 2)
            x = self.norm(x)
            if not self.use_cumulative_norm: x = x.transpose(1, 2)
            new_norm_stats = None
        if h_state is not None and c_state is not None:
            x, (new_h_state, new_c_state) = self.lstm(x, (h_state, c_state))
        else:
            x, (new_h_state, new_c_state) = self.lstm(x)
        x = self.fc(x)
        x = x + residual
        if transpose_back: x = x.transpose(1, 2)
        return x, new_h_state, new_c_state, new_norm_stats

class CausalBandRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.norm = GroupNorm(1, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        B, K, N, T = x.shape
        residual = x
        x = self.norm(x.view(B * K, N, T)).view(B, K, N, T)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B * T, K, N)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x.view(B, T, K, N)
        x = x.permute(0, 2, 3, 1)
        x = x + residual
        return x

class MaskEstimationModule(nn.Module):
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
            self.band_mlps.append(nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.GLU(dim=-1)
            ))
            self.band_mlps[-1].add_module('output', nn.Linear(hidden_dim // 2, bandwidth * 2))
    def forward(self, x, mixture_spec):
        separated = torch.zeros_like(mixture_spec)
        for i, (start_f, end_f) in enumerate(self.band_specs):
            band_feat = x[:, i, :, :]
            band_feat = band_feat.transpose(1, 2)
            band_feat = self.band_norms[i](band_feat)
            mask = self.band_mlps[i](band_feat)
            bandwidth = end_f - start_f
            mask = mask.view(x.shape[0], x.shape[3], bandwidth, 2)
            mask = mask.transpose(1, 2)
            band_mixture = mixture_spec[:, start_f:end_f, :, :]
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
        
        if band_specs is None:
            band_specs = self._create_vocal_band_specs()
        self.band_specs = band_specs
        self.num_bands = len(band_specs)
        
        self.band_split = CausalBandSplitModule(band_specs, feature_dim)
        
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
        
        # --- FIX: Increased epsilon to 1e-4 ---
        self.register_buffer('window', torch.hann_window(n_fft) + 1e-4)
    
    def _create_vocal_band_specs(self):
        band_specs = []
        for i in range(0, 1000, 100):
            start_bin = int(i * self.num_freq_bins / (44100 / 2))
            end_bin = int((i + 100) * self.num_freq_bins / (44100 / 2))
            band_specs.append((start_bin, end_bin))
        for i in range(1000, 4000, 250):
            start_bin = int(i * self.num_freq_bins / (44100 / 2))
            end_bin = int((i + 250) * self.num_freq_bins / (44100 / 2))
            band_specs.append((start_bin, end_bin))
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
    
    def forward(self, mixture, states=None):
        features = self.band_split(mixture)
        
        if states is None:
            states = self._init_states(mixture.size(0), mixture.device)
        
        new_states = {}
        
        for i in range(self.num_repeat):
            B, K, N, T = features.shape
            seq_output = []
            
            for k in range(K):
                band_feat = features[:, k, :, :]
                
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
            
            features = torch.stack(seq_output, dim=1)
            features = self.band_rnns[i](features)
        
        separated = self.mask_estimation(features, mixture)
        return separated, new_states
    
    def _init_states(self, batch_size, device):
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
                    torch.zeros(batch_size, 1, device=device)
                )
        return states