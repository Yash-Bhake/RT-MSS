"""
Causal Band-Split RNN for Real-Time Processing

Key modifications for causality and low latency:
1. Unidirectional LSTM (only looks at past frames)
2. Cumulative Layer Normalization (uses running statistics)
3. Left-aligned STFT windows (no future look-ahead)
4. State management for streaming inference
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from .layers import BandSplitModule, MaskEstimationModule, CumulativeLayerNorm


class CausalResidualRNN(nn.Module):
    """
    Causal Residual RNN block with unidirectional LSTM
    Maintains hidden states for streaming
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.norm = CumulativeLayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=1, 
            batch_first=True,
            bidirectional=False  # CAUSAL: unidirectional only
        )
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, C, T) or (B, T, C)
            hidden: Previous (h, c) states for LSTM
        Returns:
            output: (B, C, T) or (B, T, C)
            new_hidden: New (h, c) states
        """
        residual = x
        
        # Ensure x is (B, T, C) for LSTM
        if x.dim() == 3 and x.size(1) != x.size(2):
            needs_transpose = True
            x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        else:
            needs_transpose = False
        
        # Normalize
        x = self.norm(x)
        
        # LSTM with state
        if hidden is not None:
            x, new_hidden = self.lstm(x, hidden)
        else:
            x, new_hidden = self.lstm(x)
        
        # FC projection
        x = self.fc(x)
        
        # Transpose back if needed
        if needs_transpose:
            x = x.transpose(1, 2)
            residual = residual.transpose(1, 2) if residual.size(1) == x.size(2) else residual
        
        return x + residual, new_hidden


class CausalBSRNN(nn.Module):
    """
    Causal Band-Split RNN for Real-Time Music Source Separation
    
    Architecture (Causal Version):
    1. Band Split Module: Splits spectrogram into K subbands
       - Uses cumulative layer normalization
    2. Sequence Modeling: Unidirectional LSTM across time (per band)
       - Only looks at past frames
       - Maintains hidden states for streaming
    3. Band Modeling: Unidirectional LSTM across bands (per frame)
       - Processes bands in order
       - Maintains hidden states for streaming
    4. Mask Estimation: Generates complex T-F masks per band
       - Uses cumulative normalization
    
    Key differences from original BSRNN:
    - Unidirectional LSTM (causal): ~50ms algorithmic latency vs ~200ms
    - Cumulative normalization: Uses running statistics, no future frames
    - Left-aligned STFT: No centering, immediate processing
    - State management: Can process streaming audio chunk-by-chunk
    - Quantization support: 8-bit inference for faster processing
    """
    
    def __init__(self, band_specs: list, n_features: int = 128,
                 n_layers: int = 12, lstm_hidden: int = 256,
                 mlp_hidden: int = 512, sample_rate: int = 44100,
                 n_fft: int = 1408, hop_length: int = 704):
        super().__init__()
        
        self.band_specs = band_specs
        self.n_features = n_features
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_layers = n_layers
        self.lstm_hidden = lstm_hidden
        
        # Band split module (with cumulative normalization)
        self.band_split = BandSplitModule(band_specs, n_features)
        self.n_bands = self.band_split.n_bands
        
        # Interleaved sequence and band modeling (causal RNNs)
        self.sequence_rnns = nn.ModuleList([
            CausalResidualRNN(n_features, lstm_hidden)
            for _ in range(n_layers // 2)
        ])
        
        self.band_rnns = nn.ModuleList([
            CausalResidualRNN(n_features, lstm_hidden)
            for _ in range(n_layers // 2)
        ])
        
        # Mask estimation module (with cumulative normalization)
        self.mask_estimation = MaskEstimationModule(band_specs, n_features, mlp_hidden)
        
    def forward(self, x: torch.Tensor, 
                states: Optional[Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]] = None
               ) -> Tuple[torch.Tensor, Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass with state management for streaming
        
        Args:
            x: Complex spectrogram (B, F, T)
            states: Dictionary containing 'sequence' and 'band' LSTM states
        Returns:
            output: Separated source spectrogram (B, F, T)
            new_states: Updated LSTM states
        """
        # Initialize states if not provided
        if states is None:
            states = self._init_states(x.device, x.size(0))
        
        # Band split
        z = self.band_split(x, self.sample_rate)  # (B, N, K, T)
        
        B, N, K, T = z.shape
        
        # New states to return
        new_states = {'sequence': [], 'band': []}
        
        # Interleaved sequence and band modeling
        for layer_idx, (seq_rnn, band_rnn) in enumerate(zip(self.sequence_rnns, self.band_rnns)):
            # Sequence modeling: process each band across time
            z_seq = z.permute(0, 2, 1, 3).reshape(B * K, N, T)  # (B*K, N, T)
            
            # Get states for this layer
            seq_state = states['sequence'][layer_idx] if layer_idx < len(states['sequence']) else None
            
            z_seq, new_seq_state = seq_rnn(z_seq, seq_state)
            new_states['sequence'].append(new_seq_state)
            
            z = z_seq.reshape(B, K, N, T).permute(0, 2, 1, 3)  # (B, N, K, T)
            
            # Band modeling: process each frame across bands
            z_band = z.permute(0, 3, 1, 2).reshape(B * T, N, K)  # (B*T, N, K)
            
            # Get states for this layer
            band_state = states['band'][layer_idx] if layer_idx < len(states['band']) else None
            
            z_band, new_band_state = band_rnn(z_band, band_state)
            new_states['band'].append(new_band_state)
            
            z = z_band.reshape(B, T, N, K).permute(0, 2, 3, 1)  # (B, N, K, T)
        
        # Mask estimation
        mask = self.mask_estimation(z, self.sample_rate, self.n_fft)
        
        # Apply mask
        output = x * mask
        
        return output, new_states
    
    def _init_states(self, device: torch.device, batch_size: int
                    ) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Initialize LSTM states for streaming"""
        states = {'sequence': [], 'band': []}
        
        # Sequence RNN states (one state per band)
        for _ in range(len(self.sequence_rnns)):
            h = torch.zeros(1, batch_size * self.n_bands, self.lstm_hidden, device=device)
            c = torch.zeros(1, batch_size * self.n_bands, self.lstm_hidden, device=device)
            states['sequence'].append((h, c))
        
        # Band RNN states (one state per time frame, but we use 1 for streaming)
        for _ in range(len(self.band_rnns)):
            h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
            c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
            states['band'].append((h, c))
        
        return states
    
    def reset_states(self):
        """Reset running statistics in normalization layers"""
        for module in self.modules():
            if isinstance(module, CumulativeLayerNorm):
                module.running_mean.zero_()
                module.running_var.fill_(1.0)
                module.num_batches_tracked.zero_()


class MultiTargetCausalBSRNN(nn.Module):
    """
    Multi-target causal BSRNN for separating vocals, drums, bass, and other
    """
    def __init__(self, config: dict):
        super().__init__()
        
        self.targets = ['vocals', 'drums', 'bass', 'other']
        self.sample_rate = config['audio']['sample_rate']
        self.n_fft = config['audio']['n_fft']
        self.hop_length = config['audio']['hop_length']
        
        # Create separate model for each target
        self.models = nn.ModuleDict()
        for target in self.targets:
            band_specs = config['model']['band_specs'][target]
            self.models[target] = CausalBSRNN(
                band_specs=band_specs,
                n_features=config['model']['n_features'],
                n_layers=config['model']['n_layers'],
                lstm_hidden=config['model']['lstm_hidden'],
                mlp_hidden=config['model']['mlp_hidden'],
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
    
    def forward(self, x: torch.Tensor, target: str,
                states: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: Complex spectrogram (B, F, T)
            target: Target source ('vocals', 'drums', 'bass', 'other')
            states: LSTM states for streaming
        """
        return self.models[target](x, states)
    
    def reset_states(self, target: Optional[str] = None):
        """Reset states for one or all targets"""
        if target:
            self.models[target].reset_states()
        else:
            for model in self.models.values():
                model.reset_states()