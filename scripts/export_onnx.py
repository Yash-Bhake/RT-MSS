#!/usr/bin/env python3
"""Export Causal BSRNN model to ONNX format for real-time inference."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
from pathlib import Path

from src.model.causal_bsrnn import CausalBSRNN


class CausalBSRNNWrapper(torch.nn.Module):
    """Wrapper for ONNX export with stateful processing."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_bands = model.num_bands
        self.num_repeat = model.num_repeat
        self.feature_dim = model.feature_dim
        self.hidden_dim = model.hidden_dim
    
    def forward(self, mixture_spec, *state_tensors):
        """
        Forward pass with explicit state management for ONNX.
        
        Args:
            mixture_spec: (B, F, T, 2) complex spectrogram
            state_tensors: flattened list of state tensors
        
        Returns:
            separated_spec: (B, F, T, 2)
            updated_state_tensors: list of updated state tensors
        """
        # Reconstruct states from flat tensor list
        states = self._unflatten_states(state_tensors)
        
        # Forward pass
        separated_spec, new_states = self.model(mixture_spec, states)
        
        # Flatten states for ONNX output
        new_state_tensors = self._flatten_states(new_states)
        
        return (separated_spec, *new_state_tensors)
    
    def _unflatten_states(self, state_tensors):
        """Reconstruct state dict from flat tensor list."""
        if not state_tensors or state_tensors[0] is None:
            return None
        
        states = {}
        idx = 0
        
        for i in range(self.num_repeat):
            for k in range(self.num_bands):
                h_key = f'seq_{i}_band_{k}_h'
                c_key = f'seq_{i}_band_{k}_c'
                
                states[h_key] = state_tensors[idx]
                states[c_key] = state_tensors[idx + 1]
                
                # Normalization stats (mean, var, count)
                norm_key = f'seq_{i}_band_{k}_norm'
                states[norm_key] = (
                    state_tensors[idx + 2],
                    state_tensors[idx + 3],
                    state_tensors[idx + 4]
                )
                
                idx += 5
        
        return states
    
    def _flatten_states(self, states):
        """Flatten state dict to tensor list for ONNX."""
        state_list = []
        
        for i in range(self.num_repeat):
            for k in range(self.num_bands):
                h_key = f'seq_{i}_band_{k}_h'
                c_key = f'seq_{i}_band_{k}_c'
                norm_key = f'seq_{i}_band_{k}_norm'
                
                state_list.append(states[h_key])
                state_list.append(states[c_key])
                state_list.append(states[norm_key][0])
                state_list.append(states[norm_key][1])
                state_list.append(states[norm_key][2])
        
        return state_list
    
    def init_states(self, batch_size, device):
        """Initialize states for ONNX export."""
        state_list = []
        
        for i in range(self.num_repeat):
            for k in range(self.num_bands):
                # h_state
                state_list.append(torch.zeros(1, batch_size, self.hidden_dim, device=device))
                # c_state
                state_list.append(torch.zeros(1, batch_size, self.hidden_dim, device=device))
                # norm mean
                state_list.append(torch.zeros(batch_size, self.feature_dim, device=device))
                # norm var
                state_list.append(torch.zeros(batch_size, self.feature_dim, device=device))
                # norm count
                state_list.append(torch.zeros(batch_size, device=device))
        
        return state_list


def export_onnx(model_path, output_path, batch_size=1, chunk_length=256):
    """
    Export model to ONNX.
    
    Args:
        model_path: Path to PyTorch checkpoint
        output_path: Path to save ONNX model
        batch_size: Batch size for export
        chunk_length: Number of time frames per chunk
    """
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model config from checkpoint
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # Create model
    model = CausalBSRNN(
        n_fft=model_config.get('n_fft', 1024),
        hop_length=model_config.get('hop_length', 512),
        feature_dim=model_config.get('feature_dim', 128),
        num_repeat=model_config.get('num_repeat', 12),
        hidden_dim=model_config.get('hidden_dim', 256)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Wrap model
    wrapper = CausalBSRNNWrapper(model)
    wrapper.eval()
    
    # Create dummy inputs
    num_freq_bins = model.n_fft // 2 + 1
    dummy_mixture = torch.randn(batch_size, num_freq_bins, chunk_length, 2)
    dummy_states = wrapper.init_states(batch_size, 'cpu')
    
    # Input names
    input_names = ['mixture_spec']
    for i in range(len(dummy_states)):
        input_names.append(f'state_{i}')
    
    # Output names
    output_names = ['separated_spec']
    for i in range(len(dummy_states)):
        output_names.append(f'state_out_{i}')
    
    print(f"Exporting to ONNX with {len(input_names)} inputs and {len(output_names)} outputs")
    
    # Export
    torch.onnx.export(
        wrapper,
        (dummy_mixture, *dummy_states),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'mixture_spec': {0: 'batch', 2: 'time'},
            'separated_spec': {0: 'batch', 2: 'time'}
        },
        opset_version=14,
        do_constant_folding=True,
    )
    
    print(f"Model exported to {output_path}")
    
    # Verify export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")


def main():
    parser = argparse.ArgumentParser(description='Export Causal BSRNN to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='causal_bsrnn.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for export')
    parser.add_argument('--chunk_length', type=int, default=256,
                        help='Number of time frames per chunk')
    
    args = parser.parse_args()
    
    export_onnx(args.checkpoint, args.output, args.batch_size, args.chunk_length)


if __name__ == '__main__':
    main()