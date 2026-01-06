"""
Real-time processor with quantization and state management
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Optional, List
from .buffer import StreamingSTFT


class RealtimeProcessor:
    """
    Real-time music source separation processor
    
    Features:
    - 8-bit dynamic quantization for faster inference
    - State management for streaming
    - Circular buffering
    - Latency tracking
    """
    
    def __init__(self, model: nn.Module, config: dict, 
                 target: str = 'vocals', device: str = 'cpu'):
        """
        Args:
            model: Multi-target causal BSRNN model
            config: Configuration dictionary
            target: Target source to separate
            device: Device for inference
        """
        self.model = model
        self.config = config
        self.target = target
        self.device = torch.device(device)
        
        # Audio parameters
        self.sample_rate = config['audio']['sample_rate']
        self.n_fft = config['audio']['n_fft']
        self.hop_length = config['audio']['hop_length']
        self.n_channels = 2  # Stereo
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Apply quantization for faster inference
        if config['inference']['use_quantization']:
            self.quantize_model()
        
        # STFT processor
        self.stft_processor = StreamingSTFT(
            self.n_fft, 
            self.hop_length,
            self.sample_rate,
            self.n_channels
        )
        
        # Model states for streaming
        self.states = None
        
        # Performance tracking
        self.processing_times = []
        self.rtf_values = []
        
    def quantize_model(self):
        """Apply dynamic quantization to model"""
        print("Applying 8-bit dynamic quantization...")
        
        # Quantize Linear and LSTM layers
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM},
            dtype=torch.qint8
        )
        
        print("Quantization complete")
    
    def process_chunk(self, audio_chunk: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Process a single audio chunk in real-time
        
        Args:
            audio_chunk: (n_channels, n_samples) audio tensor
        Returns:
            Separated audio (n_channels, n_samples) or None if buffering
        """
        if audio_chunk.size(0) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {audio_chunk.size(0)}")
        
        # Move to device
        audio_chunk = audio_chunk.to(self.device)
        
        # STFT
        start_time = time.time()
        spec, n_frames = self.stft_processor.process(audio_chunk)
        
        if spec is None:
            return None  # Still buffering
        
        # Add batch dimension
        spec = spec.unsqueeze(0)  # (1, C, F, T)
        
        # Process each channel
        separated_specs = []
        for ch in range(self.n_channels):
            ch_spec = spec[:, ch, :, :]  # (1, F, T)
            
            # Separate
            with torch.no_grad():
                sep_spec, self.states = self.model(ch_spec, self.target, self.states)
            
            separated_specs.append(sep_spec)
        
        # Concatenate channels
        separated_spec = torch.stack(separated_specs, dim=1)  # (1, C, F, T)
        separated_spec = separated_spec.squeeze(0)  # (C, F, T)
        
        # Inverse STFT
        separated_audio = self.stft_processor.inverse(separated_spec)
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Calculate RTF (Real-Time Factor)
        audio_duration = audio_chunk.size(1) / self.sample_rate
        rtf = processing_time / audio_duration
        self.rtf_values.append(rtf)
        
        return separated_audio
    
    def process_stream(self, audio_stream: torch.Tensor, 
                      chunk_size: Optional[int] = None) -> torch.Tensor:
        """
        Process continuous audio stream
        
        Args:
            audio_stream: (n_channels, n_samples)
            chunk_size: Size of each chunk (default: hop_length)
        Returns:
            Separated audio (n_channels, n_samples)
        """
        if chunk_size is None:
            chunk_size = self.hop_length
        
        n_samples = audio_stream.size(1)
        output_chunks = []
        
        # Process in chunks
        for i in range(0, n_samples, chunk_size):
            chunk = audio_stream[:, i:i + chunk_size]
            
            # Pad last chunk if needed
            if chunk.size(1) < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.size(1)))
            
            output = self.process_chunk(chunk)
            
            if output is not None:
                output_chunks.append(output)
        
        # Concatenate outputs
        if output_chunks:
            output_audio = torch.cat(output_chunks, dim=1)
            # Trim to original length
            output_audio = output_audio[:, :n_samples]
            return output_audio
        else:
            return torch.zeros_like(audio_stream)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get processing statistics"""
        if not self.processing_times:
            return {}
        
        return {
            'mean_processing_time_ms': sum(self.processing_times) / len(self.processing_times) * 1000,
            'max_processing_time_ms': max(self.processing_times) * 1000,
            'mean_rtf': sum(self.rtf_values) / len(self.rtf_values),
            'max_rtf': max(self.rtf_values),
            'algorithmic_latency_ms': self.stft_processor.get_latency_ms(),
            'n_chunks_processed': len(self.processing_times)
        }
    
    def reset(self):
        """Reset processor state"""
        self.states = None
        self.stft_processor.reset()
        self.model.reset_states(self.target)
        self.processing_times = []
        self.rtf_values = []
    
    def get_latency_ms(self) -> float:
        """Get total latency in milliseconds"""
        algorithmic = self.stft_processor.get_latency_ms()
        
        if self.processing_times:
            processing = sum(self.processing_times) / len(self.processing_times) * 1000
        else:
            processing = 0
        
        return algorithmic + processing


class MultiTargetRealtimeProcessor:
    """
    Real-time processor for multiple targets simultaneously
    """
    
    def __init__(self, model: nn.Module, config: dict, 
                 targets: List[str] = None, device: str = 'cpu'):
        """
        Args:
            model: Multi-target causal BSRNN model
            config: Configuration dictionary
            targets: List of targets to separate (default: all 4)
            device: Device for inference
        """
        if targets is None:
            targets = ['vocals', 'drums', 'bass', 'other']
        
        self.targets = targets
        self.processors = {}
        
        # Create processor for each target
        for target in targets:
            self.processors[target] = RealtimeProcessor(
                model, config, target, device
            )
    
    def process_chunk(self, audio_chunk: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process chunk for all targets
        
        Args:
            audio_chunk: (n_channels, n_samples)
        Returns:
            Dictionary of separated audio for each target
        """
        outputs = {}
        for target in self.targets:
            output = self.processors[target].process_chunk(audio_chunk.clone())
            if output is not None:
                outputs[target] = output
        
        return outputs
    
    def process_stream(self, audio_stream: torch.Tensor, 
                      chunk_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Process stream for all targets
        
        Args:
            audio_stream: (n_channels, n_samples)
        Returns:
            Dictionary of separated audio for each target
        """
        outputs = {}
        for target in self.targets:
            outputs[target] = self.processors[target].process_stream(
                audio_stream.clone(), chunk_size
            )
        
        return outputs
    
    def get_statistics(self, target: Optional[str] = None) -> Dict:
        """Get statistics for one or all targets"""
        if target:
            return {target: self.processors[target].get_statistics()}
        else:
            return {t: p.get_statistics() for t, p in self.processors.items()}
    
    def reset(self, target: Optional[str] = None):
        """Reset one or all processors"""
        if target:
            self.processors[target].reset()
        else:
            for processor in self.processors.values():
                processor.reset()