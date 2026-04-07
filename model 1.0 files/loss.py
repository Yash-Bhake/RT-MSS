import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, 
                 fft_sizes=[2048, 1024, 512], 
                 hop_sizes=[512, 256, 128], 
                 win_lengths=[2048, 1024, 512]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.windows = nn.ParameterDict()
        for i, win_length in enumerate(win_lengths):
            self.register_buffer(f'window_{i}', torch.hann_window(win_length))

    def stft(self, x, fft_size, hop_size, win_length, window):
        """
        For the Loss *Magnitude* calculation, center=True is actually fine/preferred 
        because magnitude is shift-invariant and it stabilizes the gradients.
        We only need center=False for the exact time-domain reconstruction in HybridLoss.
        """
        return torch.stft(
            x, 
            n_fft=fft_size, 
            hop_length=hop_size, 
            win_length=win_length, 
            window=window,
            center=True, 
            normalized=True,
            return_complex=True
        )

    def forward(self, pred_audio, target_audio):
        total_loss = 0.0
        for i, (fft_size, hop_size, win_length) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = getattr(self, f'window_{i}')
            pred_stft = self.stft(pred_audio, fft_size, hop_size, win_length, window)
            target_stft = self.stft(target_audio, fft_size, hop_size, win_length, window)
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            
            sc_loss = torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)
            mag_loss = F.l1_loss(torch.log(pred_mag + 1e-7), torch.log(target_mag + 1e-7))
            total_loss += sc_loss + mag_loss
        return total_loss / len(self.fft_sizes)

class HybridLoss(nn.Module):
    def __init__(self, n_fft=1024, hop_length=512, device='cuda', alpha_time=1.0, alpha_freq=1.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        self.alpha_time = alpha_time
        self.alpha_freq = alpha_freq
        self.stft_loss = MultiResolutionSTFTLoss()
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, pred_spec, target_audio, target_length=None):
        # 1. Convert Predicted Spectrogram to Audio
        # CRITICAL: This uses center=False + Clamped Window to match model output
        pred_audio = self._spec_to_audio(pred_spec, length=target_length)
        
        # 2. Length Matching (Safety Trim)
        min_len = min(pred_audio.size(-1), target_audio.size(-1))
        pred_audio = pred_audio[..., :min_len]
        target_audio = target_audio[..., :min_len]

        # 3. Time Domain Loss (L1)
        time_loss = F.l1_loss(pred_audio, target_audio)
        
        # 4. Frequency Domain Loss (Multi-Res STFT)
        freq_loss = self.stft_loss(pred_audio, target_audio)
        
        return (self.alpha_time * time_loss) + (self.alpha_freq * freq_loss)

    def _spec_to_audio(self, spec, length=None):
        """
        Robust iSTFT that handles center=False using the Clamped Window trick.
        """
        spec_complex = torch.view_as_complex(spec.contiguous())
        
        # --- THE FIX: Clamped Window Trick ---
        # We modify the window slightly at the edges (0 and -1) to be non-zero (1e-4).
        # This satisfies the NOLA check for iSTFT when center=False, preventing the crash.
        clamped_window = self.window.clone()
        clamped_window[0] = 1e-4
        clamped_window[-1] = 1e-4
        
        try:
            audio = torch.istft(
                spec_complex, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                window=clamped_window,  # Use the safe window
                center=False,           # STRICTLY CAUSAL
                normalized=True, 
                length=length
            )
        except RuntimeError:
            # Emergency fallback: Pad spectrogram + Clamped Window
            # This handles extreme edge cases where input length < n_fft
            pad_spec = F.pad(spec_complex, (0, 1)) 
            audio = torch.istft(
                pad_spec, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                window=clamped_window,
                center=False, 
                normalized=True
            )
            if length is not None:
                audio = audio[..., :length]
                
        return audio


class MultiDomainLoss(nn.Module):
    """LEGACY: Multi-Domain Loss (Frequency + Time)"""
    def __init__(self, n_fft=1024, hop_length=512, device='cuda'):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def forward(self, pred_spec, target_spec, target_audio, target_length=None):
        # 1. Frequency Domain Loss
        freq_loss = F.l1_loss(pred_spec, target_spec)
        
        # 2. Time Domain Loss
        pred_audio = self._spec_to_audio(pred_spec, length=target_length)
        
        # Safety trim
        min_len = min(pred_audio.size(-1), target_audio.size(-1))
        pred_audio = pred_audio[..., :min_len]
        target_audio = target_audio[..., :min_len]
            
        time_loss = F.l1_loss(pred_audio, target_audio)
        
        return freq_loss + time_loss
    
    def _spec_to_audio(self, spec, length=None):
        spec_complex = torch.view_as_complex(spec.contiguous())
        audio = torch.istft(
            spec_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=False,
            normalized=True,
            length=None
        )
        if length is not None:
            audio = audio[..., :length]
        return audio