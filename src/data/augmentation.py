import torch
import torchaudio
import random
import numpy as np
import pyloudnorm as pyln
import warnings

def apply_pitch_shift(wav, sample_rate, n_steps_cents):
    """
    Apply pitch shifting using torchaudio sox effects.
    Args:
        wav: (C, T) or (T,) audio tensor
        sample_rate: sampling rate
        n_steps_cents: shift in cents (100 cents = 1 semitone)
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    
    effects = [
        ["pitch", str(n_steps_cents)],
        ["rate", str(sample_rate)]
    ]
    
    try:
        out, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sample_rate, effects)
        
        # Crop or Pad to original length
        if out.size(-1) != wav.size(-1):
            if out.size(-1) > wav.size(-1):
                out = out[..., :wav.size(-1)]
            else:
                pad_len = wav.size(-1) - out.size(-1)
                out = torch.nn.functional.pad(out, (0, pad_len))
        return out
    except Exception as e:
        return wav

def apply_dynamic_mix(stems, prob, db_range, sample_rate=44100):
    """
    Apply Perceptual Dynamic Mixing using pyloudnorm.
    Normalization Strategy:
    1. Measure Input LUFS.
    2. If stem is active (above threshold), normalize to -24 LUFS.
    3. Apply random jitter (dB) on top.
    This ensures 'Quiet' stems don't become 'Loud' stems by accident unless requested.
    """
    if random.random() > prob:
        return stems, None 
    
    new_stems = {}
    gains = {}
    
    # Initialize Meter
    meter = pyln.Meter(sample_rate)
    
    # Target loudness (Broadcast Standard)
    anchor_lufs = -24.0 
    
    # Safety Threshold: Don't normalize stems quieter than this (avoids noise explosion)
    silence_threshold_lufs = -50.0

    for name, wav in stems.items():
        # pyloudnorm requires Numpy (Time, Channels)
        # wav is Tensor (C, T) -> Numpy (T, C)
        wav_np = wav.cpu().numpy().T
        
        # Handle Mono/Stereo for meter
        if wav_np.ndim == 1:
            wav_np = wav_np[:, None] # (T, 1)

        try:
            # Measure Loudness
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                input_lufs = meter.integrated_loudness(wav_np)
        except ValueError:
            # Usually happens on absolute silence
            input_lufs = -float('inf')

        # Logic: Only normalize if the stem actually has content
        if input_lufs > silence_threshold_lufs:
            try:
                # 1. Normalize to Anchor (-24 LUFS)
                norm_audio = pyln.normalize.loudness(wav_np, input_lufs, anchor_lufs)
                
                # 2. Apply Random Jitter (e.g., +/- 3dB)
                offset_db = random.uniform(-db_range, db_range)
                gain_lin = 10 ** (offset_db / 20.0)
                
                final_audio = norm_audio * gain_lin
                
                # Convert back to Tensor (C, T)
                new_stems[name] = torch.from_numpy(final_audio.T).to(wav.device, dtype=wav.dtype)
                gains[name] = gain_lin
                
            except Exception:
                # Fallback if normalization math fails
                new_stems[name] = wav
        else:
            # Stem is too quiet (noise/silence), keep original level to preserve dynamics
            # effectively treating it as 'background'
            new_stems[name] = wav
            gains[name] = 1.0
            
    return new_stems, gains

def apply_stem_drop(stems, prob):
    """
    Randomly drop one stem (set to silent).
    """
    if random.random() > prob:
        return stems
        
    names = list(stems.keys())
    drop_target = random.choice(names)
    stems[drop_target] = torch.zeros_like(stems[drop_target])
    
    return stems

def apply_augmentation_pipeline(
    vocals, bass, drums, other, 
    sample_rate, 
    config, 
    current_epoch, 
    total_epochs
):
    """
    Master pipeline with Plateau Schedule.
    """
    stems = {'vocals': vocals, 'bass': bass, 'drums': drums, 'other': other}
    aug_cfg = config['augmentation']
    
    # --- 1. Pitch Shift ---
    pitch_cutoff = int(total_epochs * aug_cfg.get('pitch_shift_epoch_percent', 0.1))
    if current_epoch < pitch_cutoff:
        if random.random() < aug_cfg.get('pitch_shift_prob', 0.15):
            cents = random.uniform(
                -aug_cfg.get('scale_shift_cent_range', 200), 
                aug_cfg.get('scale_shift_cent_range', 200)
            )
            stems['vocals'] = apply_pitch_shift(stems['vocals'], sample_rate, cents)

    # --- 2. Dynamic Mixing (Plateau Schedule) ---
    fine_tune_start = int(total_epochs * 0.8) # Last 20% is clean
    
    if current_epoch < fine_tune_start:
        cur_mix_prob = aug_cfg.get('dynamic_mix_prob', 1.0) # High mixing initially
        cur_drop_prob = aug_cfg.get('stem_drop_prob', 0.10)
    else:
        cur_mix_prob = 0.0 # Turn off for fine-tuning
        cur_drop_prob = 0.0
    
    # Updated to use pyloudnorm version
    stems, gains = apply_dynamic_mix(
        stems, 
        cur_mix_prob, 
        aug_cfg.get('dynamic_mix_range_db', 3.0),
        sample_rate
    )
    
    stems = apply_stem_drop(stems, cur_drop_prob)
    
    # Create Mixture
    mixture = stems['vocals'] + stems['bass'] + stems['drums'] + stems['other']
    
    # Soft Limiter (Prevent Clipping)
    max_val = mixture.abs().max()
    if max_val > 0.99:
        scale = 0.99 / max_val
        mixture = mixture * scale
        stems['vocals'] = stems['vocals'] * scale

    return mixture, stems['vocals']