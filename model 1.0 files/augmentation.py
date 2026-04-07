import torch
import torchaudio
import random
import warnings

def apply_pitch_shift(wav, sample_rate, n_steps_cents):
    """
    Apply pitch shifting using torchaudio sox effects.
    NOTE: Currently disabled in config (pitch_shift_prob=0)
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    
    effects = [
        ["pitch", str(n_steps_cents)],
        ["rate", str(sample_rate)]
    ]
    
    try:
        out, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sample_rate, effects)
        
        # Ensure output length matches input
        if out.size(-1) != wav.size(-1):
            if out.size(-1) > wav.size(-1):
                out = out[..., :wav.size(-1)]
            else:
                pad_len = wav.size(-1) - out.size(-1)
                out = torch.nn.functional.pad(out, (0, pad_len))
        return out
    except Exception:
        return wav

def apply_dynamic_mix(stems, prob, db_range):
    """
    Apply random gain to stems with a given probability.
    This is the MAIN augmentation technique.
    """
    if random.random() > prob:
        return stems
    
    new_stems = {}
    for name, wav in stems.items():
        # Skip silent stems
        if wav.abs().max() < 1e-6:
            new_stems[name] = wav
            continue

        # Random gain in dB range
        gain_db = random.uniform(-db_range, db_range)
        gain_lin = 10 ** (gain_db / 20.0)
        new_stems[name] = wav * gain_lin
            
    return new_stems

def apply_stem_drop(stems, prob):
    """
    Randomly silence non-vocal stems.
    Helps model learn to separate when some sources are missing.
    """
    if random.random() > prob:
        return stems
        
    names = list(stems.keys())
    if len(names) > 0:
        drop_target = random.choice(names)
        # Never drop vocals (our target)
        if drop_target != 'vocals': 
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
    Augmentation pipeline with 80/20 phase scheduling.
    
    PHASE 1 (First 80% of epochs): Full augmentation
    PHASE 2 (Last 20% of epochs): No augmentation (fine-tuning)
    
    Args:
        vocals, bass, drums, other: (1, T) audio tensors
        sample_rate: Sample rate
        config: Data config dict
        current_epoch: Current training epoch
        total_epochs: Total number of epochs
    
    Returns:
        mixture: (1, T) mixed audio
        target: (1, T) vocals
    """
    stems = {
        'vocals': vocals, 
        'bass': bass, 
        'drums': drums, 
        'other': other
    }
    
    # Calculate fine-tuning threshold (80% of total epochs)
    fine_tune_start = int(total_epochs * config.get('fine_tune_start_epoch_percent', 0.8))
    
    # -------------------------------------------------------
    # PHASE 1: AUGMENTATION (First 80%)
    # -------------------------------------------------------
    if current_epoch < fine_tune_start:
        
        # A. Pitch Shift (disabled by default, pitch_shift_prob=0)
        pitch_percent = config.get('pitch_shift_epoch_percent', 0.2)
        pitch_cutoff = int(total_epochs * pitch_percent)
        
        if current_epoch < pitch_cutoff:
            if random.random() < config.get('pitch_shift_prob', 0.0):
                cents = random.uniform(
                    -config.get('scale_shift_cent_range', 200), 
                    config.get('scale_shift_cent_range', 200)
                )
                stems['vocals'] = apply_pitch_shift(stems['vocals'], sample_rate, cents)

        # B. Dynamic Mixing (MAIN augmentation)
        cur_mix_prob = config.get('dynamic_mix_prob', 0.95)
        cur_drop_prob = config.get('stem_drop_prob', 0.10)

    # -------------------------------------------------------
    # PHASE 2: FINE-TUNING (Last 20%)
    # -------------------------------------------------------
    else:
        # Disable all augmentations for fine-tuning
        cur_mix_prob = 0.0
        cur_drop_prob = 0.0
    
    # Apply augmentations
    stems = apply_dynamic_mix(
        stems, 
        cur_mix_prob, 
        config.get('dynamic_mix_range_db', 3.0)
    )
    
    stems = apply_stem_drop(stems, cur_drop_prob)
    
    # Create mixture and target
    mixture = stems['vocals'] + stems['bass'] + stems['drums'] + stems['other']
    target = stems['vocals']
    
    # Soft limiter to prevent clipping
    max_val = mixture.abs().max()
    if max_val > 0.99:
        scale = 0.99 / (max_val + 1e-8)
        mixture = mixture * scale
        target = target * scale

    return mixture, target