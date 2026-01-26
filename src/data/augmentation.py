import torch
import torchaudio
import random
import math

def apply_pitch_shift(wav, sample_rate, n_steps_cents):
    """
    Apply pitch shifting using torchaudio sox effects.
    
    Args:
        wav: (C, T) or (T,) audio tensor
        sample_rate: sampling rate
        n_steps_cents: shift in cents (100 cents = 1 semitone)
    """
    # Ensure shape is (C, T)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    
    # Define effects: [pitch shift in cents, rate restoration]
    # Note: 'pitch' effect in sox changes duration. 'rate' restores it? 
    # Actually, standard 'pitch' in sox usually preserves duration by resampling.
    # Typically: ['pitch', shift] -> changes pitch + duration. 
    # We want to preserve duration? 
    # Torchaudio's 'pitch' effect changes length. We usually need to fix length afterwards.
    
    effects = [
        ["pitch", str(n_steps_cents)],
        ["rate", str(sample_rate)]
    ]
    
    try:
        # apply_effects_tensor returns (augmented_wav, sample_rate)
        out, _ = torchaudio.sox_effects.apply_effects_tensor(wav, sample_rate, effects)
        
        # Crop or Pad to original length if slight mismatch occurs
        if out.size(-1) != wav.size(-1):
            if out.size(-1) > wav.size(-1):
                out = out[..., :wav.size(-1)]
            else:
                pad_len = wav.size(-1) - out.size(-1)
                out = torch.nn.functional.pad(out, (0, pad_len))
        return out
    except Exception as e:
        # Fallback if sox fails (e.g. missing backend): return original
        return wav

def apply_dynamic_mix(stems, prob, db_range):
    """
    Randomly scale stems and return new mix.
    stems: dict of {name: tensor}
    """
    if random.random() > prob:
        return stems, None # No change
    
    new_stems = {}
    gains = {}
    
    for name, wav in stems.items():
        # Random gain between -db and +db
        gain_db = random.uniform(-db_range, db_range)
        gain_lin = 10 ** (gain_db / 20.0)
        
        new_stems[name] = wav * gain_lin
        gains[name] = gain_lin
        
    return new_stems, gains

def apply_stem_drop(stems, prob):
    """
    Randomly drop one stem (set to silent).
    """
    if random.random() > prob:
        return stems
        
    names = list(stems.keys())
    names.remove('vocals')  # Never drop vocals
    # Choose one to drop
    drop_target = random.choice(names)
    
    # Zero out
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
    Master pipeline controlled by Trainer via current_epoch.
    """
    stems = {'vocals': vocals, 'bass': bass, 'drums': drums, 'other': other}
    aug_cfg = config['augmentation']
    
    # --- 1. Pitch Shift (Initial Epochs Only) ---
    pitch_cutoff = int(total_epochs * aug_cfg.get('pitch_shift_epoch_percent', 0.1))
    if current_epoch < pitch_cutoff:
        if random.random() < aug_cfg.get('pitch_shift_prob', 0.15):
            cents = random.uniform(
                -aug_cfg.get('scale_shift_cent_range', 200), 
                aug_cfg.get('scale_shift_cent_range', 200)
            )
            stems['vocals'] = apply_pitch_shift(stems['vocals'], sample_rate, cents)

    # --- 2. Dynamic Mixing (Plateau then Drop Strategy) ---
    # We want high mixing for most of training, dropping to 0 for the last 20% (Fine-tuning)
    fine_tune_start = int(total_epochs * 0.8)
    
    if current_epoch < fine_tune_start:
        # Phase 1: High Augmentation (Representation Learning)
        cur_mix_prob = aug_cfg.get('dynamic_mix_prob', 0.75)
        # Optional: Stem drop only happens during this phase
        cur_drop_prob = aug_cfg.get('stem_drop_prob', 0.05)
    else:
        # Phase 2: Fine-Tuning (Calibrate to real distribution)
        # We turn off mixing to let the model learn real volume ratios
        cur_mix_prob = 0.0
        cur_drop_prob = 0.0
    
    # Apply Dynamic Mixing
    stems, gains = apply_dynamic_mix(stems, cur_mix_prob, aug_cfg.get('dynamic_mix_range_db', 3.0))
    
    # Apply Stem Drop
    stems = apply_stem_drop(stems, cur_drop_prob)
    
    # Create Mixture
    mixture = stems['vocals'] + stems['bass'] + stems['drums'] + stems['other']
    
    # Prevent Clipping
    max_val = mixture.abs().max()
    if max_val > 0.99:
        scale = 0.99 / max_val
        mixture = mixture * scale
        stems['vocals'] = stems['vocals'] * scale

    return mixture, stems['vocals']