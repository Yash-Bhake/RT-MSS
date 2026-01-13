import torch
import torchaudio
import random
import numpy as np


def pitch_shift(audio, sample_rate, n_steps):
    """
    Pitch shift audio by n_steps semitones.
    
    Args:
        audio: (C, T) audio tensor
        sample_rate: sample rate
        n_steps: number of semitones to shift
    """
    if n_steps == 0:
        return audio
    
    # Use torchaudio pitch shift
    effects = [
        ["pitch", str(n_steps * 100)],  # cents
        ["rate", str(sample_rate)]
    ]
    
    try:
        shifted, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio.unsqueeze(0) if audio.dim() == 1 else audio,
            sample_rate,
            effects
        )
        return shifted.squeeze(0) if audio.dim() == 1 else shifted
    except:
        # Fallback: simple resampling-based pitch shift
        rate_change = 2 ** (n_steps / 12)
        new_length = int(audio.size(-1) / rate_change)
        
        if audio.dim() == 1:
            shifted = torch.nn.functional.interpolate(
                audio.unsqueeze(0).unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=False
            ).squeeze()
        else:
            shifted = torch.nn.functional.interpolate(
                audio.unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=False
            ).squeeze(0)
        
        # Pad or trim to original length
        if shifted.size(-1) < audio.size(-1):
            pad = audio.size(-1) - shifted.size(-1)
            shifted = torch.nn.functional.pad(shifted, (0, pad))
        else:
            shifted = shifted[..., :audio.size(-1)]
        
        return shifted


def apply_augmentation(vocals, bass, drums, other, config):
    """
    Apply data augmentation to stems.
    
    Args:
        vocals, bass, drums, other: (1, T) audio tensors
        config: augmentation configuration dict
    
    Returns:
        Augmented stems
    """
    stems = [vocals, bass, drums, other]
    stem_names = ['vocals', 'bass', 'drums', 'other']
    
    # Pitch shift augmentation
    if config.get('pitch_shift', False) and random.random() < 0.5:
        # Select random stem to pitch shift
        stem_idx = random.randint(0, 3)
        n_steps = random.choice([-2, -1, 1, 2])
        
        stems[stem_idx] = pitch_shift(stems[stem_idx], 44100, n_steps)
    
    # Gain augmentation (already applied in dataset, but can add more variation)
    if config.get('extra_gain', False) and random.random() < 0.3:
        stem_idx = random.randint(0, 3)
        gain_db = random.uniform(-10, 10)
        gain = 10 ** (gain_db / 20)
        stems[stem_idx] = stems[stem_idx] * gain
    
    # Stem remixing (automix-style)
    if config.get('automix', False) and random.random() < 0.2:
        # Simple version: randomly swap stems from different songs
        # This would require access to other songs, so we'll skip for now
        # In full implementation, you'd load candidate stems and mix them
        pass
    
    return stems[0], stems[1], stems[2], stems[3]