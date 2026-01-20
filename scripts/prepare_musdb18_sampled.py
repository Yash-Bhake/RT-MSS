import os
import musdb
import librosa
import soundfile as sf
from tqdm import tqdm
import yaml

# Load config
with open("config/vocal_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

TARGET_SR = config['data']['sample_rate']
OUTPUT_ROOT = "src/data/musdb18_sampled"

STEMS = ["vocals", "bass", "drums", "other"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_audio(path, audio, sr):
    sf.write(path, audio, sr, subtype="PCM_16")


def process_track(track, out_dir):
    ensure_dir(out_dir)

    # Mixture
    mix = track.audio.T  # shape: (channels, samples)
    mix_mono = mix.mean(axis=0)
    if TARGET_SR != track.rate:
        mix_16k = librosa.resample(mix_mono, orig_sr=track.rate, target_sr=TARGET_SR)
    else:
        mix_16k = mix_mono
    save_audio(os.path.join(out_dir, "mixture.wav"), mix_16k, TARGET_SR)

    # Stems
    for stem in STEMS:
        audio = track.targets[stem].audio.T
        audio_mono = audio.mean(axis=0)
        audio_16k = librosa.resample(
            audio_mono, orig_sr=track.rate, target_sr=TARGET_SR
        )
        save_audio(os.path.join(out_dir, f"{stem}.wav"), audio_16k, TARGET_SR)


def process_subset(subset):
    print(f"\nProcessing {subset} set")
    mus = musdb.DB(download=True, subsets=subset)

    out_subset_dir = os.path.join(OUTPUT_ROOT, subset)
    ensure_dir(out_subset_dir)

    for track in tqdm(mus.tracks):
        track_name = track.name.replace("/", "_")
        track_dir = os.path.join(out_subset_dir, track_name)
        process_track(track, track_dir)


def main():
    ensure_dir(OUTPUT_ROOT)
    process_subset("train")
    process_subset("test")
    print(f"\nDone. MUSDB18 prepared at {TARGET_SR} Hz.")


if __name__ == "__main__":
    main()
