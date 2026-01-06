# Real-Time Causal Band-Split RNN

Low-latency music source separation system based on Band-Split RNN, optimized for real-time processing on laptop hardware.

## Features

- **Causal Architecture**: Unidirectional LSTM with cumulative layer normalization
- **Low Latency**: <40ms total latency (32ms window, 16ms hop)
- **8-bit Quantization**: Dynamic quantization for faster inference
- **Circular Buffer**: Efficient real-time audio processing
- **Data Augmentation**: Pitch shifting and gain scaling
- **Multi-target**: Separates vocals, drums, bass, and other

## Architecture Comparison

### Original BSRNN
- Bidirectional LSTM (non-causal)
- Standard layer normalization
- Centered STFT windows
- Processing time: ~100-200ms per chunk

### Causal BSRNN (This Implementation)
- Unidirectional LSTM (causal)
- Cumulative layer normalization
- Left-aligned STFT windows
- Processing time: <16ms per chunk
- 8-bit quantization during inference

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python scripts/train.py --data_path /path/to/musdb18 --config config/model_config.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data_path /path/to/musdb18
```

### Real-time Processing

```python
from src.inference.realtime_processor import RealtimeProcessor

processor = RealtimeProcessor(checkpoint_path='checkpoints/best_model.pth')
output_audio = processor.process_stream(input_audio_chunk)
```

## Performance Metrics

- SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
- SDR (Signal-to-Distortion Ratio)
- RTF (Real-Time Factor)

Target RTF < 1.0 for real-time processing on laptop CPU.

## Configuration

Edit `config/model_config.yaml` to adjust:
- Model architecture (feature dimensions, layers)
- Training hyperparameters
- Band-split configurations
- Latency settings

## Citation

Based on the paper:
```
@article{luo2022bandsplit,
  title={Music Source Separation with Band-split RNN},
  author={Luo, Yi and Yu, Jianwei},
  journal={arXiv preprint arXiv:2209.15174},
  year={2022}
}
```