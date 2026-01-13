#!/usr/bin/env python3
"""Real-time audio processor using ONNX runtime."""

import sounddevice as sd
import numpy as np
from scipy.signal.windows import hann
from numpy.fft import rfft, rfftfreq, irfft
import onnxruntime as ort
from multiprocessing import Process, Queue
import argparse
import time


FS = 44100
BUFFER_SIZE = 512
N_FFT = 1024
HOP_LENGTH = 512


class AudioProcessor(Process):
    """Audio processor for real-time vocal separation."""
    
    def __init__(self, in_queue, out_queue, onnx_path, buffer_size=BUFFER_SIZE, loopback=False):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.onnx_path = onnx_path
        self.buffer_size = buffer_size
        self.loopback = loopback
    
    def run(self):
        # Setup ONNX session
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        opts.enable_profiling = False
        
        providers = ['CPUExecutionProvider']
        if ort.get_device() == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        print(f"Loading ONNX model: {self.onnx_path}")
        session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=providers)
        
        # Get input/output names
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        
        print(f"Model inputs: {input_names[:3]}...")  # Print first few
        print(f"Model outputs: {output_names[:3]}...")
        
        # Initialize states (all zeros initially)
        state_inputs = {}
        for name in input_names[1:]:  # Skip first input (mixture_spec)
            # Infer shape from session
            shape = session.get_inputs()[input_names.index(name)].shape
            # Replace dynamic dims with concrete values
            shape = [1 if isinstance(s, str) else s for s in shape]
            state_inputs[name] = np.zeros(shape, dtype=np.float32)
        
        # STFT setup
        window = hann(N_FFT)
        in_buffer = np.zeros((N_FFT - HOP_LENGTH,))
        out_buffer = np.zeros((N_FFT - HOP_LENGTH,))
        
        # Frequency bins for causal STFT
        num_freq_bins = N_FFT // 2 + 1
        
        print("Audio processor ready!")
        
        while True:
            # Get input audio
            indata = self.in_queue.get()
            if indata is None:
                break
            
            # Convert to mono if stereo
            if indata.shape[1] == 2:
                indata = indata.mean(axis=1, keepdims=True)
            
            # Build frame with overlap
            frame = np.concatenate((in_buffer, indata[:, 0]))
            
            # Compute STFT (left-aligned for causality)
            # Pad to N_FFT
            if len(frame) < N_FFT:
                frame = np.pad(frame, (0, N_FFT - len(frame)))
            
            # Apply window and FFT
            windowed_frame = frame[:N_FFT] * window
            Frame = rfft(windowed_frame, n=N_FFT)
            
            # Convert to real/imag format (B, F, T, 2)
            Frame_complex = np.stack([Frame.real, Frame.imag], axis=-1)
            Frame_complex = Frame_complex[np.newaxis, :, np.newaxis, :]  # (1, F, 1, 2)
            
            if self.loopback:
                # Bypass processing
                output_complex = Frame_complex
            else:
                # Prepare inputs
                inputs = {'mixture_spec': Frame_complex.astype(np.float32)}
                inputs.update(state_inputs)
                
                # Run inference
                outputs = session.run(output_names, inputs)
                
                # Extract separated spec and update states
                output_complex = outputs[0]  # (1, F, 1, 2)
                
                # Update states for next frame
                for i, name in enumerate(output_names[1:], 1):
                    state_inputs[input_names[i]] = outputs[i]
            
            # Convert back to complex
            output_complex = output_complex[0, :, 0, :]  # (F, 2)
            output_fft = output_complex[:, 0] + 1j * output_complex[:, 1]
            
            # Inverse FFT
            output_frame = irfft(output_fft, n=N_FFT)
            output_frame = output_frame * window  # Apply window
            
            # Overlap-add
            output_audio = out_buffer + output_frame[:HOP_LENGTH]
            out_buffer = output_frame[HOP_LENGTH:N_FFT].copy()
            
            # Update input buffer
            in_buffer = indata[:, 0].copy() if len(indata[:, 0]) == (N_FFT - HOP_LENGTH) else frame[HOP_LENGTH:N_FFT].copy()
            
            # Send output
            self.out_queue.put(output_audio.reshape(-1, 1).astype(np.float32))


def audio_callback(indata, outdata, frames, time_info, status):
    """Callback for sounddevice stream."""
    if status:
        print(status)
    
    in_queue.put(indata.copy())
    
    try:
        oframe = out_queue.get_nowait()
        outdata[:] = oframe
    except:
        outdata[:] = np.zeros_like(outdata)


def main():
    global in_queue, out_queue
    
    parser = argparse.ArgumentParser(description='Real-time vocal separation')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--loopback', action='store_true',
                        help='Enable loopback (bypass processing)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List audio devices and exit')
    parser.add_argument('--input-device', type=int, default=None,
                        help='Input device ID')
    parser.add_argument('--output-device', type=int, default=None,
                        help='Output device ID')
    
    args = parser.parse_args()
    
    if args.list_devices:
        print("Available audio devices:")
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            print(f"{idx}: {device['name']}")
            print(f"   Input channels: {device['max_input_channels']}")
            print(f"   Output channels: {device['max_output_channels']}")
        return
    
    # Select devices
    input_device = args.input_device if args.input_device is not None else sd.default.device[0]
    output_device = args.output_device if args.output_device is not None else sd.default.device[1]
    
    print(f"Input device: {input_device}")
    print(f"Output device: {output_device}")
    
    # Create queues
    in_queue = Queue(maxsize=4)
    out_queue = Queue(maxsize=4)
    
    # Start audio processor
    audio_processor = AudioProcessor(
        in_queue, out_queue, args.model,
        buffer_size=HOP_LENGTH,
        loopback=args.loopback
    )
    audio_processor.start()
    
    # Give processor time to initialize
    time.sleep(2)
    
    try:
        print(f"Starting audio stream at {FS} Hz...")
        print("Press Ctrl+C to stop")
        
        with sd.Stream(
            device=(input_device, output_device),
            channels=1,
            blocksize=HOP_LENGTH,
            callback=audio_callback,
            samplerate=FS
        ) as stream:
            stream.start()
            
            # Keep running
            while True:
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        # Cleanup
        in_queue.put(None)
        audio_processor.join()
        print("Stopped.")


if __name__ == "__main__":
    main()