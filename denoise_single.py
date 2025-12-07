import torch
from network import CleanUNet
from scipy.io import wavfile
import numpy as np
import librosa
import soundfile as sf

def process_with_overlap_add(model, audio, chunk_size=16000, overlap=4000, device='cpu'):
    model.eval()
    
    # Calculate hop size
    hop_size = chunk_size - overlap
    total_length = len(audio)
    output = np.zeros(total_length)
    weight = np.zeros(total_length)
    
    # Process chunks
    for start in range(0, total_length, hop_size):
        end = start + chunk_size
        if end > total_length:
            # Pad last chunk
            chunk = np.pad(audio[start:], (0, end - total_length), mode='constant')
        else:
            chunk = audio[start:end]
        
        # Convert to tensor and process
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            denoised_chunk = model(chunk_tensor)
        
        denoised_chunk = denoised_chunk.squeeze().cpu().numpy()
        
        # Apply window function to reduce artifacts
        window = np.hanning(len(denoised_chunk))
        denoised_chunk = denoised_chunk * window
        
        # Add to output with overlap
        if end > total_length:
            actual_length = total_length - start
            output[start:start + actual_length] += denoised_chunk[:actual_length]
            weight[start:start + actual_length] += window[:actual_length]
        else:
            output[start:end] += denoised_chunk
            weight[start:end] += window
    
    # Normalize by weights to avoid amplitude issues
    output = output / (weight + 1e-8)
    return output

PRETRAINED_MODEL = "./exp/DNS-large-high/checkpoint/pretrained.pkl"
INPUT_FILE = "sample_crowd_audiotest.wav"
OUTPUT_FILE = "sample_crowd_audiotest.wav"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CleanUNet()
checkpoint = torch.load(PRETRAINED_MODEL, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.to(device)

TARGET_SR = 16000
waveform, _ = librosa.load(INPUT_FILE, sr=TARGET_SR, mono=True)

denoised_np = process_with_overlap_add(model, waveform, chunk_size=32000, overlap=8000, device=device)

def post_process_audio(audio, sr=16000):
    """Apply post-processing to reduce artifacts"""
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.95
    
    # Gentle low-pass filter to reduce high-frequency artifacts
    from scipy import signal
    b, a = signal.butter(4, 7000/(sr/2), 'low')
    audio = signal.filtfilt(b, a, audio)
    
    return audio

denoised_np = post_process_audio(denoised_np, TARGET_SR)

denoised_int16 = np.clip(denoised_np * 32768.0, -32768, 32767).astype(np.int16)
wavfile.write(OUTPUT_FILE, TARGET_SR, denoised_int16)
print(f"Denoised audio saved to {OUTPUT_FILE}")