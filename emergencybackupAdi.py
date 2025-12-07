import torch
import torch.nn as nn
import numpy as np
import subprocess
import os
import librosa
from scipy.io import wavfile

# === CONFIG ===
PRETRAINED_AUDIO_MODEL = "./exp/DNS-large-high/checkpoint/pretrained.pkl"
INPUT_VIDEO = "car2.mp4"
TMP_AUDIO = "temp_audio.wav"
CLEAN_AUDIO = "clean_audio_audio_only.wav"
OUTPUT_VIDEO = "car2_audio_only_denoised.mp4"
TARGET_SR = 16000

# === Video Processing Utilities ===
def extract_audio(video_path: str, output_path: str, sr: int):
    """Extract audio from video using ffmpeg"""
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ar", str(sr), "-ac", "1", "-acodec", "pcm_s16le",
        output_path
    ], check=True, capture_output=True)

# === Main Processing Pipeline ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 1. Extract audio ===
    print("Extracting audio from video...")
    extract_audio(INPUT_VIDEO, TMP_AUDIO, TARGET_SR)

    # === 2. Load audio ===
    waveform, sr = librosa.load(TMP_AUDIO, sr=TARGET_SR, mono=True)
    waveform = waveform.astype(np.float32)

    # === 3. Load CleanUNet model ===
    print("Loading CleanUNet model...")
    from network import CleanUNet
    base_audio_model = CleanUNet()
    checkpoint = torch.load(PRETRAINED_AUDIO_MODEL, map_location=device)
    base_audio_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    base_audio_model.to(device)
    base_audio_model.eval()

    # === 4. Denoise audio ===
    print("Denoising audio (audio-only)...")
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        std = waveform_tensor.std(dim=2, keepdim=True) + 1e-8
        denoised = base_audio_model(waveform_tensor)
        denoised = denoised * std

    denoised = denoised.squeeze().cpu().numpy()

    # === 5. Post-processing ===
    orig_rms = np.sqrt(np.mean(waveform ** 2))
    denoised_rms = np.sqrt(np.mean(denoised ** 2))
    denoised = denoised * (orig_rms / (denoised_rms + 1e-8))
    denoised_int16 = np.clip(denoised * 32768.0, -32768, 32767).astype(np.int16)

    # === 6. Save audio ===
    print("Saving cleaned audio...")
    wavfile.write(CLEAN_AUDIO, TARGET_SR, denoised_int16)

    # === 7. Combine with original video ===
    print("Combining audio-only denoised audio with video...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", INPUT_VIDEO,
        "-i", CLEAN_AUDIO,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        OUTPUT_VIDEO
    ], check=True, capture_output=True)

    # Cleanup
    if os.path.exists(TMP_AUDIO):
        os.remove(TMP_AUDIO)

    print(f"✅ Audio-only denoised video saved to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
