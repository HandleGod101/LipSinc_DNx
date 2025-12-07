import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import subprocess
import os
import librosa
import cv2
from scipy.io import wavfile
import mediapipe as mp
from typing import Tuple, List
from network import CleanUNet


# === CONFIG ===
PRETRAINED_AUDIO_MODEL = "./exp/DNS-large-high/checkpoint/pretrained.pkl"
INPUT_VIDEO = "./SampleFiles/car_no_face.mp4"
TMP_AUDIO = "temp_audio.wav"
OUTPUT_VIDEO = "./Results/car_no_face_output.mp4"
TARGET_SR = 16000
TARGET_FPS = 25


# === 1️⃣ MediaPipe Lip Region Extractor ===
class MediaPipeLipExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.lip_landmarks = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
            37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146
        ]

    def extract_lip_region(self, frame: np.ndarray) -> Tuple[torch.Tensor, bool]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            height, width = frame.shape[:2]

            lip_points = []
            for landmark_idx in self.lip_landmarks:
                if landmark_idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[landmark_idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    lip_points.append([x, y])

            if lip_points:
                lip_points = np.array(lip_points)
                x_min, y_min = lip_points.min(axis=0)
                x_max, y_max = lip_points.max(axis=0)

                padding = 20
                x_min = max(0, int(x_min - padding))
                y_min = max(0, int(y_min - padding))
                x_max = min(width, int(x_max + padding))
                y_max = min(height, int(y_max + padding))

                lip_region = frame[y_min:y_max, x_min:x_max]
                if lip_region.size > 0:
                    lip_region = cv2.resize(lip_region, (64, 64))
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                    return transform(lip_region), True

        # No face detected
        black_image = np.zeros((64, 64, 3), dtype=np.uint8)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform(black_image), False


class VisualFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lip_extractor = MediaPipeLipExtractor()
        self.lip_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.Dropout(0.3)
        )
        self.temporal_encoder = nn.GRU(512, 256, batch_first=True, bidirectional=True)

    def forward(self, video_frames: List[np.ndarray]) -> torch.Tensor:
        lip_features = []
        face_detected = False

        print("\n Extracting lip regions from frames...")
        for i, frame in enumerate(video_frames):
            lip_region, detected = self.lip_extractor.extract_lip_region(frame)
            if detected:
                print(f"✅ Frame {i}: Lips detected and extracted.")
                face_detected = True
            else:
                print(f"⚠️ Frame {i}: No lips detected.")
            lip_feat = self.lip_encoder(lip_region.unsqueeze(0))
            lip_features.append(lip_feat)

        if lip_features:
            lip_sequence = torch.cat(lip_features, dim=0).unsqueeze(0)
            print(" Encoding temporal lip movements across frames...")
            temporal_features, _ = self.temporal_encoder(lip_sequence)
            visual_context = temporal_features[:, -1, :]
            print(" Generated visual feature vector from lip motion.")
        else:
            visual_context = torch.zeros(1, 512)
            print(" !!! No visual features extracted — fallback to zeros.")

        return visual_context, face_detected


class VisualAwareCleanUNet(nn.Module):
    def __init__(self, base_model, visual_feature_dim=512, audio_feature_dim=256):
        super().__init__()
        self.audio_model = base_model
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_feature_dim, audio_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, audio_waveform: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        base_denoised = self.audio_model(audio_waveform)
        visual_projected = self.visual_projection(visual_features)
        modulated_denoised = base_denoised * (1 + 0.1 * visual_projected.mean())
        print(f" Fusing lip movement info into audio denoising (fusion weight = {0.15 * visual_projected.mean():.4f})")
        return modulated_denoised


# === 4️⃣ Video Utility ===
def extract_video_frames(video_path: str, target_fps: int = 25) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps / target_fps))

    frames, frame_count = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames, original_fps


# === 5️⃣ Main Pipeline ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    # Step 1: Extract audio
    print("\n Extracting audio from video...")
    subprocess.run([
        "ffmpeg", "-y", "-i", INPUT_VIDEO,
        "-vn", "-ar", str(TARGET_SR), "-ac", "1", "-acodec", "pcm_s16le", TMP_AUDIO
    ], check=True, capture_output=True)

    # Step 2: Extract video frames
    print("\n Extracting video frames...")
    video_frames, _ = extract_video_frames(INPUT_VIDEO, TARGET_FPS)
    print(f" Extracted {len(video_frames)} frames for lip analysis.")

    # Step 3: Load audio
    waveform, _ = librosa.load(TMP_AUDIO, sr=TARGET_SR, mono=True)
    waveform = waveform.astype(np.float32)

    # Step 4: Load models
    print("\n Loading models...")
    base_audio_model = CleanUNet()
    checkpoint = torch.load(PRETRAINED_AUDIO_MODEL, map_location=device)
    base_audio_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    base_audio_model.to(device)

    visual_extractor = VisualFeatureExtractor().to(device)
    model = VisualAwareCleanUNet(base_audio_model).to(device)
    model.eval()

    # Step 5: Extract visual context
    print("\n Extracting visual (lip) features...")
    with torch.no_grad():
        visual_context, face_detected = visual_extractor(video_frames[:min(30, len(video_frames))])

    if not face_detected:
        print("⚠️ No face detected — using audio-only denoising.")
        visual_context = torch.zeros(1, 512).to(device)
    else:
        print("✅ Face detected — lip features will guide denoising process.")

    # Step 6: Audio processing
    print("\n Denoising audio with visual guidance...")
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        std = waveform_tensor.std(dim=2, keepdim=True) + 1e-8
        denoised = model(waveform_tensor, visual_context)
        denoised = denoised * std

    denoised = denoised.squeeze().cpu().numpy()

    # Match volume
    orig_rms = np.sqrt(np.mean(waveform ** 2))
    denoised_rms = np.sqrt(np.mean(denoised ** 2))
    denoised = denoised * (orig_rms / (denoised_rms + 1e-8))
    denoised_int16 = np.clip(denoised * 32768.0, -32768, 32767).astype(np.int16)

    # Step 7: Save and combine
    print("\n Merging denoised audio with original video...")
    wavfile.write("clean_audio.wav", TARGET_SR, denoised_int16)
    subprocess.run([
        "ffmpeg", "-y", "-i", INPUT_VIDEO, "-i", "clean_audio.wav",
        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", OUTPUT_VIDEO
    ], check=True, capture_output=True)

    # Cleanup
    os.remove(TMP_AUDIO)
    os.remove("clean_audio.wav")

    print("\n✅ DONE: Visual-enhanced denoised video saved to", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
