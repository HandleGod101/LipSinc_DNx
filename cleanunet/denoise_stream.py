# import torch

import os

file_path = './exp/DNS-large-high/checkpoint/pretrained.pkl'  # Replace with the actual path to your .pkl file

if os.path.exists(file_path):
    print(f"The file '{file_path}' exists.")
else:
    print(f"The file '{file_path}' does not exist.")
# import sounddevice as sd
# import numpy as np
# from network import CleanUNet  # adjust if your model class name/path is different
# import argparse

# # ---------------------------
# # Arguments
# # ---------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
# parser.add_argument('--samplerate', type=int, default=16000, help='Audio sample rate')
# parser.add_argument('--chunk_size', type=int, default=512, help='Chunk size in samples (~32ms at 16kHz)')
# args = parser.parse_args()

# # ---------------------------
# # Load model
# # ---------------------------
# device = 'cpu'
# model = CleanUNet()  # adjust init parameters if needed
# checkpoint = torch.load(args.ckpt, map_location=device)
# model.load_state_dict(checkpoint['state_dict'])
# model.to(device)
# model.eval()

# print("Model loaded. Starting real-time denoising...")

# # ---------------------------
# # Audio callback
# # ---------------------------
# def audio_callback(indata, outdata, frames, time, status):
#     if status:
#         print(status)
#     # Convert to tensor
#     audio_chunk = torch.from_numpy(indata[:,0]).float().unsqueeze(0).to(device)  # [1, samples]
    
#     with torch.no_grad():
#         denoised_chunk = model(audio_chunk)  # [1, samples]
    
#     # Ensure shape matches
#     outdata[:,0] = denoised_chunk.squeeze(0).cpu().numpy()

# # ---------------------------
# # Start streaming
# # ---------------------------
# with sd.Stream(channels=1,
#                callback=audio_callback,
#                samplerate=args.samplerate,
#                blocksize=args.chunk_size):
#     print("Streaming... Press Ctrl+C to stop.")
#     try:
#         while True:
#             sd.sleep(1000)
#     except KeyboardInterrupt:
#         print("Stopped.")
