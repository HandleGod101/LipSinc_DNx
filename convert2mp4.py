from pydub import AudioSegment
import os

def mp4_to_wav_pydub(mp4_path, wav_path=None):
    """
    Convert MP4 to WAV using pydub (requires ffmpeg)
    
    Args:
        mp4_path (str): Path to input MP4 file
        wav_path (str): Path to output WAV file (optional)
    
    Returns:
        str: Path to the created WAV file
    """
    if wav_path is None:
        wav_path = mp4_path.replace('.mp4', '.wav')
    
    # Load MP4 file and export as WAV
    audio = AudioSegment.from_file(mp4_path, format="mp4")
    audio.export(wav_path, format="wav")
    
    return wav_path

# Usage
mp4_to_wav_pydub("./Results/sample_crowd_output.mp4", "sample_crowd_videotest.wav")