import librosa
import numpy as np
import os
import soundfile as sf

def preprocess_audio(file_path, target_sr=16000, frame_ms=30, hop_ms=10,
                     trim_silence=True, save_path=None):
    """
    Preprocess audio: resample, normalize, trim silence, frame, optionally save processed wav.
    """
    # Load & resample
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)

    # Normalize
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Trim silence
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=20)

    # Frame the signal
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T

    # Save processed wav if save_path is given
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sf.write(save_path, frames.flatten(), sr)

    return frames, sr
