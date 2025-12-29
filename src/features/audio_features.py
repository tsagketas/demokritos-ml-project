from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import numpy as np

def extract_short_term_features(file_path, st_win=0.050, st_step=0.025):
    """
    Extract short-term features from a wav file using pyAudioAnalysis.
    
    Args:
        file_path (str): path to wav file
        st_win (float): short-term window length in seconds
        st_step (float): short-term step in seconds
        
    Returns:
        F (np.ndarray): feature matrix (num_features x num_frames)
        f_names (list): list of feature names
        Fs (int): sampling rate
        x (np.ndarray): audio signal
    """
    # Load audio
    Fs, x = audioBasicIO.read_audio_file(file_path)
    
    # Convert window/step from seconds to samples
    win_samp = int(st_win * Fs)
    step_samp = int(st_step * Fs)
    
    # Extract short-term features
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, win_samp, step_samp)
    
    return F, f_names, Fs, x