import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import librosa
from pyAudioAnalysis.MidTermFeatures import mid_feature_extraction


class FeatureExtractor:
    def __init__(self, 
                 sr: int = 22050,
                 mid_window: float = 1.0,
                 mid_step: float = 1.0,
                 short_window: float = 0.050,
                 short_step: float = 0.025):
        self.sr = sr
        self.mid_window = mid_window
        self.mid_step = mid_step
        self.short_window = short_window
        self.short_step = short_step
    
    def extract_features(self, audio_path: str) -> Dict[str, float]:
        try:
            x, Fs = librosa.load(audio_path, sr=self.sr)
            _, mid_features, feature_names = mid_feature_extraction(
                x, 
                Fs, 
                int(self.mid_window * Fs),
                int(self.mid_step * Fs),
                int(self.short_window * Fs),
                int(self.short_step * Fs)
            )
            
            features = {}
            if mid_features.size > 0 and len(feature_names) > 0:
                feature_vector = np.mean(mid_features.T, axis=0)
                for i in range(len(feature_vector)):
                    if i < len(feature_names):
                        features[feature_names[i]] = float(feature_vector[i])
                    else:
                        features[f'feature_{i}'] = float(feature_vector[i])
            
            features['duration'] = len(x) / Fs
            return features
        except Exception:
            return {}
    
    def extract_batch(self, 
                     filepaths: List[str], 
                     labels: Optional[List[str]] = None,
                     progress: bool = True) -> pd.DataFrame:
        from tqdm import tqdm
        
        features_list = []
        iterator = tqdm(filepaths, desc="Extracting features") if progress else filepaths
        
        for i, filepath in enumerate(iterator):
            features = self.extract_features(filepath)
            features['filepath'] = filepath
            features['filename'] = Path(filepath).name
            
            if labels and i < len(labels):
                features['label'] = labels[i]
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
