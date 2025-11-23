import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import librosa
import soundfile as sf


class CREMADLoader:
    EMOTION_MAP = {
        'ANG': 'Anger',
        'DIS': 'Disgust', 
        'FEA': 'Fear',
        'HAP': 'Happy',
        'NEU': 'Neutral',
        'SAD': 'Sad'
    }
    
    INTENSITY_MAP = {
        'LO': 'Low',
        'MD': 'Medium',
        'HI': 'High',
        'XX': 'Unspecified'
    }
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.audio_path = self.dataset_path / "AudioWAV"
        
    def parse_filename(self, filename: str) -> Optional[Dict]:
        if not filename.endswith('.wav'):
            return None
            
        name = filename.replace('.wav', '')
        parts = name.split('_')
        
        if len(parts) != 4:
            return None
            
        actor_id, sentence_code, emotion_code, intensity_code = parts
        
        return {
            'filename': filename,
            'actor_id': actor_id,
            'sentence_code': sentence_code,
            'emotion_code': emotion_code,
            'emotion': self.EMOTION_MAP.get(emotion_code, 'Unknown'),
            'intensity_code': intensity_code,
            'intensity': self.INTENSITY_MAP.get(intensity_code, 'Unknown')
        }
    
    def load_metadata(self, metadata_path: Optional[str] = None) -> pd.DataFrame:
        if metadata_path and Path(metadata_path).exists():
            return pd.read_csv(metadata_path)
        
        metadata = []
        for wav_file in self.audio_path.glob("*.wav"):
            parsed = self.parse_filename(wav_file.name)
            if parsed:
                parsed['filepath'] = str(wav_file)
                metadata.append(parsed)
        
        return pd.DataFrame(metadata)
    
    def load_audio(self, filepath: str, sr: int = 22050) -> tuple:
        return librosa.load(filepath, sr=sr)
    
    def get_audio_info(self, filepath: str) -> Dict:
        try:
            info = sf.info(filepath)
            return {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'samples': info.frames,
                'channels': info.channels,
                'subtype': info.subtype
            }
        except Exception as e:
            return {'error': str(e)}

