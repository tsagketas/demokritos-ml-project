import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from .data_loader import CREMADLoader


class DatasetAnalyzer:
    def __init__(self, metadata_df: pd.DataFrame, loader: CREMADLoader):
        self.metadata = metadata_df
        self.loader = loader
        
    def get_basic_stats(self) -> Dict:
        stats = {
            'total_files': len(self.metadata),
            'unique_actors': self.metadata['actor_id'].nunique(),
            'unique_sentences': self.metadata['sentence_code'].nunique(),
            'emotions': self.metadata['emotion'].value_counts().to_dict(),
            'intensities': self.metadata['intensity'].value_counts().to_dict(),
            'emotion_distribution': self.metadata['emotion'].value_counts(normalize=True).to_dict()
        }
        return stats
    
    def analyze_audio_characteristics(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        files_to_analyze = self.metadata
        if sample_size:
            files_to_analyze = self.metadata.sample(min(sample_size, len(self.metadata)))
        
        audio_info = []
        for _, row in files_to_analyze.iterrows():
            info = self.loader.get_audio_info(row['filepath'])
            if 'error' not in info:
                info['filename'] = row['filename']
                info['emotion'] = row['emotion']
                audio_info.append(info)
        
        return pd.DataFrame(audio_info)
    
    def plot_emotion_distribution(self, save_path: Optional[str] = None):
        emotion_counts = self.metadata['emotion'].value_counts()
        
        plt.figure(figsize=(10, 6))
        emotion_counts.plot(kind='bar', color='steelblue')
        plt.title('CREMA-D Emotion Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Emotion', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_intensity_distribution(self, save_path: Optional[str] = None):
        intensity_counts = self.metadata['intensity'].value_counts()
        
        plt.figure(figsize=(10, 6))
        intensity_counts.plot(kind='bar', color='coral')
        plt.title('CREMA-D Intensity Level Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Intensity', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_emotion_intensity_heatmap(self, save_path: Optional[str] = None):
        cross_tab = pd.crosstab(self.metadata['emotion'], self.metadata['intensity'])
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
        plt.title('Emotion × Intensity Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Intensity', fontsize=12)
        plt.ylabel('Emotion', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_duration_distribution(self, audio_info_df: pd.DataFrame, save_path: Optional[str] = None):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(audio_info_df['duration'], bins=50, color='skyblue', edgecolor='black')
        plt.title('Duration Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Duration (seconds)', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        audio_info_df.boxplot(column='duration', by='emotion', ax=plt.gca())
        plt.title('Duration by Emotion', fontsize=12, fontweight='bold')
        plt.suptitle('')
        plt.xlabel('Emotion', fontsize=10)
        plt.ylabel('Duration (seconds)', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def generate_report(self, output_path: str, audio_info_df: Optional[pd.DataFrame] = None):
        stats = self.get_basic_stats()
        
        report = f"""# CREMA-D Dataset Analysis Report

## Basic Statistics

- **Total Files**: {stats['total_files']}
- **Unique Actors**: {stats['unique_actors']}
- **Unique Sentences**: {stats['unique_sentences']}

## Emotion Distribution

"""
        for emotion, count in sorted(stats['emotions'].items()):
            percentage = stats['emotion_distribution'][emotion] * 100
            report += f"- **{emotion}**: {count} samples ({percentage:.2f}%)\n"
        
        report += f"""
## Intensity Distribution

"""
        for intensity, count in sorted(stats['intensities'].items()):
            percentage = (count / stats['total_files']) * 100
            report += f"- **{intensity}**: {count} samples ({percentage:.2f}%)\n"
        
        if audio_info_df is not None and not audio_info_df.empty:
            report += f"""
## Audio Characteristics

- **Mean Duration**: {audio_info_df['duration'].mean():.2f} seconds
- **Std Duration**: {audio_info_df['duration'].std():.2f} seconds
- **Min Duration**: {audio_info_df['duration'].min():.2f} seconds
- **Max Duration**: {audio_info_df['duration'].max():.2f} seconds
- **Mean Sample Rate**: {audio_info_df['sample_rate'].mean():.0f} Hz
"""
        
        report += """
## Class Imbalance Analysis

"""
        emotion_counts = list(stats['emotions'].values())
        max_count = max(emotion_counts)
        min_count = min(emotion_counts)
        imbalance_ratio = max_count / min_count
        
        report += f"- **Imbalance Ratio**: {imbalance_ratio:.2f}:1 (max/min)\n"
        report += f"- **Most Common Emotion**: {max(stats['emotions'], key=stats['emotions'].get)} ({max_count} samples)\n"
        report += f"- **Least Common Emotion**: {min(stats['emotions'], key=stats['emotions'].get)} ({min_count} samples)\n"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

