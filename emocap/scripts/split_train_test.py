"""
Train/test split using Leave-One-Session-Out (LOSO) for speaker-independent evaluation.
Uses session 5 for testing, sessions 1-4 for training. This ensures no speaker overlap
between train and test sets, making the model more generalizable to unseen speakers.
"""

import os
import pandas as pd
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(BASE_DIR, 'data', 'iemocap', 'iemocap_features.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'iemocap')
TRAIN_CSV = os.path.join(OUTPUT_DIR, 'train.csv')
TEST_CSV = os.path.join(OUTPUT_DIR, 'test.csv')

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Total samples: {len(df)}")
    
    if 'session' not in df.columns:
        print("WARNING: 'session' column not found. Using random split...")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['emotion'], random_state=42)
    else:
        test_session = 5
        test_df = df[df['session'] == test_session].copy()
        train_df = df[df['session'] != test_session].copy()
        print(f"LOSO split: Test session {test_session}, Train sessions {sorted(train_df['session'].unique().tolist())}")
    
    print(f"\nTrain: {len(train_df)} samples ({100*len(train_df)/len(df):.1f}%)")
    print(f"Test: {len(test_df)} samples ({100*len(test_df)/len(df):.1f}%)")
    
    train_emotions = Counter(train_df['emotion'])
    test_emotions = Counter(test_df['emotion'])
    print("\nEmotion distribution:")
    for emotion in sorted(set(train_emotions.keys()) | set(test_emotions.keys())):
        print(f"  {emotion}: Train={train_emotions.get(emotion, 0)}, Test={test_emotions.get(emotion, 0)}")
    
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    print(f"\nSaved to {TRAIN_CSV} and {TEST_CSV}")

if __name__ == "__main__":
    main()
