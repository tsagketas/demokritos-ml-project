import os
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(BASE_DIR, 'data', 'iemocap', 'iemocap_features_original.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'data', 'iemocap', 'iemocap_features.csv')

def clean_and_map_data():
    """
    Cleans IEMOCAP data by:
    1. Dropping emotions with < 10 occurrences.
    2. Keeping only samples with agreement >= 2 (removing ambiguous labels).
    3. Mapping specific emotions to consolidated 4-class setup.
    """
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # 1. Drop emotions with < 10 occurrences
    counts = df['emotion'].value_counts()
    emotions_to_drop = counts[counts < 10].index.tolist()
    if emotions_to_drop:
        print(f"Dropping emotions with < 10 occurrences: {emotions_to_drop}")
        df = df[~df['emotion'].isin(emotions_to_drop)]

    # 2. Filter by agreement (at least 2 annotators must agree)
    # This removes ambiguous samples that confuse the classifier
    if 'agreement' in df.columns:
        initial_len = len(df)
        df = df[df['agreement'] >= 2]
        print(f"Kept {len(df)} samples with agreement >= 2 (Dropped {initial_len - len(df)})")
    
    # 3. Define mapping (4-class setup: angry, happy_excited, sad, neutral)
    mapping = {
        'ang': 'angry',
        'fru': 'angry',       # Frustrated merged into angry
        'hap': 'happy_excited',
        'exc': 'happy_excited',
        'sad': 'sad',
        'neu': 'neutral'
    }
    
    print("Mapping emotions to 4-class setup...")
    # Apply mapping
    df['emotion'] = df['emotion'].map(mapping)
    
    # 4. Drop rows that are not in the mapping
    initial_len = len(df)
    df = df.dropna(subset=['emotion'])
    dropped_count = initial_len - len(df)
    
    print(f"Kept {len(df)} rows. Dropped {dropped_count} rows not in target mapping.")
    
    # Final distribution
    print("\nFinal class distribution:")
    print(df['emotion'].value_counts())
    
    # Save to new file
    print(f"\nSaving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done!")

if __name__ == "__main__":
    clean_and_map_data()

