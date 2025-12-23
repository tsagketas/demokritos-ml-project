import os
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(BASE_DIR, 'data', 'iemocap', 'iemocap_features_original.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'data', 'iemocap', 'iemocap_features.csv')

def clean_and_map_data():
    """
    Cleans IEMOCAP data by dropping emotions with < 10 occurrences
    and mapping specific emotions to consolidated classes.
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
    
    # 2. Define mapping
    # angry	angry
    # happy	happy_excited
    # excited	happy_excited
    # sad	sad
    # neutral	neutral
    # frustrated -> angry
    mapping = {
        'ang': 'angry_frustrated',
        'fru': 'angry_frustrated',
        'hap': 'happy_excited',
        'exc': 'happy_excited',
        'sad': 'sad',
        'neu': 'neutral'
    }
    
    print("Mapping emotions and filtering classes...")
    # Apply mapping
    df['emotion'] = df['emotion'].map(mapping)
    
    # 3. Drop rows that are not in the mapping (e.g., fru, sur, fea)
    initial_len = len(df)
    df = df.dropna(subset=['emotion'])
    dropped_count = initial_len - len(df)
    
    print(f"Kept {len(df)} rows. Dropped {dropped_count} rows that were not in the target mapping.")
    
    # Final distribution
    print("\nFinal class distribution:")
    print(df['emotion'].value_counts())
    
    # Save to new file
    print(f"\nSaving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done!")

if __name__ == "__main__":
    clean_and_map_data()

