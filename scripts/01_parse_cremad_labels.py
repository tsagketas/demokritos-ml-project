import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import CREMADLoader


def main():
    dataset_path = Path(__file__).parent.parent / "datasets" / "cremad"
    output_path = Path(__file__).parent.parent / "data" / "cremad_metadata.csv"
    
    loader = CREMADLoader(str(dataset_path))
    
    if not loader.audio_path.exists():
        print(f"ERROR: Audio directory not found: {loader.audio_path}")
        sys.exit(1)
    
    wav_files = list(loader.audio_path.glob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")
    
    print("Parsing filenames...")
    metadata = []
    errors = []
    
    for wav_file in wav_files:
        parsed = loader.parse_filename(wav_file.name)
        if parsed:
            parsed['filepath'] = str(wav_file.relative_to(Path(__file__).parent.parent))
            metadata.append(parsed)
        else:
            errors.append(wav_file.name)
    
    df = pd.DataFrame(metadata)
    
    print(f"Parsed: {len(metadata)} files")
    if errors:
        print(f"Failed: {len(errors)} files")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    print(f"\nTotal: {len(df)}")
    print(f"Actors: {df['actor_id'].nunique()}")
    print(f"Sentences: {df['sentence_code'].nunique()}")
    print("\nEmotions:")
    print(df['emotion'].value_counts())


if __name__ == "__main__":
    main()
