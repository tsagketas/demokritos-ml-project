import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import CREMADLoader
from src.dataset_analyzer import DatasetAnalyzer


def main():
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "datasets" / "cremad"
    metadata_path = project_root / "data" / "cremad_metadata.csv"
    reports_dir = project_root / "reports"
    plots_dir = project_root / "plots"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    loader = CREMADLoader(str(dataset_path))
    
    if not metadata_path.exists():
        print(f"ERROR: Metadata file not found: {metadata_path}")
        sys.exit(1)
    
    print("Loading metadata...")
    metadata_df = pd.read_csv(metadata_path)
    
    metadata_df['filepath'] = metadata_df['filepath'].apply(
        lambda x: str(project_root / x) if not Path(x).is_absolute() else x
    )
    
    analyzer = DatasetAnalyzer(metadata_df, loader)
    
    print("Calculating statistics...")
    stats = analyzer.get_basic_stats()
    
    print(f"\nTotal files: {stats['total_files']}")
    print(f"Actors: {stats['unique_actors']}")
    print(f"Sentences: {stats['unique_sentences']}")
    
    print("\nEmotions:")
    for emotion, count in sorted(stats['emotions'].items()):
        percentage = stats['emotion_distribution'][emotion] * 100
        print(f"  {emotion:12s}: {count:4d} ({percentage:5.2f}%)")
    
    print("\nAnalyzing audio (sampling 500 files)...")
    audio_info_df = analyzer.analyze_audio_characteristics(sample_size=500)
    
    if not audio_info_df.empty:
        print(f"Mean duration: {audio_info_df['duration'].mean():.2f}s")
        print(f"Sample rate: {audio_info_df['sample_rate'].mean():.0f} Hz")
    
    print("\nGenerating visualizations...")
    analyzer.plot_emotion_distribution(
        save_path=str(plots_dir / "emotion_distribution.png")
    )
    analyzer.plot_intensity_distribution(
        save_path=str(plots_dir / "intensity_distribution.png")
    )
    analyzer.plot_emotion_intensity_heatmap(
        save_path=str(plots_dir / "emotion_intensity_heatmap.png")
    )
    
    if not audio_info_df.empty:
        analyzer.plot_duration_distribution(
            audio_info_df,
            save_path=str(plots_dir / "duration_distribution.png")
        )
    
    report_path = reports_dir / "cremad_dataset_report.md"
    analyzer.generate_report(str(report_path), audio_info_df)
    print(f"\nReport saved: {report_path}")
    
    emotion_counts = list(stats['emotions'].values())
    max_count = max(emotion_counts)
    min_count = min(emotion_counts)
    imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")


if __name__ == "__main__":
    main()
