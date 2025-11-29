import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
project_root = Path(__file__).parent.parent
metadata_path = project_root / "data" / "iemocap_full_dataset.csv"
readme_path = project_root / "datasets"/"iemocap"/"README.md"

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(metadata_path)

# -----------------------------
# Compute statistics
# -----------------------------
total_samples = len(df)
sessions_count = df["session"].nunique()
methods_counts = df["method"].value_counts()
gender_counts = df["gender"].value_counts()
emotion_counts = df["emotion"].value_counts()
missing_values = df.isnull().sum().sum()

# -----------------------------
# Build README content
# -----------------------------
readme_text = f"""# 🗣️ IEMOCAP Dataset
Interactive Emotional Dyadic Motion Capture (IEMOCAP) Dataset

## Χαρακτηριστικά

- **Σύνολο δειγμάτων**: {total_samples}
- **Sessions**: {sessions_count}
- **Μέθοδοι ηχογράφησης**:
  - Scripted (Βάση Σεναρίου) : {methods_counts.get('script', 0)}
  - Improvised (Αυτοσχεδιασμός) : {methods_counts.get('impro', 0)}
- **Φύλο Ομιλητών**:
  - Άνδρες: {gender_counts.get('M', 0)}
  - Γυναίκες: {gender_counts.get('F', 0)}
- **Γλώσσα**: English
- **Τύπος δεδομένων**: Audio + Metadata CSV
- **Missing values**: {missing_values}

## Συναισθήματα (Emotion Labels)

"""
# Append emotion counts
for emotion, count in emotion_counts.items():
    if emotion == "xxx":
        readme_text += f"- **No Agreement ({emotion})** — {count} *(ασυμφωνία annotators)*\n"
    else:
        readme_text += f"- **{emotion}** — {count}\n"

# Metadata fields
readme_text += """

## Στήλες Μεταδεδομένων (Metadata Fields)

- **session** — Session ID (1–5)
- **method** — Scripted / Improvised
- **gender** — M / F
- **emotion** — Emotion annotation label
- **n_annotators** — Πόσοι annotators αξιολόγησαν το δείγμα
- **agreement** — Πόσοι από τους annotators συμφώνησαν μεταξύ τους
- **path** — Διαδρομή προς το αρχείο ήχου

## Άδεια

⚠️ Τα δεδομένα του IEMOCAP **δεν είναι ανοιχτά**.  
Απαιτείται άδεια από το *Speech Analysis and Interpretation Laboratory* (SAIL) 
του Πανεπιστημίου της Νότιας Καλιφόρνιας: [follow link](https://sail.usc.edu/iemocap/)

## Citation

C. Busso et al., “IEMOCAP: interactive emotional dyadic motion capture database,” 
Lang. Resour. Evaluation, vol. 42, no. 4, pp. 335–359, 2008.
"""

# -----------------------------
# Write README.md
# -----------------------------
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_text)

print("README.md generated successfully for IEMOCAP!")
