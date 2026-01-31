# 🗣️ IEMOCAP Dataset
Interactive Emotional Dyadic Motion Capture (IEMOCAP) Dataset

## Χαρακτηριστικά

- **Σύνολο δειγμάτων**: 10039
- **Sessions**: 5
- **Μέθοδοι ηχογράφησης**:
  - Scripted (Βάση Σεναρίου) : 5255
  - Improvised (Αυτοσχεδιασμός) : 4784
- **Φύλο Ομιλητών**:
  - Άνδρες: 5098
  - Γυναίκες: 4941
- **Γλώσσα**: English
- **Τύπος δεδομένων**: Audio + Metadata CSV
- **Missing values**: 0

## Συναισθήματα (Emotion Labels)

- **No Agreement (xxx)** — 2507 *(ασυμφωνία annotators)*
- **Frustation (fru)** — 1849
- **Neutral state (neu)** — 1708
- **Anger (ang)** — 1103
- **Sadness (sad)** — 1084
- **Excited (exc)** — 1041
- **Happiness (hap)** — 595
- **Surprise (sur)** — 107
- **Fear (fea)** — 40
- **Other (oth)** — 3
- **Disgust (dis)** — 2


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
