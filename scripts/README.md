# Scripts Overview

## 01_parse_cremad_labels.py

Διαβάζει τα audio files και βγάζει τα labels από τα ονόματα. Τα αρχεία έχουν format `1091_IEO_ANG_HI.wav` που σημαίνει actor 1091, sentence IEO, emotion Anger, intensity High. Το script παίρνει όλα αυτά και τα βάζει σε ένα CSV με metadata για κάθε αρχείο. Απλά organization του dataset.

## 02_explore_cremad.py

Κάνει μια πρώτη ματιά στο dataset. Μετράει πόσα samples έχουμε για κάθε emotion, βλέπει αν είναι balanced, βγάζει στατιστικά για διάρκεια και sample rate, και φτιάχνει μερικά plots για να δούμε τι έχουμε. Βασικά για να καταλάβουμε αν το dataset είναι καλό και αν χρειάζεται κάτι.

## 03_extract_features.py

Εδώ γίνεται η δουλειά. Παίρνει κάθε audio file και βγάζει features με pyAudioAnalysis. Κάθε αρχείο γίνεται ένα vector με ~68 features (spectral features, MFCCs, zero crossing rate, κλπ). Μετά τα βάζει όλα σε ένα CSV. Αυτό είναι το raw feature matrix που θα χρησιμοποιήσουμε.

## 04_analyze_features.py

Ελέγχει τα features που βγάλαμε. Βλέπει αν έχουμε missing values, αν υπάρχουν correlations, αν κάποια features είναι άχρηστα (low variance), και υπολογίζει ποια features είναι πιο σημαντικά με Mutual Information. Επίσης κάνει PCA visualization για να δούμε αν τα features διαχωρίζουν καλά τα emotions. Γενικά quality check πριν προχωρήσουμε.

## 05_preprocess_features.py

Εδώ καθαρίζουμε και προετοιμάζουμε τα features για training. Αφαιρούμε features με χαμηλό variance (δεν λένε τίποτα), αφαιρούμε highly correlated features (redundancy), και κρατάμε τα σημαντικά. Μετά κάνουμε scaling (standardize) ώστε όλα τα features να είναι στο ίδιο scale.

**Γιατί όλο αυτό;**

Γιατί τα audio files δεν μπορούν να μπουν απευθείας σε ένα model. Χρειάζεται να βγάλουμε features (αριθμητικά χαρακτηριστικά) που περιγράφουν κάθε audio, να τα καθαρίσουμε, και να τα κάνουμε scale ώστε το model να μπορεί να τα μάθει. Αυτό το pipeline κάνει όλη αυτή τη δουλειά από το raw audio μέχρι τα έτοιμα features.
