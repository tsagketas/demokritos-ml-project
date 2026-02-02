# Datasets (project root)

Όλα τα raw datasets μπαίνουν εδώ. Τα scripts (και Docker με mount `.:/workspace`) διαβάζουν από `./datasets/`.

## IEMOCAP

- **Φάκελος:** `datasets/iemocap/`
- **Απαιτείται:**
  - `iemocap_full_dataset.csv` — metadata με columns π.χ. `path`, `emotion`
  - `IEMOCAP_full_release/` — φάκελος με τα wav (τα paths στο CSV να δείχνουν σχετικά από αυτό το φάκελο)

## CREMA-D

- **Φάκελος:** `datasets/cremad/`
- **Απαιτείται:**
  - `AudioWAV/` — φάκελος με `.wav` αρχεία (filename format: `..._<EMOTION>_...`)

## Docker

Με `docker-compose` το project root είναι mount στο `/workspace`, οπότε τα paths μέσα στο container είναι `/workspace/datasets/iemocap/`, `/workspace/datasets/cremad/` κλπ. Βεβαιωθείτε ότι αυτός ο φάκελος υπάρχει και περιέχει τα αρχεία πριν τρέξετε το workflow.
