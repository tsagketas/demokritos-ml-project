# Machine Learning Project - Docker Setup

## Προαπαιτούμενα

### 1. Εγκατάσταση Docker Desktop

Κατεβάστε και εγκαταστήστε το Docker Desktop από:
- **Windows/Mac**: https://www.docker.com/products/docker-desktop

### 2. Άνοιγμα Docker Desktop

Μετά την εγκατάσταση, ανοίξτε το Docker Desktop και βεβαιωθείτε ότι τρέχει (θα δείτε το Docker icon στην taskbar).

## Εγκατάσταση και Εκκίνηση

### 1. Build και Start του Container

```bash
docker-compose up --build -d 
```

Αυτή η εντολή θα:
- Δημιουργήσει το Docker image με Python και τα απαραίτητα packages (sklearn, pandas, pytorch, numpy)
- Ξεκινήσει το container `mlproject-container` στο background

### 2. Έλεγχος ότι ο container τρέχει σωστά

Μπορείτε να ελέγξετε αν ο container είναι σε λειτουργία με δύο τρόπους:

- Μέσα από το Docker Desktop, όπου θα πρέπει να δείτε το `mlproject-container` στη λίστα των ενεργών containers.
- Εναλλακτικά, εκτελέστε στο terminal:
  
  ```bash
  docker ps
  ```
  και ελέγξτε ότι το `mlproject-container` εμφανίζεται στη λίστα.


## Χρήση

### Εκτέλεση Python Script

Για να τρέξετε ένα Python script μέσα στο container:

```bash
docker exec -it mlproject-container python your_script.py
```

**Παράδειγμα:**
```bash
docker exec -it mlproject-container python test.py
```

### Ανοίγμα Interactive Python Shell

```bash
docker exec -it mlproject-container python
```

### Ανοίγμα Bash Shell στο Container

```bash
docker exec -it mlproject-container bash
```

## Σταμάτημα του Container

```bash
docker-compose down
```

## Επανεκκίνηση

```bash
docker-compose restart
```

## Προσθήκη Νέων Βιβλιοθηκών

Αν προσθέσεις βιβλιοθήκη στο `requirements.txt`, κάνε rebuild τον container:

```bash
docker-compose down
docker-compose up --build -d
```

