# Prompt για εμπλουτισμό διαφανειών αποτελεσμάτων (Gemini)

Αντιγράψε το παρακάτω και στείλε το στο Gemini μαζί με τα αρχεία που ζητά (ή πέσε στο context του project).

---

## Αντικειμενικό

Έχω μια παρουσίαση (HTML slides) για αποτελέσματα ML pipeline (LOSO emotion recognition). Θέλω να **εμπλουτίσεις τις διαφάνειες 4 και 5** ώστε να μην εμφανίζεται μόνο η **Accuracy**, αλλά και **UAR (Unweighted Average Recall)**, **F1 weighted**, **F1 macro**, και αν χωράει **Precision/Recall weighted**, με μορφή mean ± std. Τα δεδομένα τα έχω ήδη· πρέπει απλά να τα αντλήσεις από τα αρχεία και να τα βάλεις στα HTML. Προαιρετικά: **μία ακόμα διαφάνεια** (νέα σελίδα) που να δείχνει **αποτελέσματα ανά fold** (ενδιάμεσα), π.χ. πίνακας ή σύντομη σύνοψη ανά fold_0 … fold_4.

---

## Δεδομένα – πηγές αλήθειας

**1) LOSO aggregated (μέσος όρος 5 folds)**  
- Αρχείο: `workflows/iemocap_loso/results/loso_summary.txt`  
- Μέτρα (κάθε ένα με mean ± std ανά μοντέλο):  
  **Accuracy**, **F1_weighted**, **F1_macro**, **UAR**, **Precision_weighted**, **Recall_weighted**  
- Μοντέλα: dtr, knn, logistic, nb, rf, svm, xgb (ίδια σειρά/ονομασία όπως στα τρέχοντα slides).

**2) Zero-shot (CREMAD) ανά fold**  
- Ένα CSV ανά fold:  
  `workflows/iemocap_loso/one_shot_results/fold_0/evaluation_summary.csv`  
  … μέχρι `fold_4/evaluation_summary.csv`  
- Κάθε CSV έχει columns: model, accuracy, f1_weighted, f1_macro, uar, precision_weighted, recall_weighted.  
- Για τη σύνοψη zero-shot: υπολόγισε **μέσο όρο** των 5 folds ανά μοντέλο και ανά μετρική (όχι μόνο accuracy).

---

## Αρχεία που πρέπει να τροποποιηθούν/δημιουργηθούν

- **page4.html** – Αποτελέσματα LOSO (IEMOCAP).  
  Εμπλούτισέ το ώστε να φαίνονται τουλάχιστον: **Accuracy**, **F1 (weighted)**, **UAR**, και αν χωράει και F1_macro ή Precision/Recall. Μορφή: mean ± std (π.χ. 56.38% ± 3.4%). Διατήρησε το ίδιο στυλ (Tailwind, πίνακες, ίδιο layout).

- **page5.html** – Zero-shot (CREMAD).  
  Ίδιο πνεύμα: μην μένεις μόνο στην Accuracy. Πρόσθεσε στήλες ή μικρούς πίνακες για **F1 weighted**, **UAR**, και όποια άλλα μέτρα χωράνε. Τα CREMAD νούμερα να είναι **μέσος όρος των 5 folds** (από τα evaluation_summary.csv ανά fold).

- **Προαιρετικά – νέα διαφάνεια (π.χ. page4b.html ή ενσωματωμένα στο page4):**  
  Μία σελίδα (ή τμήμα) που να αναφέρεται **ανά fold** (fold_0 … fold_4): π.χ. πίνακας με Accuracy / UAR / F1 ανά fold και ανά μοντέλο, ή σύντομη λίστα “ανά fold” ώστε να φαίνεται η ενδιάμεση διακύμανση. Δεδομένα: από τα ίδια αρχεία (ανά fold reports ή τα one_shot evaluation_summary ανά fold).

---

## Στυλ και περιορισμοί

- Διατήρησε **ακριβώς** το υπάρχον στυλ: ίδιο slide-container (1280×720), ίδια γραμματοσειρά (Montserrat / Open Sans), ίδιο χρωματικό (indigo/slate/gray, χωρίς πολύ “fancy” στοιχεία).
- Μην βάζεις εικονίδια σε λίστες· χρησιμοποίησε **bullets** (list-disc).
- Όλα τα νούμερα να προέρχονται **μόνο** από τα αρχεία που ανέφερα (loso_summary.txt και evaluation_summary.csv ανά fold). Μορφή ποσοστών: π.χ. 0.5638 → 56.38% και std 0.0339 → ± 3.4%.
- Γλώσσα: **Ελληνικά** για τίτλους και κείμενα (Accuracy, F1, UAR μπορούν να μένουν στα Αγγλικά αν έτσι είναι τώρα).

---

## Σύνοψη εντολών

1. Πάρε τα μέτρα **Accuracy, F1_weighted, F1_macro, UAR, Precision_weighted, Recall_weighted** από `loso_summary.txt` και βάλτα στο **page4.html** (πίνακες/στήλες, mean ± std).
2. Πάρε τα ίδια μέτρα από τα **5 evaluation_summary.csv** (zero-shot CREMAD), υπολόγισε μέσο όρο ανά μοντέλο, και εμπλούτισε το **page5.html** (όχι μόνο Accuracy).
3. Προαιρετικά: πρόσθεσε **μία διαφάνεια** (ή τμήμα) με αποτελέσματα **ανά fold** (ενδιάμεσα), χρησιμοποιώντας τα ίδια αρχεία.
4. Κράτα το layout και το στυλ των υπαρχόντων σελίδων· μην αλλάξεις γραμματοσειρές, χρώματα ή δομή slide.

Μόλις τελειώσεις, έξοδος: τα τροποποιημένα **page4.html** και **page5.html** (και προαιρετικά το νέο αρχείο ή το τμήμα “ανά fold”) με όλες τις αλλαγές ενσωματωμένες.
