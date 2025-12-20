"""
Feature Extraction Script for IEMOCAP Dataset
Uses librosa and pyAudioAnalysis for audio feature extraction

Usage:
    python scripts/iemocap_feature_extraction.py
    
    # Process only first 100 files (for testing)
    python scripts/iemocap_feature_extraction.py --max-files 100
    
    # Custom paths
    python scripts/iemocap_feature_extraction.py --csv-path path/to/csv --base-path path/to/dataset --output-path path/to/output.pkl

Features extracted:
    - Librosa: MFCC, spectral features, chroma, tonnetz, tempo, RMS energy, etc.
    - pyAudioAnalysis: Short-term audio features (ZCR, energy, spectral features, MFCC, chroma)
    
Output:
    - Saves features as both .pkl (for Python) and .csv (for inspection)
    - Includes metadata: session, method, gender, emotion, n_annotators, agreement
"""

# ============================================================================
# IMPORTS - Εισαγωγή των απαραίτητων βιβλιοθηκών
# ============================================================================
import os  # Για διαχείριση αρχείων και paths
import sys  # Για system operations
import pandas as pd  # Για διαχείριση δεδομένων σε DataFrame format
import numpy as np  # Για numerical operations και arrays
import librosa  # Βιβλιοθήκη για audio analysis - εξαγωγή audio features
import librosa.feature  # Module για εξαγωγή specific audio features
import warnings  # Για suppression of warnings
from tqdm import tqdm  # Για progress bars κατά την επεξεργασία
import pickle  # Για serialization/deserialization των features
from pathlib import Path  # Για διαχείριση file paths (cross-platform)
import argparse  # Για command-line arguments parsing

# Try to import pyAudioAnalysis
# ΣΗΜΕΙΩΣΗ: Το pyAudioAnalysis 0.3.14 έχει διαφορετική δομή από παλαιότερες εκδόσεις
# Χρησιμοποιούμε ShortTermFeatures, MidTermFeatures και audioBasicIO modules
try:
    from pyAudioAnalysis.ShortTermFeatures import feature_extraction as st_feature_extraction
    from pyAudioAnalysis.MidTermFeatures import mid_feature_extraction
    from pyAudioAnalysis import audioBasicIO as aIO  # Audio file I/O operations
    PYAUDIO_AVAILABLE = True  # Flag που δείχνει αν είναι διαθέσιμο
except ImportError as e:
    # Αν δεν είναι εγκατεστημένο ή λείπει dependency, συνεχίζουμε μόνο με librosa
    print(f"Warning: pyAudioAnalysis not available ({str(e)}). Only librosa features will be extracted.")
    PYAUDIO_AVAILABLE = False



# Αγνοούμε warnings για καθαρότερο output
warnings.filterwarnings('ignore')

# Προσθέτουμε το project root στο Python path
# Αυτό επιτρέπει imports από άλλα modules του project
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def extract_librosa_features(audio_path, sr=22050):
    """
    Εξάγει audio features χρησιμοποιώντας τη βιβλιοθήκη librosa.
    
    Η librosa είναι μια εξειδικευμένη βιβλιοθήκη για audio analysis που προσφέρει
    πολλά features που είναι χρήσιμα για emotion recognition:
    - MFCC: Mel-frequency cepstral coefficients (κύρια features για speech)
    - Spectral features: περιγράφουν το frequency content
    - Chroma: tonal features που σχετίζονται με τη μουσική αρμονία
    - Tempo: ρυθμός του audio
    
    Parameters:
    -----------
    audio_path : str
        Η διαδρομή προς το audio file (.wav format)
    sr : int
        Sample rate (samples per second) - default 22050 Hz
        Το IEMOCAP dataset έχει 16kHz, αλλά το librosa μπορεί να resample
    
    Returns:
    --------
    dict : Dictionary με όλα τα extracted features
        Κάθε feature έχει mean και std values (statistics over time)
    """
    try:
        # ========================================================================
        # 1. LOAD AUDIO FILE - Φόρτωση του audio file
        # ========================================================================
        # Το librosa.load() διαβάζει το audio file και το μετατρέπει σε numpy array
        # y: audio time series (array με τα audio samples)
        # sr: sample rate (πόσα samples per second)
        # duration=None: φορτώνουμε όλο το audio (όχι μόνο ένα κομμάτι)
        y, sr = librosa.load(audio_path, sr=sr, duration=None)
        
        # ========================================================================
        # 2. BASIC AUDIO PROPERTIES - Βασικές ιδιότητες του audio
        # ========================================================================
        # Duration: πόσο χρόνο διαρκεί το audio (σε seconds)
        # Χρήσιμο για normalization και context
        duration = librosa.get_duration(y=y, sr=sr)
        
        # ========================================================================
        # 3. ZERO CROSSING RATE (ZCR) - Ρυθμός μηδενικών διαβάσεων
        # ========================================================================
        # Το ZCR μετράει πόσες φορές το signal περνάει από το μηδέν
        # Υψηλό ZCR = πιο noisy/harsh sound (π.χ. anger)
        # Χαμηλό ZCR = πιο smooth sound (π.χ. sadness)
        # Υπολογίζουμε mean και std για να πάρουμε statistics over time
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)  # Μέσος όρος
        zcr_std = np.std(zcr)    # Τυπική απόκλιση (variability)
        
        # ========================================================================
        # 4. SPECTRAL FEATURES - Χαρακτηριστικά του frequency spectrum
        # ========================================================================
        # Αυτά τα features περιγράφουν πώς κατανέμεται η ενέργεια στα frequencies
        
        # Spectral Centroid: το "κέντρο βάρους" του spectrum
        # Υψηλό = περισσότερα high frequencies (bright sound, excitement)
        # Χαμηλό = περισσότερα low frequencies (dark sound, sadness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroids)
        spectral_centroid_std = np.std(spectral_centroids)
        
        # Spectral Rolloff: το frequency κάτω από το οποίο βρίσκεται το 85% της ενέργειας
        # Μετράει το "cutoff" frequency - πόσο "wide" είναι το spectrum
        # Χρήσιμο για να διακρίνουμε voiced vs unvoiced sounds
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        spectral_rolloff_std = np.std(spectral_rolloff)
        
        # Spectral Bandwidth: το "εύρος" του spectrum
        # Μετράει πόσο spread out είναι η ενέργεια στα frequencies
        # Χρήσιμο για να διακρίνουμε pure tones από noisy sounds
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_std = np.std(spectral_bandwidth)
        
        # ========================================================================
        # 5. MFCC FEATURES - Mel-Frequency Cepstral Coefficients
        # ========================================================================
        # Τα πιο σημαντικά features για speech emotion recognition!
        # Το MFCC μιμείται την ανθρώπινη ακοή (Mel scale)
        # 13 coefficients είναι το standard (πρώτος = energy, υπόλοιποι = spectral shape)
        # Χρήσιμα γιατί capture το timbre και το spectral envelope
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # axis=1: υπολογίζουμε statistics για κάθε coefficient over time
        mfcc_means = np.mean(mfccs, axis=1)  # Mean για κάθε MFCC coefficient
        mfcc_stds = np.std(mfccs, axis=1)    # Std για κάθε MFCC coefficient
        
        # ========================================================================
        # 6. CHROMA FEATURES - Χρωματικά χαρακτηριστικά
        # ========================================================================
        # Το chroma περιγράφει το "tonal content" - ποια notes/tones υπάρχουν
        # 12 features (ένα για κάθε note στο chromatic scale: C, C#, D, D#, ...)
        # Χρήσιμα για να capture τη μουσική/tonal ποιότητα του voice
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)  # Mean για κάθε chroma bin
        chroma_std = np.std(chroma, axis=1)    # Std για κάθε chroma bin
        
        # ========================================================================
        # 7. TONNETZ FEATURES - Tonal Network Features
        # ========================================================================
        # Το Tonnetz είναι ένας 6D space που περιγράφει τις harmonic relationships
        # Χρήσιμο για να capture complex tonal patterns
        # Χρησιμοποιούμε μόνο το harmonic component (χωρίς percussive)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)
        
        # ========================================================================
        # 8. TEMPO - Ρυθμός/Τέμπο
        # ========================================================================
        # Μετράει τον ρυθμό του audio (beats per minute)
        # Γρήγορο tempo = excitement, anger
        # Αργό tempo = sadness, calmness
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # ========================================================================
        # 9. RMS ENERGY - Root Mean Square Energy
        # ========================================================================
        # Μετράει τη μέση ενέργεια/δύναμη του signal
        # Υψηλό RMS = δυνατή φωνή (anger, excitement)
        # Χαμηλό RMS = αδύναμη φωνή (sadness, fear)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # ========================================================================
        # 10. SPECTRAL CONTRAST - Αντίθεση στο spectrum
        # ========================================================================
        # Μετράει τη διαφορά μεταξύ peaks και valleys στο spectrum
        # 7 features (ένα για κάθε frequency subband)
        # Χρήσιμο για να διακρίνουμε harmonic από noisy sounds
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        spectral_contrast_std = np.std(spectral_contrast, axis=1)
        
        # ========================================================================
        # 11. MEL SPECTROGRAM - Mel-scale frequency representation
        # ========================================================================
        # Το mel spectrogram είναι μια frequency representation που μιμείται
        # την ανθρώπινη ακοή (Mel scale - logarithmic frequency scale)
        # 128 mel bands = 128 διαφορετικά frequency ranges
        # Παίρνουμε μόνο τα πρώτα 20 για να μην έχουμε πολλά features
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_mean = np.mean(mel_spec, axis=1)
        mel_spec_std = np.std(mel_spec, axis=1)
        
        # ========================================================================
        # 12. POLY FEATURES - Polynomial features
        # ========================================================================
        # Fit ένα polynomial στο spectral envelope
        # 2 features (coefficients) - capture το overall spectral shape
        poly_features = librosa.feature.poly_features(y=y, sr=sr)
        poly_features_mean = np.mean(poly_features, axis=1)
        poly_features_std = np.std(poly_features, axis=1)
        
        # ========================================================================
        # 13. BUILD FEATURE DICTIONARY - Κατασκευή dictionary με όλα τα features
        # ========================================================================
        # Οργανώνουμε όλα τα features σε ένα dictionary
        # Κάθε feature έχει όνομα που περιγράφει τι είναι (π.χ. 'mfcc_0_mean')
        features = {
            'duration': duration,  # Διάρκεια σε seconds
            'zcr_mean': zcr_mean,
            'zcr_std': zcr_std,
            'spectral_centroid_mean': spectral_centroid_mean,
            'spectral_centroid_std': spectral_centroid_std,
            'spectral_rolloff_mean': spectral_rolloff_mean,
            'spectral_rolloff_std': spectral_rolloff_std,
            'spectral_bandwidth_mean': spectral_bandwidth_mean,
            'spectral_bandwidth_std': spectral_bandwidth_std,
            'tempo': tempo,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
        }
        
        # Προσθέτουμε τα MFCC features (13 coefficients × 2 statistics = 26 features)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = mfcc_means[i]
            features[f'mfcc_{i}_std'] = mfcc_stds[i]
        
        # Προσθέτουμε τα chroma features (12 notes × 2 statistics = 24 features)
        for i in range(12):
            features[f'chroma_{i}_mean'] = chroma_mean[i]
            features[f'chroma_{i}_std'] = chroma_std[i]
        
        # Προσθέτουμε τα tonnetz features (6 dimensions × 2 statistics = 12 features)
        for i in range(6):
            features[f'tonnetz_{i}_mean'] = tonnetz_mean[i]
            features[f'tonnetz_{i}_std'] = tonnetz_std[i]
        
        # Προσθέτουμε τα spectral contrast features (7 subbands × 2 statistics = 14 features)
        for i in range(7):
            features[f'spectral_contrast_{i}_mean'] = spectral_contrast_mean[i]
            features[f'spectral_contrast_{i}_std'] = spectral_contrast_std[i]
        
        # Προσθέτουμε mel spectrogram features (πρώτα 20 bands × 2 statistics = 40 features)
        # Χρησιμοποιούμε min() για safety αν το mel_spec έχει λιγότερα bands
        for i in range(min(20, len(mel_spec_mean))):
            features[f'mel_spec_{i}_mean'] = mel_spec_mean[i]
            features[f'mel_spec_{i}_std'] = mel_spec_std[i]
        
        # Προσθέτουμε poly features (2 coefficients × 2 statistics = 4 features)
        for i in range(min(2, len(poly_features_mean))):
            features[f'poly_{i}_mean'] = poly_features_mean[i]
            features[f'poly_{i}_std'] = poly_features_std[i]
        
        return features
        
    except Exception as e:
        # Αν προκύψει error, το print και return None
        # Αυτό επιτρέπει στο script να συνεχίσει με τα υπόλοιπα files
        print(f"Error processing {audio_path}: {str(e)}")
        return None


def extract_pyaudioanalysis_features(audio_path, win_size=0.050, step_size=0.025, midterm_win=1.0, midterm_step=0.1):
    """
    Εξάγει audio features χρησιμοποιώντας τη βιβλιοθήκη pyAudioAnalysis.
    
    Το pyAudioAnalysis προσφέρει:
    - Short-term features: υπολογίζονται σε μικρά windows (50ms)
    - Mid-term features: υπολογίζονται σε μεγαλύτερα windows (1s) με statistics
    
    Parameters:
    -----------
    audio_path : str
        Η διαδρομή προς το audio file
    win_size : float
        Μέγεθος παραθύρου για short-term features σε seconds (default: 0.050 = 50ms)
    step_size : float
        Βήμα μεταξύ windows για short-term features σε seconds (default: 0.025 = 25ms)
    midterm_win : float
        Μέγεθος παραθύρου για mid-term features σε seconds (default: 1.0 = 1s)
    midterm_step : float
        Βήμα μεταξύ windows για mid-term features σε seconds (default: 0.1 = 100ms)
    
    Returns:
    --------
    dict : Dictionary με όλα τα extracted features
        Κάθε feature έχει mean, std, min, max values
    """
    # Αν δεν είναι διαθέσιμο το pyAudioAnalysis, return None
    if not PYAUDIO_AVAILABLE:
        return None
    
    try:
        # ========================================================================
        # 1. READ AUDIO FILE - Διάβασμα του audio file
        # ========================================================================
        # Το pyAudioAnalysis έχει δικό του audio reader
        # Fs: sample rate (samples per second)
        # x: audio signal (numpy array)
        [Fs, x] = aIO.read_audio_file(audio_path)
        
        features = {}
        
        # ========================================================================
        # 2. EXTRACT SHORT-TERM FEATURES - Εξαγωγή short-term features
        # ========================================================================
        # Το st_feature_extraction() διαιρεί το audio σε overlapping windows
        # και υπολογίζει features για κάθε window
        # win_size και step_size πρέπει να είναι σε samples (όχι seconds)
        # 
        # ΣΗΜΕΙΩΣΗ: Στο pyAudioAnalysis 0.3.14, το feature_extraction function
        # επιστρέφει (features_matrix, feature_names)
        features_matrix, feature_names = st_feature_extraction(
            x,  # Audio signal
            Fs,  # Sample rate
            int(win_size * Fs),  # Window size σε samples (50ms * samples/sec)
            int(step_size * Fs)  # Step size σε samples (25ms * samples/sec)
        )
        # Το features_matrix είναι 2D array: [num_features, num_windows]
        # Κάθε column είναι ένα window, κάθε row είναι ένα feature
        
        # Feature names για short-term features
        if feature_names is None or len(feature_names) == 0:
            feature_names = [
                'zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread',
                'spectral_entropy', 'spectral_flux', 'spectral_rolloff', 'mfcc_1', 'mfcc_2',
                'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9',
                'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_1', 'chroma_2',
                'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8',
                'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12', 'chroma_std'
            ]
        
        # Handle edge case: αν το features_matrix είναι 1D, το κάνουμε 2D
        if features_matrix.ndim == 1:
            features_matrix = features_matrix.reshape(-1, 1)
        
        # Υπολογίζουμε statistics για short-term features
        num_features = min(len(feature_names), features_matrix.shape[0])
        for i in range(num_features):
            feature_values = features_matrix[i, :]
            if len(feature_values) == 0:
                continue
            feature_values = feature_values[np.isfinite(feature_values)]
            if len(feature_values) == 0:
                continue
            
            # Short-term features με prefix 'pyaudio_st_'
            features[f'pyaudio_st_{feature_names[i]}_mean'] = np.mean(feature_values)
            features[f'pyaudio_st_{feature_names[i]}_std'] = np.std(feature_values)
            features[f'pyaudio_st_{feature_names[i]}_min'] = np.min(feature_values)
            features[f'pyaudio_st_{feature_names[i]}_max'] = np.max(feature_values)
        
        # ========================================================================
        # 3. EXTRACT MID-TERM FEATURES - Εξαγωγή mid-term features
        # ========================================================================
        # Τα mid-term features υπολογίζονται σε μεγαλύτερα windows (1s)
        # και δίνουν statistics πάνω από μεγαλύτερα time segments
        # Αυτό capture longer-term patterns στο audio
        try:
            # Το mid_feature_extraction επιστρέφει (mid_features, mid_feature_names, mid_time)
            result = mid_feature_extraction(
                x,  # Audio signal
                Fs,  # Sample rate
                int(midterm_win * Fs),  # Mid-term window size σε samples (1s)
                int(midterm_step * Fs),  # Mid-term step size σε samples (100ms)
                int(win_size * Fs),  # Short-term window size (για υπολογισμό)
                int(step_size * Fs)  # Short-term step size (για υπολογισμό)
            )
            
            # Handle different return formats
            if len(result) == 2:
                mid_features, mid_feature_names = result
            elif len(result) == 3:
                mid_features, mid_feature_names, mid_time = result
            else:
                raise ValueError(f"Unexpected return format from mid_feature_extraction: {len(result)} values")
            
            # Αν το mid_features είναι 2D array, υπολογίζουμε statistics
            if mid_features is not None and len(mid_features) > 0:
                if mid_features.ndim == 1:
                    mid_features = mid_features.reshape(-1, 1)
                
                # Αν δεν έχουμε feature names, δημιουργούμε generic ones
                if mid_feature_names is None or len(mid_feature_names) == 0:
                    mid_feature_names = [f'mid_feature_{i}' for i in range(mid_features.shape[0])]
                
                num_mid_features = min(len(mid_feature_names), mid_features.shape[0])
                for i in range(num_mid_features):
                    mid_values = mid_features[i, :]
                    if len(mid_values) == 0:
                        continue
                    mid_values = mid_values[np.isfinite(mid_values)]
                    if len(mid_values) == 0:
                        continue
                    
                    # Mid-term features με prefix 'pyaudio_mt_'
                    features[f'pyaudio_mt_{mid_feature_names[i]}_mean'] = np.mean(mid_values)
                    features[f'pyaudio_mt_{mid_feature_names[i]}_std'] = np.std(mid_values)
                    features[f'pyaudio_mt_{mid_feature_names[i]}_min'] = np.min(mid_values)
                    features[f'pyaudio_mt_{mid_feature_names[i]}_max'] = np.max(mid_values)
        except Exception as e:
            # Αν τα mid-term features αποτύχουν, συνεχίζουμε μόνο με short-term
            print(f"Warning: Mid-term features extraction failed: {str(e)}")
        
        return features
        
    except Exception as e:
        print(f"Error processing {audio_path} with pyAudioAnalysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def extract_all_features(audio_path, base_path):
    """
    Εξάγει όλα τα features από ένα audio file χρησιμοποιώντας και τα δύο tools.
    
    Αυτή η συνάρτηση συνδυάζει τα features από librosa και pyAudioAnalysis
    για να πάρουμε μια πλήρη εικόνα του audio signal. Κάθε tool προσφέρει
    διαφορετική perspective και το συνδυασμό τους δίνει καλύτερα results.
    
    Parameters:
    -----------
    audio_path : str
        Relative path προς το audio file (από το base_path)
        Π.χ. "Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav"
    base_path : str
        Base path προς το IEMOCAP dataset
        Π.χ. "datasets/IEMOCAP_full_release"
    
    Returns:
    --------
    dict : Combined feature dictionary με όλα τα features
        None αν υπάρξει πρόβλημα
    """
    # ========================================================================
    # 1. BUILD FULL PATH - Κατασκευή της πλήρους διαδρομής
    # ========================================================================
    full_path = os.path.join(base_path, audio_path)
    
    # Ελέγχουμε αν το file υπάρχει
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        return None
    
    # ========================================================================
    # 2. EXTRACT LIBROSA FEATURES - Εξαγωγή features με librosa
    # ========================================================================
    librosa_features = extract_librosa_features(full_path)
    
    # ========================================================================
    # 3. EXTRACT PYAUDIOANALYSIS FEATURES - Εξαγωγή features με pyAudioAnalysis
    # ========================================================================
    pyaudio_features = None
    if PYAUDIO_AVAILABLE:
        pyaudio_features = extract_pyaudioanalysis_features(full_path)
    
    # ========================================================================
    # 4. COMBINE FEATURES - Συνδυασμός των features
    # ========================================================================
    # Αν και τα δύο απέτυχαν, return None
    if librosa_features is None and pyaudio_features is None:
        return None
    
    # Δημιουργούμε ένα combined dictionary
    combined_features = {}
    
    # Προσθέτουμε τα librosa features (αν υπάρχουν)
    if librosa_features:
        combined_features.update(librosa_features)
    
    # Προσθέτουμε τα pyAudioAnalysis features (αν υπάρχουν)
    if pyaudio_features:
        combined_features.update(pyaudio_features)
    
    return combined_features


def process_dataset(csv_path, base_path, output_path=None, max_files=None):
    """
    Επεξεργάζεται ολόκληρο το IEMOCAP dataset και εξάγει features.
    
    Αυτή είναι η κύρια συνάρτηση που διαβάζει το CSV με τα metadata,
    επεξεργάζεται κάθε audio file, εξάγει features, και τα αποθηκεύει.
    
    Parameters:
    -----------
    csv_path : str
        Διαδρομή προς το CSV file με τα metadata του dataset
        Πρέπει να έχει columns: session, method, gender, emotion, n_annotators, agreement, path
    base_path : str
        Base path προς το IEMOCAP dataset folder
    output_path : str
        Διαδρομή για αποθήκευση των features (default: datasets/iemocap_features.pkl)
    max_files : int
        Μέγιστος αριθμός files για επεξεργασία (None = όλα)
        Χρήσιμο για testing - π.χ. max_files=100 για γρήγορο test
    """
    # ========================================================================
    # 1. READ CSV FILE - Διάβασμα του CSV με τα metadata
    # ========================================================================
    print(f"Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total entries in CSV: {len(df)}")
    
    # ========================================================================
    # 2. FILTER DATA - Φιλτράρισμα των δεδομένων
    # ========================================================================
    # Αφαιρούμε entries με 'xxx' emotion (δεν υπάρχει consensus μεταξύ annotators)
    # Αυτά δεν είναι χρήσιμα για training
    xxx_count = len(df[df['emotion'] == 'xxx'])
    if xxx_count > 0:
        print(f"Filtering out {xxx_count} entries with 'xxx' emotion (no consensus)")
    df = df[df['emotion'] != 'xxx']
    print(f"Valid entries after filtering: {len(df)}")
    
    # Αν έχει οριστεί max_files, παίρνουμε μόνο τα πρώτα N files
    # Χρήσιμο για testing/development
    if max_files:
        df = df.head(max_files)
    
    print(f"Processing {len(df)} audio files...")
    
    # ========================================================================
    # 3. EXTRACT FEATURES - Εξαγωγή features για κάθε file
    # ========================================================================
    all_features = []  # List για τα successful extractions
    failed_files = []  # List για τα failed files (για debugging)
    
    # Χρησιμοποιούμε tqdm για progress bar
    # iterrows() επαναλαμβάνει κάθε row του DataFrame
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        audio_path = row['path']  # Relative path από το CSV
        
        # Εξάγουμε features για αυτό το audio file
        features = extract_all_features(audio_path, base_path)
        
        if features is not None:
            # ====================================================================
            # 4. ADD METADATA - Προσθήκη metadata στο features dictionary
            # ====================================================================
            # Προσθέτουμε τα metadata από το CSV για να ξέρουμε:
            # - session: ποια session (1-5)
            # - method: script ή improvisation
            # - gender: F (female) ή M (male)
            # - emotion: ground truth emotion label (ang, hap, sad, neu, fru, exc, fea, sur, dis, oth)
            # - n_annotators: πόσοι annotators αξιολόγησαν
            # - agreement: πόσοι συμφώνησαν (consensus)
            # - audio_path: το path για reference
            features['session'] = row['session']
            features['method'] = row['method']
            features['gender'] = row['gender']
            features['emotion'] = row['emotion']
            features['n_annotators'] = row['n_annotators']
            features['agreement'] = row['agreement']
            features['audio_path'] = audio_path
            
            # Προσθέτουμε στο list
            all_features.append(features)
        else:
            # Αν απέτυχε, το προσθέτουμε στο failed list
            failed_files.append(audio_path)
    
    # ========================================================================
    # 5. PRINT SUMMARY - Εκτύπωση summary των results
    # ========================================================================
    print(f"\nSuccessfully processed: {len(all_features)} files")
    print(f"Failed: {len(failed_files)} files")
    
    # Αν υπάρχουν failed files, τα print (για debugging)
    if len(failed_files) > 0:
        print("\nFailed files:")
        for f in failed_files[:10]:  # Δείχνουμε τα πρώτα 10
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    # ========================================================================
    # 6. CONVERT TO DATAFRAME - Μετατροπή σε pandas DataFrame
    # ========================================================================
    # Το DataFrame είναι πιο βολικό για analysis και manipulation
    # Κάθε row = ένα audio file, κάθε column = ένα feature
    features_df = pd.DataFrame(all_features)
    
    # ========================================================================
    # 7. SET OUTPUT PATH - Ορισμός της διαδρομής αποθήκευσης
    # ========================================================================
    if output_path is None:
        # Αν δεν δόθηκε output path, χρησιμοποιούμε default
        project_root = Path(__file__).parent.parent
        output_path = os.path.join(project_root, 'datasets', 'iemocap_features.pkl')
    
    # ========================================================================
    # 8. SAVE FEATURES - Αποθήκευση των features
    # ========================================================================
    # Αποθηκεύουμε σε pickle format (preserves data types, fast, efficient)
    print(f"\nSaving features to {output_path}...")
    features_df.to_pickle(output_path)
    
    # Επίσης αποθηκεύουμε σε CSV (για εύκολη inspection με Excel/Python)
    # Το CSV είναι human-readable αλλά χάνει κάποια precision
    csv_output = output_path.replace('.pkl', '.csv')
    features_df.to_csv(csv_output, index=False)
    print(f"Also saved as CSV: {csv_output}")
    
    # ========================================================================
    # 9. PRINT DETAILED SUMMARY - Εκτύπωση αναλυτικού summary
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"{'='*60}")
    print(f"Total files processed: {len(features_df)}")
    print(f"Total features extracted: {len(features_df.columns)}")
    
    # ========================================================================
    # 10. COUNT FEATURE TYPES - Καταμέτρηση των τύπων features
    # ========================================================================
    # Χωρίζουμε τα features σε κατηγορίες:
    # - librosa_features: features από librosa (όχι pyaudio_, όχι metadata)
    # - pyaudio_features: features από pyAudioAnalysis (ξεκινάνε με 'pyaudio_')
    # - metadata_features: metadata από το CSV (session, emotion, κτλ)
    librosa_features = [c for c in features_df.columns 
                       if not c.startswith('pyaudio_') 
                       and c not in ['session', 'method', 'gender', 'emotion', 
                                    'n_annotators', 'agreement', 'audio_path']]
    pyaudio_features = [c for c in features_df.columns if c.startswith('pyaudio_')]
    metadata_features = [c for c in features_df.columns 
                        if c in ['session', 'method', 'gender', 'emotion', 
                                'n_annotators', 'agreement', 'audio_path']]
    
    print(f"\nFeature breakdown:")
    print(f"  - Librosa features: {len(librosa_features)}")
    print(f"  - pyAudioAnalysis features: {len(pyaudio_features)}")
    print(f"  - Metadata features: {len(metadata_features)}")
    
    # Δείχνουμε sample feature names για να δούμε τι εξάγαμε
    print(f"\nSample feature names (first 15):")
    all_feat = librosa_features + pyaudio_features
    for feat in all_feat[:15]:
        print(f"  - {feat}")
    if len(all_feat) > 15:
        print(f"  ... and {len(all_feat) - 15} more")
    
    # ========================================================================
    # 11. EMOTION DISTRIBUTION - Κατανομή των emotions
    # ========================================================================
    # Δείχνουμε πόσα samples έχουμε για κάθε emotion
    # Αυτό είναι σημαντικό για να δούμε αν το dataset είναι balanced
    print(f"\nEmotion distribution:")
    emotion_counts = features_df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  - {emotion}: {count}")
    
    print(f"\n{'='*60}")
    
    return features_df


def main():
    """
    Κύριο function - entry point του script.
    
    Αυτή η συνάρτηση:
    1. Parse τα command-line arguments
    2. Ορίζει τα paths (default ή custom)
    3. Ελέγχει αν υπάρχουν τα αρχεία
    4. Καλεί τη process_dataset() για να κάνει την επεξεργασία
    """
    # ========================================================================
    # 1. SET UP ARGUMENT PARSER - Ρύθμιση command-line arguments
    # ========================================================================
    parser = argparse.ArgumentParser(
        description='Extract audio features from IEMOCAP dataset using librosa and pyAudioAnalysis'
    )
    
    # CSV path argument
    parser.add_argument(
        '--csv-path',
        type=str,
        default=None,
        help='Path to CSV file with dataset metadata (default: datasets/iemocap_full_dataset.csv)'
    )
    
    # Base path argument
    parser.add_argument(
        '--base-path',
        type=str,
        default=None,
        help='Base path to IEMOCAP dataset (default: datasets/IEMOCAP_full_release)'
    )
    
    # Output path argument
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output path for features (default: datasets/iemocap_features.pkl)'
    )
    
    # Max files argument (χρήσιμο για testing)
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to process (default: all files)'
    )
    
    # Sample rate argument (για future use)
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=22050,
        help='Sample rate for audio loading (default: 22050)'
    )
    
    # Parse τα arguments
    args = parser.parse_args()
    
    # ========================================================================
    # 2. SET PATHS - Ορισμός των διαδρομών
    # ========================================================================
    project_root = Path(__file__).parent.parent
    
    # CSV path: default ή custom
    if args.csv_path is None:
        csv_path = os.path.join(project_root, 'datasets', 'iemocap_full_dataset.csv')
    else:
        csv_path = args.csv_path
    
    # Base path: default ή custom
    if args.base_path is None:
        base_path = os.path.join(project_root, 'datasets', 'IEMOCAP_full_release')
    else:
        base_path = args.base_path
    
    # Output path: default ή custom
    if args.output_path is None:
        output_path = os.path.join(project_root, 'datasets', 'iemocap_features.pkl')
    else:
        output_path = args.output_path
    
    # ========================================================================
    # 3. VALIDATE PATHS - Επαλήθευση ότι υπάρχουν τα paths
    # ========================================================================
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    if not os.path.exists(base_path):
        print(f"Error: Dataset base path not found at {base_path}")
        return
    
    # ========================================================================
    # 4. PROCESS DATASET - Επεξεργασία του dataset
    # ========================================================================
    print(f"Processing IEMOCAP dataset...")
    print(f"CSV path: {csv_path}")
    print(f"Base path: {base_path}")
    print(f"Output path: {output_path}")
    if args.max_files:
        print(f"Max files: {args.max_files}")
    print()
    
    # Καλούμε τη process_dataset() που κάνει όλη τη δουλειά
    features_df = process_dataset(
        csv_path=csv_path,
        base_path=base_path,
        output_path=output_path,
        max_files=args.max_files
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()