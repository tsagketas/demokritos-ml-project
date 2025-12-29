from src.preprocessing.audio_preprocess import preprocess_audio
import os

example_file = input("Δώσε το πλήρες path του .wav αρχείου: ").strip()
if not os.path.isfile(example_file):
    print(f"Σφάλμα: Το αρχείο {example_file} δεν βρέθηκε!")
    exit(1)

output_file = os.path.join("processed", os.path.basename(example_file))
frames, sr = preprocess_audio(example_file, save_path=output_file)

print("\n=== Preprocessing Results ===")
print("Sampling rate:", sr)
print("Frames shape (n_frames, frame_length):", frames.shape)
print("Frame duration (ms):", frames.shape[1]/sr*1000)
print("Number of frames:", frames.shape[0])
print("Processed file saved at:", output_file)
print("=============================")
