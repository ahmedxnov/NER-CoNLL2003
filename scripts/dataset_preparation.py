from src.data.loaders import load_huggingface_dataset
from src.data.preprocessing import prepare_dataset_crf_format, save_dataset_crf_format
from src.utils.constants import SPLIT
import time

def main():
    print("Loading CoNLL-2003 dataset...")
    ds = load_huggingface_dataset()
    
    # Extract labels once (they're constant across all splits)
    pos_labels = ds["train"].features["pos_tags"].feature.names
    chunk_labels = ds["train"].features["chunk_tags"].feature.names
    ner_labels = ds["train"].features["ner_tags"].feature.names
    
    for split in SPLIT:
        print(f"\nProcessing {split} split ({ds[split].num_rows} sentences)...")
        start_time = time.time()
        
        # Use the parallelized prepare function with pre-extracted labels
        X, y = prepare_dataset_crf_format(ds, split, pos_labels, chunk_labels, ner_labels)
        
        # Save the processed data
        save_dataset_crf_format(X, y, split)
        
        elapsed = time.time() - start_time
        print(f"Completed {split} in {elapsed:.2f} seconds")
    
    print("\nAll splits processed successfully!")

if __name__ == "__main__":
    main()
