from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset
from src.data.preprocessing import prepare_dataset_crf_format, save_dataset_crf_format

def main():
    ds = load_dataset("conll2003")
    
    for split in ["train", "validation", "test"]:
        X, y = prepare_dataset_crf_format(ds, split)
        save_dataset_crf_format(X, y, split)

if __name__ == "__main__":
    main()
