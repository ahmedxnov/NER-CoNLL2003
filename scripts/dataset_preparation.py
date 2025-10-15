from src.data.loaders import load_huggingface_dataset
from src.data.preprocessing import prepare_dataset_crf_format, save_dataset_crf_format
from src.utils.constants import SPLIT

def main():
    ds = load_huggingface_dataset()
    
    for split in SPLIT:
        X, y = prepare_dataset_crf_format(ds, split)
        save_dataset_crf_format(X, y, split)

if __name__ == "__main__":
    main()
