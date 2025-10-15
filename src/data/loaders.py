import joblib
from pathlib import Path
from src.utils.constants import SPLIT
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

def load_crf_dataset(split: str) -> tuple[list[list[dict]], list[list[str]]]:
    if split not in SPLIT:
        raise ValueError(f"Invalid split '{split}'. Must be one of {SPLIT}.")
    file_path = Path("data") / f"conll2003_{split}_crf_format.pkl"
    try:
        return joblib.load(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CRF-formatted dataset for split '{split}' not found at {file_path}. Please run the dataset preparation script first.")

def load_huggingface_dataset() -> DatasetDict:
    return load_dataset("conll2003")