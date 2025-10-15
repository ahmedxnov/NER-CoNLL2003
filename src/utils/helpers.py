from .constants import ROOT_DIR, SPLIT
import joblib

def create_chunks(data: list[dict], workers: int) -> list[list[dict]]:
    if not data:
        return []

    chunk_size = max(1, len(data) // workers)
    chunks = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks

def save_dataset_crf_format(items : dict):
    out_dir = ROOT_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for split, (X, y) in items.items():
        file_path = out_dir / f"conll2003_{split}_crf_format.pkl"

        joblib.dump((X, y), file_path)
        print(f"Saved {split} set in CRF-friendly format to: {file_path.resolve()}")

def load_crf_dataset(split: str) -> tuple[list[list[dict]], list[list[str]]]:
    if split not in SPLIT:
        raise ValueError(f"Invalid split '{split}'. Must be one of {SPLIT}.")
    file_path = ROOT_DIR / "data" / f"conll2003_{split}_crf_format.pkl"
    try:
        return joblib.load(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CRF-formatted dataset for split '{split}' not found at {file_path}. Please run the dataset preparation script first.")
