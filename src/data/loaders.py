from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

def load_huggingface_dataset() -> DatasetDict:
    # Load CoNLL2003 dataset from HuggingFace
    return load_dataset("conll2003", trust_remote_code=True)
