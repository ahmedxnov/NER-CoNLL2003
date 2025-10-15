import joblib
from src.utils.constants import _SHAPE_RE, ROOT_DIR
from datasets.dataset_dict import DatasetDict


def _word_shape(word: str) -> str:
    return _SHAPE_RE.sub(
        lambda m: 'A' if m.group().isupper() else ('a' if m.group().islower() else '0'),
        word
    )

def _word_shape_compact(word: str) -> str:
    s = _word_shape(word)
    out = list()
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return ''.join(out)

def _len_bin(n: int) -> str:
    if n <= 2: return 'L<=2'
    if n <= 4: return 'L<=4'
    if n <= 8: return 'L<=8'
    if n <= 12: return 'L<=12'
    return 'L>12'

def word2features(HF_sentence : dict, j : int, pos_labels : list[str], chunk_labels : list[str]) -> dict:
    word = HF_sentence['tokens'][j]
    features = {
        # word-level features
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(), # is the word titlecased(starts with a capital letter)?
        'word.isdigit()': word.isdigit(), # is the word all digits?
        'word.isalpha()': word.isalpha(), # is the word all alphabetic?
        'word.isalnum()': word.isalnum(), # is the word all alphanumeric(mix of alphabet and numbers only)?
        'has_quote': "'" in word or "’" in word, 
        'has_hyphen': '-' in word,
        'word_has_dot': '.' in word,
        'word_shape': _word_shape(word), # shape of the word ("Xxxx", "xxxx", "XxXx", "d", "dd", etc...)
        'word_shape_compact': _word_shape_compact(word), # compacted shape of the word (e.g., "Xx", "x", "XxXx", "d", "d", etc.)
        'word_len_bin': _len_bin(len(word)), # create ranges of lengths for the word
        'postag': pos_labels[HF_sentence['pos_tags'][j]], # part-of-speech tag of the word
        'chunktag': chunk_labels[HF_sentence['chunk_tags'][j]], # syntactic chunk tag of the word
        
        # Morphological Features (substrings)
        'prefix_2': word[:2], # first two characters of the word
        'prefix_3': word[:3], # first three characters of the word
        'suffix_2': word[-2:], # last two characters of the word
        'suffix_3': word[-3:], # last three characters of the word
        
        # Contextual Features (Neighbor Tokens)
        'prev_word_pos': pos_labels[HF_sentence['pos_tags'][j-1]] if j > 0 else 'BOS_POS',
        'next_word_pos': pos_labels[HF_sentence['pos_tags'][j+1]] if j != len(HF_sentence['tokens']) - 1 else 'EOS_POS',
        'prev_word_chunk': chunk_labels[HF_sentence['chunk_tags'][j-1]] if j > 0 else 'BOS_CHUNK',
        'next_word_chunk': chunk_labels[HF_sentence['chunk_tags'][j+1]] if j != len(HF_sentence['tokens']) - 1 else 'EOS_CHUNK'
    }
    return features

def process_sentence_chunk_optimized(sentences : list[dict], pos_labels : list[str], chunk_labels : list[str], ner_labels : list[str]) -> tuple[list[list[dict]], list[list[str]]]:
    chunk_X = list()
    chunk_y = list()
    
    for sentence in sentences:
        sentence_x = list()
        sentence_y = list()
        for j in range(len(sentence['tokens'])):
            sentence_x.append(word2features(sentence, j, pos_labels, chunk_labels))
            sentence_y.append(ner_labels[sentence['ner_tags'][j]])
        chunk_X.append(sentence_x)
        chunk_y.append(sentence_y)
    
    return chunk_X, chunk_y

def prepare_dataset_crf_format(ds: DatasetDict, split: str, pos_labels: list[str], chunk_labels: list[str], ner_labels: list[str]) -> tuple[list[list[dict]], list[list[str]]]:
    from multiprocessing import Pool, cpu_count
    from functools import partial
    from src.utils.helpers import create_chunks
    
    sentences = [ds[split][i] for i in range(ds[split].num_rows)]
    chunks = create_chunks(sentences, cpu_count())
    
    # fixating all values since pool only expects chunks
    process_chunk_with_labels = partial(
        process_sentence_chunk_optimized, 
        pos_labels=pos_labels, 
        chunk_labels=chunk_labels, 
        ner_labels=ner_labels
    )
    
    with Pool() as pool:
        results = pool.map(process_chunk_with_labels, chunks)
    
    X = list()
    y = list()
    for chunk_X, chunk_y in results:
        X.extend(chunk_X)
        y.extend(chunk_y)
    
    return X, y

def save_dataset_crf_format(X : list[list[dict]], y : list[list[str]], split : str):
    out_dir = ROOT_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = out_dir / f"conll2003_{split}_crf_format.pkl"

    joblib.dump((X, y), file_path)
    print(f"Saved {split} set in CRF-friendly format to: {file_path.resolve()}")