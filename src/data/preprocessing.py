from src.utils.constants import SHAPE_RE, SPLIT
from datasets.dataset_dict import DatasetDict
import time

def _word_shape(word: str) -> str:
    return SHAPE_RE.sub(
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
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:], # England -> and
        'word[-2:]': word[-2:], # Germany -> ny
        'word.isupper()': word.isupper(), # is it AAAA
        'word.istitle()': word.istitle(), # is it Aaaa
        'word.isdigit()': word.isdigit(), # is it 1234
        'word_has_dot': '.' in word,
        'word_has_hyphen': '-' in word,
        'word.isalpha()': word.isalpha(),
        'word.isalnum()': word.isalnum(),
        'postag': pos_labels[HF_sentence['pos_tags'][j]], # NNP
        'postag[:2]': pos_labels[HF_sentence['pos_tags'][j]][:2], # NN
        'chunktag': chunk_labels[HF_sentence['chunk_tags'][j]], # B-NP
        'word.shape()': _word_shape(word),
        'word.shape().compact()': _word_shape_compact(word),
        'word.len_bin()': _len_bin(len(word)),
    }
    if j > 0:
        prev_word = HF_sentence['tokens'][j - 1]
        features.update({
            'prev_word.lower()': prev_word.lower(),
            'prev_word[-3:]': prev_word[-3:], # England -> and
            'prev_word[-2:]': prev_word[-2:], # Germany -> ny
            'prev_word.istitle()': prev_word.istitle(),
            'prev_word.isupper()': prev_word.isupper(),
            'prev_word.isdigit()': prev_word.isdigit(),
            'pos_tag[prev_word]': pos_labels[HF_sentence['pos_tags'][j - 1]], # NNP
            'pos_tag[prev_word][:2]': pos_labels[HF_sentence['pos_tags'][j - 1]][:2], # NN
            'chunktag[prev_word]': chunk_labels[HF_sentence['chunk_tags'][j - 1]], # B-NP
            'word.shape()[prev_word]': _word_shape(prev_word),
            'word.shape().compact()[prev_word]': _word_shape_compact(prev_word),
            'word.len_bin()[prev_word]': _len_bin(len(prev_word)),
        })
    else:
        features['BOS'] = True
        
    
    if j < len(HF_sentence['tokens']) - 1:
        next_word = HF_sentence['tokens'][j + 1]
        features.update({
            'next_word.lower()': next_word.lower(),
            'next_word[-3:]': next_word[-3:], # England -> and
            'next_word[-2:]': next_word[-2:], # Germany -> ny
            'next_word.istitle()': next_word.istitle(),
            'next_word.isupper()': next_word.isupper(),
            'next_word.isdigit()': next_word.isdigit(),
            'pos_tag[next_word]': pos_labels[HF_sentence['pos_tags'][j + 1]], # NNP
            'pos_tag[next_word][:2]': pos_labels[HF_sentence['pos_tags'][j + 1]][:2], # NN
            'chunktag[next_word]': chunk_labels[HF_sentence['chunk_tags'][j + 1]], # B-NP
            'word.shape()[next_word]': _word_shape(next_word),
            'word.shape().compact()[next_word]': _word_shape_compact(next_word),
            'word.len_bin()[next_word]': _len_bin(len(next_word)),
        })
    else:
        features['EOS'] = True
    
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


def hf_to_crf(ds : DatasetDict):
    # Extract labels once (they're constant across all splits)
    pos_labels = ds["train"].features["pos_tags"].feature.names
    chunk_labels = ds["train"].features["chunk_tags"].feature.names
    ner_labels = ds["train"].features["ner_tags"].feature.names
    
    items = dict()
    for split in SPLIT:
        print(f"\nProcessing {split} split ({ds[split].num_rows} sentences)...")
        start_time = time.time()
        
        # Use the parallelized prepare function with pre-extracted labels
        X, y = prepare_dataset_crf_format(ds, split, pos_labels, chunk_labels, ner_labels)
        items[split] = (X, y)
        
        elapsed = time.time() - start_time
        print(f"Completed {split} preprocessing in {elapsed:.2f} seconds")
    
    print("\nAll splits processed successfully!")
    return items