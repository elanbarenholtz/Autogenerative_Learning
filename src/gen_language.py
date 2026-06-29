"""
Group D — natural language pipeline (char-level) + destroyed-structure control.

Works on any UTF-8 text file. Provides:
  - CharTokenizer (vocab from corpus)
  - contiguous train/val/test split
  - windowing for LM training
  - D2 controls that destroy long-range structure while keeping low-order surface
    stats: char n-gram resample, and local word-block shuffle.

No oracle (language has no closed-form next-token law); learnability is measured by
loss / gap-to-an-n-gram-floor, and productivity by an acceptability proxy (labeled soft).
"""
import random
from collections import defaultdict
from typing import List


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos.get(i, '') for i in ids)


def load_corpus(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def split_text(text: str, fracs=(0.8, 0.1, 0.1)):
    n = len(text)
    a = int(n * fracs[0]); b = a + int(n * fracs[1])
    return {'train': text[:a], 'val': text[a:b], 'test': text[b:]}


def make_windows(ids: List[int], seq_len=128, stride=None):
    stride = stride or seq_len
    return [ids[i:i + seq_len + 1] for i in range(0, len(ids) - seq_len - 1, stride)]


# ---- D2 destroyed-structure controls ----
def word_block_shuffle(text: str, block_words=5, seed=0) -> str:
    """Shuffle words within local blocks: destroys longer-range order, keeps local words."""
    rng = random.Random(seed)
    words = text.split(' ')
    out = []
    for i in range(0, len(words), block_words):
        blk = words[i:i + block_words]
        rng.shuffle(blk)
        out.extend(blk)
    return ' '.join(out)


def char_ngram_resample(ids: List[int], n=5, length=None, seed=0, vocab_size=None) -> List[int]:
    """Sample a new char stream preserving order-(n-1) statistics; destroys long-range structure."""
    rng = random.Random(seed)
    length = length or len(ids)
    ctx_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(ids) - n + 1):
        ctx = tuple(ids[i:i + n - 1])
        ctx_counts[ctx][ids[i + n - 1]] += 1
    out = list(ids[:n - 1])
    for _ in range(length - (n - 1)):
        ctx = tuple(out[-(n - 1):])
        nxt = ctx_counts.get(ctx)
        if not nxt:
            out.append(rng.choice(out[-50:] if out else [0]))
            continue
        syms, ws = zip(*nxt.items())
        out.append(rng.choices(syms, weights=ws, k=1)[0])
    return out


def build(path: str, seq_len=128, control=None, n_ngram=5, seed=0):
    """Return dict with tokenizer + windowed splits. control in {None,'word_shuffle','ngram'}."""
    text = load_corpus(path)
    tok = CharTokenizer(text)
    parts = split_text(text)
    if control == 'word_shuffle':
        parts = {k: word_block_shuffle(v, seed=seed) for k, v in parts.items()}
    out = {'tokenizer': tok, 'vocab_size': tok.vocab_size, 'control': control, 'splits': {}}
    for k, v in parts.items():
        ids = tok.encode(v)
        if control == 'ngram':
            ids = char_ngram_resample(ids, n=n_ngram, seed=seed, vocab_size=tok.vocab_size)
        out['splits'][k] = {'n_chars': len(ids), 'windows': make_windows(ids, seq_len)}
    return out
