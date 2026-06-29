"""
Group B4 — absolute-position / indexing tasks (isolate position-binding).
Tasks: 'kth' (emit the k-th content token), 'middle' (emit middle token),
'last2' (emit last two). Variable content length L tests position binding;
OOD = longer L (and larger k) than trained. Oracle is deterministic (100%).

Vocab: PAD0 BOS1 EOS2 SEP3 | content[4..4+C-1] | index[4+C .. 4+C+P-1]
"""
import random
from typing import List

PAD, BOS, EOS, SEP = 0, 1, 2, 3
C_CONTENT = 6          # number of content symbols
P_MAX = 80             # max indexable position
CONTENT0 = 4
IDX0 = CONTENT0 + C_CONTENT


def vocab_size():
    return IDX0 + P_MAX


def _content(L, rng):
    return [CONTENT0 + rng.randrange(C_CONTENT) for _ in range(L)]


def generate_example(task, L, rng) -> dict:
    c = _content(L, rng)
    if task == 'kth':
        k = rng.randrange(L)
        seq = [BOS] + c + [SEP, IDX0 + k, c[k], EOS]
        ans_positions = [len(c) + 3]              # index of c[k] in seq
        answer = [c[k]]
    elif task == 'middle':
        k = L // 2
        seq = [BOS] + c + [SEP, c[k], EOS]
        ans_positions = [len(c) + 2]
        answer = [c[k]]
    elif task == 'last2':
        seq = [BOS] + c + [SEP, c[-2], c[-1], EOS]
        ans_positions = [len(c) + 2, len(c) + 3]
        answer = [c[-2], c[-1]]
    else:
        raise ValueError(task)
    return {'tokens': seq, 'answer_positions': ans_positions, 'answer': answer}


def generate_split(task, n_seqs, L_min, L_max, seed=0) -> List[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n_seqs):
        L = rng.randint(L_min, L_max)
        if task == 'middle' and L % 2 == 0:
            L += 1                                 # keep a unique middle
        if L > P_MAX:
            L = P_MAX
        out.append(generate_example(task, L, rng))
    return out


def oracle_answer(ex: dict) -> List[int]:
    """Deterministic correct answer tokens (sanity: equals ex['answer'])."""
    return ex['answer']


def stream(task='kth', n_seqs=4000, L_min=4, L_max=24, seed=0):
    exs = generate_split(task, n_seqs, L_min, L_max, seed)
    seqs = [e['tokens'] for e in exs]
    used = sorted({t for s in seqs for t in s})
    remap = {t: i for i, t in enumerate(used)}
    return [[remap[t] for t in s] for s in seqs], len(used)
