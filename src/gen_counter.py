"""
Group B2 — Counter languages a^n b^n and a^n b^n c^n.
Counter (unbounded-memory) languages: validity requires remembering the count n,
which grows with the string. Open-span, like Dyck but a pure counter.

Vocab: PAD0 BOS1 EOS2 A3 B4 C5.  Oracle gives the legal next-token set.
"""
import random
from typing import List, Set, Tuple, Optional

PAD, BOS, EOS, A, B, C = 0, 1, 2, 3, 4, 5
ID2STR = {0: 'PAD', 1: 'BOS', 2: 'EOS', 3: 'a', 4: 'b', 5: 'c'}


def vocab_size(lang='anbn'):
    return 6


def generate_string(n: int, lang='anbn') -> List[int]:
    s = [BOS] + [A] * n + [B] * n
    if lang == 'anbncn':
        s += [C] * n
    return s + [EOS]


def generate_split(n_seqs, n_min, n_max, lang='anbn', seed=0) -> List[List[int]]:
    rng = random.Random(seed)
    return [generate_string(rng.randint(n_min, n_max), lang) for _ in range(n_seqs)]


def is_valid(tokens: List[int], lang='anbn') -> Tuple[bool, Optional[int], Optional[str]]:
    """Validate a complete a^n b^n (c^n) string."""
    if not tokens or tokens[0] != BOS or tokens[-1] != EOS:
        return False, 0, 'no_bos_eos'
    body = tokens[1:-1]
    expect = [A, B, C] if lang == 'anbncn' else [A, B]
    counts, phase = [], 0
    run = 0
    prev = None
    seq = []
    for i, t in enumerate(body):
        if t not in expect:
            return False, i, 'bad_symbol'
        if prev is None or t == prev:
            run += 1
        else:
            seq.append((prev, run)); run = 1
        prev = t
    if prev is not None:
        seq.append((prev, run))
    # must be exactly expect[] in order, each run equal
    if [sym for sym, _ in seq] != expect:
        return False, None, 'wrong_phase_order'
    ns = [r for _, r in seq]
    if len(set(ns)) != 1 or ns[0] == 0:
        return False, None, 'unequal_counts'
    return True, None, None


def oracle_next_legal(prefix: List[int], lang='anbn', n_cap=10000) -> Set[int]:
    """Legal next tokens given a valid prefix, to stay completable."""
    expect = [A, B, C] if lang == 'anbncn' else [A, B]
    # count run lengths per phase seen so far (after BOS)
    body = prefix[1:] if prefix and prefix[0] == BOS else prefix
    counts = {A: 0, B: 0, C: 0}
    for t in body:
        if t in counts:
            counts[t] += 1
    na = counts[A]
    if na == 0:
        return {A}                       # must start with a
    if lang == 'anbn':
        nb = counts[B]
        if nb == 0:
            return {A, B}                # more a's, or switch to b
        if nb < na:
            return {B}
        return {EOS}
    else:  # anbncn
        nb, nc = counts[B], counts[C]
        if nb == 0:
            return {A, B}
        if nb < na:
            return {B}
        # nb == na, now c-phase
        if nc == 0:
            return {C}
        if nc < na:
            return {C}
        return {EOS}


def stream(n_seqs=4000, n_min=1, n_max=20, lang='anbn', seed=0):
    """Serialized stream for complexity measurement (per-sequence)."""
    seqs = generate_split(n_seqs, n_min, n_max, lang, seed)
    used = sorted({t for s in seqs for t in s})
    remap = {t: i for i, t in enumerate(used)}
    return [[remap[t] for t in s] for s in seqs], len(used)
