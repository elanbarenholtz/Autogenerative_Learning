"""
Serialized streams for each generator, as the learner sees them.
Each function returns (list_of_sequences, alphabet_size). Block estimators pool
blocks WITHIN sequences (no crossing), so per-sequence boundaries need no token.
Small alphabets / modest widths so plug-in block entropy is feasible.
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp_ca as eca
import exp_dyck as edy

CA_SEP = 2   # binary CA cells use tokens 0,1; row separator = 2


def ca_stream(rule, width=8, n_traj=3000, n_rows=40, density=0.5, seed0=0):
    """Flattened row-by-row CA token stream (cells + row separators)."""
    seqs = []
    for i in range(n_traj):
        rng = random.Random(seed0 + i)
        init = [1 if rng.random() < density else 0 for _ in range(width)]
        rows = eca.generate_trajectory(rule, init, n_rows)
        s = []
        for r in rows:
            s.extend(r)
            s.append(CA_SEP)
        seqs.append(s)
    return seqs, 3


def dyck_stream(bracket_types=1, n_seqs=4000, min_pairs=1, max_pairs=20,
                max_depth=6, seed=0):
    """Dyck bracket strings (incl. BOS/EOS as string delimiters), compacted to 0..A-1."""
    tok = edy.DyckTokenizer(bracket_types)
    raw = edy.generate_split(tok, n_seqs, min_pairs, max_pairs, max_depth, seed)
    used = sorted({t for s in raw for t in s})
    remap = {t: i for i, t in enumerate(used)}
    seqs = [[remap[t] for t in s] for s in raw]
    return seqs, len(used)


def fib_mod_stream(p=31, n_seqs=4000, length=25, seed0=0):
    """Fibonacci-mod-p residue stream: a_{n+1} = (a_n + a_{n-1}) mod p, random seeds."""
    seqs = []
    for i in range(n_seqs):
        rng = random.Random(seed0 + i)
        a, b = rng.randrange(p), rng.randrange(p)
        s = [a, b]
        for _ in range(length - 2):
            a, b = b, (a + b) % p
            s.append(b)
        seqs.append(s)
    return seqs, p


def markov_stream(order, n_seqs=200, length=5000, seed0=0):
    """Random stationary order-k binary Markov source (control points on the axis)."""
    import numpy as np
    rng = np.random.default_rng(seed0)
    ctxs = [tuple(((c >> j) & 1) for j in range(order)) for c in range(2 ** order)] if order else [()]
    p1 = {c: float(rng.uniform(0.1, 0.9)) for c in ctxs}
    seqs = []
    for i in range(n_seqs):
        r = random.Random(seed0 + i)
        hist = tuple(r.randint(0, 1) for _ in range(order))
        s = list(hist)
        for _ in range(length - order):
            nb = 1 if r.random() < p1[hist] else 0
            s.append(nb)
            hist = (hist + (nb,))[1:] if order else ()
        seqs.append(s)
    return seqs, 2
