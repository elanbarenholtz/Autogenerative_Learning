"""
CSSR-style causal-state reconstruction for statistical complexity Cmu, from raw
sequences only (no model). Histories of length L are grouped into causal states by
a chi-square test of homogeneity on their next-symbol counts (sample-size aware:
a real difference separates given enough data; noise within a true state merges).
Cmu = H(stationary state distribution). We sweep L and significance alpha and pick
Cmu at the shortest L where the causal-state count stabilizes (sensitivity reported).
Labeled 'CSSR-style' for honesty.
"""
import numpy as np
from collections import defaultdict
from scipy.stats import chi2_contingency


def _hist_next_counts(sequences, L, n_sym):
    counts = defaultdict(lambda: np.zeros(n_sym))
    for seq in sequences:
        for i in range(len(seq) - L):
            counts[tuple(seq[i:i + L])][seq[i + L]] += 1
    return counts


def _same_state(c1, c2, alpha):
    """True if next-symbol counts c1,c2 are NOT significantly different (merge)."""
    table = np.vstack([c1, c2])
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    if table.shape[1] < 2:
        return True
    try:
        _, p, _, _ = chi2_contingency(table, correction=False)
    except ValueError:
        return True
    return p > alpha


def _cluster(counts, alpha, min_count):
    reps, weights = [], []      # reps: representative count vectors (highest-count history)
    items = sorted(counts.items(), key=lambda kv: -kv[1].sum())
    for h, c in items:
        if c.sum() < min_count:
            continue
        placed = False
        for k in range(len(reps)):
            if _same_state(reps[k], c, alpha):
                weights[k] += c.sum()
                placed = True
                break
        if not placed:
            reps.append(c.copy())
            weights.append(c.sum())
    return np.array(weights) if weights else np.array([0.0])


def cmu_at_L(sequences, L, n_sym, alpha=1e-3, min_count=30):
    w = _cluster(_hist_next_counts(sequences, L, n_sym), alpha, min_count)
    if w.sum() == 0:
        return 0.0, 0
    pi = w / w.sum()
    pi = pi[pi > 0]
    return float(-np.sum(pi * np.log2(pi))), len(w)


def cssr_cmu(sequences, n_sym, Lmax=8, alphas=(1e-2, 1e-3, 1e-4), min_count=30):
    grid = {}
    for alpha in alphas:
        for L in range(1, Lmax + 1):
            Cmu, nst = cmu_at_L(sequences, L, n_sym, alpha, min_count)
            grid[f'L{L}_a{alpha:.0e}'] = {'Cmu': round(Cmu, 4), 'n_states': nst}
    mid = alphas[len(alphas) // 2]
    nst = [grid[f'L{L}_a{mid:.0e}']['n_states'] for L in range(1, Lmax + 1)]
    cmus = [grid[f'L{L}_a{mid:.0e}']['Cmu'] for L in range(1, Lmax + 1)]
    sel, stable = len(nst) - 1, False
    for i in range(len(nst) - 1):
        if nst[i] == nst[i + 1] and nst[i] > 0:
            sel, stable = i, True
            break
    return {'plateau_Cmu': round(cmus[sel], 4), 'selected_L': sel + 1,
            'n_states_at_selected': nst[sel], 'state_count_stabilized': stable,
            'alpha_used': mid, 'grid': grid}
