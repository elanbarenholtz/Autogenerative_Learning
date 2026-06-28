"""
Information-theoretic estimators for symbol streams (pure data, no model).

All quantities derived from empirical block entropies H(L):
  - H(L): Shannon entropy (bits) of length-L blocks, Miller-Madow bias-corrected.
  - recoverability-at-width r(w) = H(next | last w) = H(w+1) - H(w).
  - entropy rate hmu = lim_w r(w).
  - excess entropy E = lim_L [H(L) - hmu*L]  (intercept of the large-L linear fit).

These are the estimators used for the recoverability axis. Cmu (statistical
complexity) is computed separately (analytic + CSSR), not here.
"""
import numpy as np
from collections import Counter


def block_entropy(sequences, L, alphabet_size=None):
    """Miller-Madow-corrected entropy (bits) of length-L blocks pooled over sequences."""
    counts = Counter()
    total = 0
    for seq in sequences:
        n = len(seq)
        if n < L:
            continue
        for i in range(n - L + 1):
            counts[tuple(seq[i:i + L])] += 1
            total += 1
    if total == 0:
        return 0.0, 0, 0
    probs = np.array(list(counts.values()), dtype=float) / total
    H = -np.sum(probs * np.log2(probs))
    K = len(counts)                      # observed distinct blocks
    H_mm = H + (K - 1) / (2 * total * np.log(2))   # Miller-Madow correction (bits)
    return float(H_mm), K, total


def block_entropy_curve(sequences, Lmax, alphabet_size=None):
    """H(L) for L=0..Lmax. H(0)=0 by convention."""
    H = [0.0]
    meta = [{'L': 0, 'K': 1, 'N': sum(len(s) for s in sequences)}]
    for L in range(1, Lmax + 1):
        h, K, N = block_entropy(sequences, L, alphabet_size)
        H.append(h)
        meta.append({'L': L, 'K': K, 'N': N})
    return np.array(H), meta


def recoverability_at_width(H):
    """r(w) = H(next | last w) = H[w+1]-H[w], for w = 0..len(H)-2."""
    H = np.asarray(H)
    return H[1:] - H[:-1]      # index w -> conditional entropy given w previous symbols


def entropy_rate(H, tail=2):
    """hmu = floor of the recoverability curve = min of the last `tail` finite
    differences H[L]-H[L-1]. (Robust when the curve drops sharply near the end,
    e.g. deterministic recurrences where r(w) hits 0 at small w.)"""
    diffs = np.diff(H)
    return float(np.min(diffs[-tail:]))


def excess_entropy(H, fit_from=None):
    """
    E = intercept of the large-L linear fit of H(L) ~ hmu*L + E.
    Returns (E, hmu_fit). Fit over L >= fit_from (default: second half of curve).
    """
    H = np.asarray(H)
    Lmax = len(H) - 1
    if fit_from is None:
        fit_from = max(1, Lmax // 2)
    Ls = np.arange(fit_from, Lmax + 1)
    ys = H[fit_from:Lmax + 1]
    A = np.vstack([Ls, np.ones_like(Ls)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(intercept), float(slope)


def summarize(sequences, Lmax, alphabet_size=None, label=""):
    """Full estimator summary for one process's sampled stream(s)."""
    H, meta = block_entropy_curve(sequences, Lmax, alphabet_size)
    r = recoverability_at_width(H)
    hmu = entropy_rate(H)
    E, hmu_fit = excess_entropy(H)
    # data-sufficiency flag: at largest L, want N >> K
    last = meta[-1]
    undersampled = last['N'] < 20 * last['K']
    return {
        'label': label,
        'Lmax': Lmax,
        'H_of_L': [round(float(x), 5) for x in H],
        'recoverability_at_width': [round(float(x), 5) for x in r],
        'entropy_rate_hmu': round(hmu, 5),
        'excess_entropy_E': round(E, 5),
        'hmu_from_fit': round(hmu_fit, 5),
        'block_counts_K': [m['K'] for m in meta],
        'n_samples_at_Lmax': last['N'],
        'undersampled_at_Lmax': bool(undersampled),
    }
