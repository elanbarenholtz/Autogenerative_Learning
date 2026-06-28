"""
Known epsilon-machines (unifilar HMMs) with exactly computable Cmu / hmu / E.
Used to validate the empirical estimators and CSSR before measuring real generators.

A machine: dict state -> {symbol: (next_state, prob)}, probs per state sum to 1.
Unifilar: (state, symbol) -> unique next_state.
"""
import numpy as np


def states_of(m):
    return list(m.keys())


def symbols_of(m):
    s = set()
    for st in m.values():
        s.update(st.keys())
    return sorted(s)


def state_transition_matrix(m):
    sts = states_of(m)
    idx = {s: i for i, s in enumerate(sts)}
    M = np.zeros((len(sts), len(sts)))
    for s, trans in m.items():
        for sym, (s2, p) in trans.items():
            M[idx[s], idx[s2]] += p
    return M, sts, idx


def stationary_distribution(m):
    M, sts, idx = state_transition_matrix(m)
    vals, vecs = np.linalg.eig(M.T)
    i = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, i])
    pi = pi / pi.sum()
    pi = np.clip(pi, 0, None)
    pi = pi / pi.sum()
    return pi, sts, idx


def Cmu_analytic(m):
    pi, _, _ = stationary_distribution(m)
    p = pi[pi > 0]
    return float(-np.sum(p * np.log2(p)))


def hmu_analytic(m):
    pi, sts, idx = stationary_distribution(m)
    h = 0.0
    for s, trans in m.items():
        ps = np.array([p for (_, p) in trans.values()])
        ps = ps[ps > 0]
        Hs = -np.sum(ps * np.log2(ps))
        h += pi[idx[s]] * Hs
    return float(h)


def exact_block_entropy(m, L):
    """Exact H(L) (bits) by enumerating nonzero length-L words via a state-vector DP."""
    pi, sts, idx = stationary_distribution(m)
    syms = symbols_of(m)
    word_probs = []

    def rec(vec, depth):
        if depth == L:
            tot = vec.sum()
            if tot > 0:
                word_probs.append(tot)
            return
        for x in syms:
            newvec = np.zeros(len(sts))
            for s, trans in m.items():
                v = vec[idx[s]]
                if v > 0 and x in trans:
                    s2, p = trans[x]
                    newvec[idx[s2]] += v * p
            if newvec.sum() > 1e-15:
                rec(newvec, depth + 1)

    rec(pi.copy(), 0)
    wp = np.array(word_probs)
    wp = wp[wp > 0]
    return float(-np.sum(wp * np.log2(wp)))


def excess_entropy_analytic(m, Lbig=18):
    h = hmu_analytic(m)
    HL = exact_block_entropy(m, Lbig)
    return float(HL - h * Lbig)


def sample(m, n, seed=0):
    rng = np.random.default_rng(seed)
    pi, sts, idx = stationary_distribution(m)
    s = sts[rng.choice(len(sts), p=pi)]   # index-based to keep tuple states intact
    out = []
    for _ in range(n):
        trans = m[s]
        syms = list(trans.keys())
        ps = np.array([trans[x][1] for x in syms])
        x = rng.choice(syms, p=ps / ps.sum())
        out.append(int(x))
        s = trans[x][0]
    return out


# --- Known machines ---
def iid_fair():
    return {'A': {0: ('A', 0.5), 1: ('A', 0.5)}}

def period2():
    return {'A': {0: ('B', 1.0)}, 'B': {1: ('A', 1.0)}}

def golden_mean(p=0.5):
    # no '00': state A (last sym 1) can emit 0 or 1; state B (last sym 0) must emit 1
    return {'A': {1: ('A', p), 0: ('B', 1 - p)}, 'B': {1: ('A', 1.0)}}

def order2_markov():
    # states = last two bits; P(1|context) below
    p1 = {(0, 0): 0.7, (0, 1): 0.3, (1, 0): 0.6, (1, 1): 0.2}
    m = {}
    for ctx, pp in p1.items():
        a, b = ctx
        m[ctx] = {1: ((b, 1), pp), 0: ((b, 0), 1 - pp)}
    return m

KNOWN = {
    'iid_fair': iid_fair(),
    'period2': period2(),
    'golden_mean': golden_mean(0.5),
    'order2_markov': order2_markov(),
}
