"""
Validity gate: empirical estimators vs exact analytic ε-machine quantities.
If hmu_est ~ hmu_analytic and E_est ~ E_analytic on these known machines,
the estimators are trustworthy for the real generators.
(Cmu is validated separately with CSSR in cssr validation.)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from complexity import machines as M
from complexity import estimators as E

N_SEQ = 40
SEQ_LEN = 20000
LMAX = 12

def main():
    print(f"samples: {N_SEQ} x {SEQ_LEN} | Lmax={LMAX}\n")
    hdr = f"{'machine':<16}{'Cmu':>8}{'hmu_an':>9}{'hmu_est':>9}{'E_an':>8}{'E_est':>8}"
    print(hdr); print('-' * len(hdr))
    for name, m in M.KNOWN.items():
        cmu = M.Cmu_analytic(m)
        hmu_an = M.hmu_analytic(m)
        E_an = M.excess_entropy_analytic(m, Lbig=18)
        seqs = [M.sample(m, SEQ_LEN, seed=s) for s in range(N_SEQ)]
        summ = E.summarize(seqs, LMAX, label=name)
        print(f"{name:<16}{cmu:>8.3f}{hmu_an:>9.3f}{summ['entropy_rate_hmu']:>9.3f}"
              f"{E_an:>8.3f}{summ['excess_entropy_E']:>8.3f}")
    print("\nrecoverability-at-width r(w)=H(next|last w), should fall to hmu:")
    for name, m in M.KNOWN.items():
        seqs = [M.sample(m, SEQ_LEN, seed=s) for s in range(N_SEQ)]
        summ = E.summarize(seqs, LMAX, label=name)
        r = summ['recoverability_at_width']
        print(f"  {name:<16} r(0..6) = " + " ".join(f"{x:.3f}" for x in r[:7]))

if __name__ == '__main__':
    main()
