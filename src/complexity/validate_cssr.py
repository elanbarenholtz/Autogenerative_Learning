"""Validate CSSR-style Cmu against exact analytic Cmu on known machines."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from complexity import machines as M
from complexity import cssr

N_SEQ, SEQ_LEN = 40, 20000

def main():
    print(f"samples {N_SEQ}x{SEQ_LEN}\n")
    hdr = f"{'machine':<16}{'Cmu_analytic':>14}{'Cmu_CSSR':>11}{'n_states':>10}{'delta':>7}"
    print(hdr); print('-' * len(hdr))
    for name, m in M.KNOWN.items():
        n_sym = len(M.symbols_of(m))
        cmu_an = M.Cmu_analytic(m)
        seqs = [M.sample(m, SEQ_LEN, seed=s) for s in range(N_SEQ)]
        res = cssr.cssr_cmu(seqs, n_sym, Lmax=6)
        print(f"{name:<16}{cmu_an:>14.3f}{res['plateau_Cmu']:>11.3f}"
              f"{res['n_states_at_selected']:>10}{res['alpha_used']:>7.0e}"
              f"   (L*={res['selected_L']}, stable={res['state_count_stabilized']})")

if __name__ == '__main__':
    main()
