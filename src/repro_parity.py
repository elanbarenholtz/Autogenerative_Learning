"""
Scaled reproduction of the Parity-CA held-out-neighborhood generalization result.
(Validity gate -- teacher-forced A/B breakdown; free-run rollout skipped for speed.)

Prior saved numbers (full scale, n_train=5000, 50 epochs):
  ood_b_inclusive: overall ~91%, A ~91.7%, B ~90.4%, A-B gap ~1.3
  n-gram(5) on B:  ~49% (chance)
Target: transformer B-accuracy >> n-gram B-accuracy  ==> rule learned, generalizes to unseen neighborhoods.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import exp_parity_ca as epc
from exp5_baselines import NgramBaseline

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
OUT = 'runs/repro_gate'
os.makedirs(OUT, exist_ok=True)

# Scaled config (uniform downscale of prior 5000/50)
WIDTH = 64; K = 1; A_FRAC = 0.6
N_TRAIN = 5000; N_TEST = 300; EPOCHS = 50
SEEDS = [42, 123, 7]

def main():
    t0 = time.time()
    allowed, blocked = epc.select_allowed_set(epc.N_PATTERNS, A_FRAC, epc.DATA_SEED)
    graph = epc.build_constraint_graph(allowed)
    print(f"|A|={len(allowed)} |B|={len(blocked)}  device={DEVICE}")

    train_w = epc.generate_training_data(N_TRAIN, WIDTH, allowed, graph, epc.DATA_SEED)
    base = epc.DATA_SEED + 100000
    id_w = epc.generate_id_test_data(N_TEST, WIDTH, allowed, graph, base)
    ood_w = epc.generate_test_data(N_TEST, WIDTH, base + 1)
    splits = {'id_test': id_w, 'ood_b_inclusive': ood_w}
    train_tok = epc.tokenize_windows(train_w, K)
    split_tok = {k: epc.tokenize_windows(v, K) for k, v in splits.items()}

    # disjointness + oracle gate
    for nm, w in [('train', train_w)] + list(splits.items()):
        orc = epc.evaluate_oracle(w, WIDTH, allowed)
        assert orc['overall_cell_acc'] == 100.0, (nm, orc)
    print("oracle 100% all splits: PASS")

    # n-gram baseline (matched), A/B breakdown
    ng_train = epc.target_cell_examples(train_tok, K, WIDTH, 10)
    ng = NgramBaseline(n=5); ng.train(ng_train)
    ng_b = {}
    for sname in splits:
        r = epc.evaluate_baseline_parity(ng, split_tok[sname], splits[sname], K, WIDTH, allowed, 10)
        ng_b[sname] = r
        print(f"ngram5 {sname}: overall={r['overall_cell_acc']:.2f} A={r['a_cell_acc']:.2f} B={r['b_cell_acc']:.2f}")

    per_seed = {}
    for rs in SEEDS:
        rundir = os.path.join(OUT, f'parity_RS{rs}')
        model, _ = epc.train_lm(train_tok, rs, rundir, vocab_size=epc.CATokenizer.vocab_size,
                                epochs=EPOCHS, batch_size=32, lr=0.001, max_seq_len=len(train_tok[0]) + 50,
                                device=DEVICE, verbose=False)
        sd = {}
        for sname in splits:
            tf = epc.evaluate_teacher_forced_parity(model, split_tok[sname], splits[sname],
                                                    WIDTH, allowed, K, epc.RADIUS, DEVICE)
            sd[sname] = tf
            print(f"RS{rs} {sname}: overall={tf['overall_cell_acc']:.2f} "
                  f"A={tf['a_cell_acc']:.2f} B={tf['b_cell_acc']:.2f} gap={tf['a_cell_acc']-tf['b_cell_acc']:.2f}")
        per_seed[rs] = sd

    # aggregate
    agg = {}
    for sname in splits:
        for metric in ['overall_cell_acc', 'a_cell_acc', 'b_cell_acc']:
            vals = [per_seed[rs][sname][metric] for rs in SEEDS]
            agg[f'{sname}.{metric}'] = [float(np.mean(vals)), float(np.std(vals))]
    out = {'config': dict(width=WIDTH, n_train=N_TRAIN, n_test=N_TEST, epochs=EPOCHS,
                          seeds=SEEDS, a_fraction=A_FRAC, device=DEVICE, scaled=True),
           'ngram5': ng_b, 'transformer_aggregate': agg,
           'transformer_per_seed': {str(k): v for k, v in per_seed.items()},
           'elapsed_sec': time.time() - t0}
    with open(os.path.join(OUT, 'parity_repro.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\n=== SUMMARY ===")
    print(f"OOD B-incl transformer: overall={agg['ood_b_inclusive.overall_cell_acc'][0]:.2f}%, "
          f"A={agg['ood_b_inclusive.a_cell_acc'][0]:.2f}%, B={agg['ood_b_inclusive.b_cell_acc'][0]:.2f}%")
    print(f"OOD B-incl n-gram(5):   B={ng_b['ood_b_inclusive']['b_cell_acc']:.2f}%")
    print(f"elapsed {out['elapsed_sec']:.0f}s -> {os.path.join(OUT,'parity_repro.json')}")

if __name__ == '__main__':
    main()
