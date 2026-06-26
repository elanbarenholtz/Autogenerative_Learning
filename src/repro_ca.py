"""
Scaled reproduction of the elementary-CA generalization result (Rule 30 / 90).
Teacher-forced per-cell accuracy on ID + OOD splits; free-run skipped for this gate.

Prior saved numbers (full scale, 2000 seeds, 50 epochs):
  Rule30: 100% on id_test, longer_ood, density 0.2/0.8  (free-run also 100%)
  Rule90: ~99.4% id, ~99.7% longer_ood, ~99.3-99.4% density
Target: transformer ~99-100% on all splits incl. OOD; >> n-gram.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import exp_ca as eca
from exp5_baselines import NgramBaseline

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
OUT = 'runs/repro_gate'
os.makedirs(OUT, exist_ok=True)

WIDTH = 40; K = 4
N_TRAIN = 200; N_TEST = 100
STEPS = 20; STEPS_OOD = 36
EPOCHS = 30
SEEDS = [42, 123, 7]
RULES = [30, 90]

def windows_tokens(trajs):
    return eca.tokenize_windows(eca.create_windows(trajs, K))

def main():
    t0 = time.time()
    results = {}
    for rule in RULES:
        print(f"\n=== Rule {rule} (device={DEVICE}) ===")
        train_tr = eca.generate_trajectories(rule, N_TRAIN, WIDTH, STEPS, 0.5, eca.DATA_SEED)
        tbase = eca.DATA_SEED + N_TRAIN
        splits_tr = {
            'id_test': eca.generate_trajectories(rule, N_TEST, WIDTH, STEPS, 0.5, tbase),
            'longer_ood': eca.generate_trajectories(rule, N_TEST, WIDTH, STEPS_OOD, 0.5, tbase),
            'density0.2': eca.generate_trajectories(rule, N_TEST, WIDTH, STEPS, 0.2, tbase + N_TEST),
            'density0.8': eca.generate_trajectories(rule, N_TEST, WIDTH, STEPS, 0.8, tbase + 2 * N_TEST),
        }
        assert eca.DATA_SEED + N_TRAIN <= tbase, "seed overlap"
        train_tok = windows_tokens(train_tr)
        split_tok = {k: windows_tokens(v) for k, v in splits_tr.items()}

        # n-gram baseline
        ng = NgramBaseline(n=5)
        ng.train(eca.target_cell_examples(train_tok, K, WIDTH, 10))
        ng_res = {k: eca.evaluate_baseline_ca(ng, split_tok[k], K, WIDTH, 10)['per_cell_accuracy']
                  for k in split_tok}
        print("  ngram5:", {k: round(v, 2) for k, v in ng_res.items()})

        per_seed = {}
        for rs in SEEDS:
            rundir = os.path.join(OUT, f'rule{rule}_RS{rs}')
            model, _ = eca.train_lm(train_tok, rs, rundir, vocab_size=eca.CATokenizer.vocab_size,
                                    epochs=EPOCHS, batch_size=32, lr=0.001,
                                    max_seq_len=len(train_tok[0]) + 50, device=DEVICE, verbose=False)
            sd = {k: eca.evaluate_teacher_forced_ca(model, split_tok[k], K, WIDTH, DEVICE)['per_cell_accuracy']
                  for k in split_tok}
            per_seed[rs] = sd
            print(f"  RS{rs}:", {k: round(v, 2) for k, v in sd.items()})

        agg = {k: [float(np.mean([per_seed[rs][k] for rs in SEEDS])),
                   float(np.std([per_seed[rs][k] for rs in SEEDS]))] for k in split_tok}
        results[f'Rule{rule}'] = {'ngram5': ng_res, 'transformer': agg,
                                  'per_seed': {str(r): v for r, v in per_seed.items()}}
        print(f"  Rule{rule} transformer agg:", {k: round(v[0], 2) for k, v in agg.items()})

    out = {'config': dict(width=WIDTH, k_rows=K, n_train_seeds=N_TRAIN, n_test=N_TEST,
                          steps=STEPS, steps_ood=STEPS_OOD, epochs=EPOCHS, seeds=SEEDS,
                          device=DEVICE, scaled=True),
           'results': results, 'elapsed_sec': time.time() - t0}
    with open(os.path.join(OUT, 'ca_repro.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nelapsed {out['elapsed_sec']:.0f}s -> {os.path.join(OUT, 'ca_repro.json')}")

if __name__ == '__main__':
    main()
