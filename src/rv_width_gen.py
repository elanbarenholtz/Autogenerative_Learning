"""
Width-generalization probe for CA (Rule 30): does the learner generalize the
local rule across row widths it was never trained on?

Train at width 64 (fixed sinusoidal absolute positional encoding, the current model).
Test teacher-forced on widths {32,48,64,96,128}. Narrower widths use only positions
seen in training; wider widths push into positions never seen -> the real test of
whether 'where to look' was learned relative/delimiter-anchored or as an absolute
token offset. Oracle is 100% at every width by construction, so any drop is the
learner's positional generalization, not the task.
Outputs runs/width_gen/width_gen.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import exp_ca as eca

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
OUT = 'runs/width_gen'
os.makedirs(OUT, exist_ok=True)

RULE = 30
K = 4
TRAIN_W = 64
TEST_WIDTHS = [32, 48, 64, 96, 128]
N_TRAIN = 200
N_TEST = 100
STEPS = 20
EPOCHS = 25
SEEDS = [42, 123, 7]
# size PE table for the widest test sequence: 1 + (K+1)*(Wmax+1) + slack
MAX_W = max(TEST_WIDTHS)
MAX_SEQ = 1 + (K + 1) * (MAX_W + 1) + 60

def tok(trajs):
    return eca.tokenize_windows(eca.create_windows(trajs, K))

def main():
    t0 = time.time()
    print(f"device={DEVICE} train_width={TRAIN_W} test_widths={TEST_WIDTHS} max_seq={MAX_SEQ}")
    train_tr = eca.generate_trajectories(RULE, N_TRAIN, TRAIN_W, STEPS, 0.5, eca.DATA_SEED)
    train_tok = tok(train_tr)
    print(f"train: {len(train_tok)} windows, seq_len={len(train_tok[0])}")

    # pre-generate test sets per width (disjoint seeds, far from train)
    test_sets = {}
    for W in TEST_WIDTHS:
        trajs = eca.generate_trajectories(RULE, N_TEST, W, STEPS, 0.5, base_seed=500000 + W)
        test_sets[W] = tok(trajs)

    per_seed = {}
    for rs in SEEDS:
        model, _ = eca.train_lm(train_tok, rs, os.path.join(OUT, f'RS{rs}'),
                                vocab_size=eca.CATokenizer.vocab_size, epochs=EPOCHS,
                                batch_size=32, lr=0.001, max_seq_len=MAX_SEQ,
                                device=DEVICE, verbose=False)
        row = {}
        for W in TEST_WIDTHS:
            r = eca.evaluate_teacher_forced_ca(model, test_sets[W], K, W, DEVICE)
            row[W] = r['per_cell_accuracy']
        per_seed[rs] = row
        print(f"RS{rs}: " + "  ".join(f"W{W}={row[W]:.1f}%" for W in TEST_WIDTHS))

    agg = {W: [float(np.mean([per_seed[rs][W] for rs in SEEDS])),
               float(np.std([per_seed[rs][W] for rs in SEEDS]))] for W in TEST_WIDTHS}
    out = {'rule': RULE, 'train_width': TRAIN_W, 'test_widths': TEST_WIDTHS,
           'config': dict(k_rows=K, n_train=N_TRAIN, n_test=N_TEST, steps=STEPS,
                          epochs=EPOCHS, seeds=SEEDS, pos_encoding='sinusoidal_absolute',
                          device=DEVICE, max_seq=MAX_SEQ),
           'per_seed': {str(k): v for k, v in per_seed.items()},
           'aggregate_cell_acc': {str(W): agg[W] for W in TEST_WIDTHS},
           'elapsed_sec': time.time() - t0}
    with open(os.path.join(OUT, 'width_gen.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\n=== width-generalization (Rule 30, train W=64, 3-seed mean cell acc) ===")
    for W in TEST_WIDTHS:
        tag = " (train)" if W == TRAIN_W else (" wider->new positions" if W > TRAIN_W else " narrower")
        print(f"  W={W:3d}: {agg[W][0]:6.2f}% ± {agg[W][1]:.2f}{tag}")
    print(f"elapsed {out['elapsed_sec']:.0f}s -> {OUT}/width_gen.json")

if __name__ == '__main__':
    main()
