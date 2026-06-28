"""
RoPE arm of the width-generalization probe. Same protocol as rv_width_gen.py
(Rule 30, train width 64, test {32,48,64,96,128}) but with a relative (RoPE) model.
Prediction: RoPE generalizes across widths where absolute PE collapsed to chance.
Outputs runs/width_gen/width_gen_rope.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import exp_ca as eca
from experiment_framework import set_all_seeds
from model_rope import RoPETransformer, count_parameters

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
OUT = 'runs/width_gen'
os.makedirs(OUT, exist_ok=True)

RULE, K, TRAIN_W = 30, 4, 64
TEST_WIDTHS = [32, 48, 64, 96, 128]
N_TRAIN, N_TEST, STEPS, EPOCHS = 200, 100, 20, 25
SEEDS = [42, 123, 7]

def tok(trajs):
    return eca.tokenize_windows(eca.create_windows(trajs, K))

def train_rope(train_tok, seed):
    set_all_seeds(seed)
    ds = eca.CADataset(train_tok)
    loader = DataLoader(ds, batch_size=32, shuffle=True,
                        collate_fn=lambda b: eca.collate_ca(b, eca.CATokenizer.PAD))
    model = RoPETransformer(vocab_size=eca.CATokenizer.vocab_size).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets, _ in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            opt.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, eca.CATokenizer.vocab_size),
                                   targets.reshape(-1), ignore_index=eca.CATokenizer.PAD)
            loss.backward(); opt.step()
    model.eval()
    return model

def main():
    t0 = time.time()
    print(f"device={DEVICE} model=RoPE params={count_parameters(RoPETransformer(vocab_size=6)):,}")
    train_tr = eca.generate_trajectories(RULE, N_TRAIN, TRAIN_W, STEPS, 0.5, eca.DATA_SEED)
    train_tok = tok(train_tr)
    test_sets = {W: tok(eca.generate_trajectories(RULE, N_TEST, W, STEPS, 0.5, base_seed=500000 + W))
                 for W in TEST_WIDTHS}

    per_seed = {}
    for rs in SEEDS:
        model = train_rope(train_tok, rs)
        row = {W: eca.evaluate_teacher_forced_ca(model, test_sets[W], K, W, DEVICE)['per_cell_accuracy']
               for W in TEST_WIDTHS}
        per_seed[rs] = row
        print(f"RS{rs}: " + "  ".join(f"W{W}={row[W]:.1f}%" for W in TEST_WIDTHS))

    agg = {W: [float(np.mean([per_seed[rs][W] for rs in SEEDS])),
               float(np.std([per_seed[rs][W] for rs in SEEDS]))] for W in TEST_WIDTHS}
    out = {'rule': RULE, 'train_width': TRAIN_W, 'test_widths': TEST_WIDTHS,
           'config': dict(k_rows=K, n_train=N_TRAIN, n_test=N_TEST, steps=STEPS,
                          epochs=EPOCHS, seeds=SEEDS, pos_encoding='rope', device=DEVICE),
           'per_seed': {str(k): v for k, v in per_seed.items()},
           'aggregate_cell_acc': {str(W): agg[W] for W in TEST_WIDTHS},
           'elapsed_sec': time.time() - t0}
    with open(os.path.join(OUT, 'width_gen_rope.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\n=== width-generalization (Rule 30, RoPE, train W=64, 3-seed mean) ===")
    for W in TEST_WIDTHS:
        tag = " (train)" if W == TRAIN_W else (" wider" if W > TRAIN_W else " narrower")
        print(f"  W={W:3d}: {agg[W][0]:6.2f}% ± {agg[W][1]:.2f}{tag}")
    print(f"elapsed {out['elapsed_sec']:.0f}s")

if __name__ == '__main__':
    main()
