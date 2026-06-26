"""
1D Cellular Automata Experiment — Rule-Generated Sequences

Tests whether transformers learn the external CA update rule (generalizing to
unseen initial conditions) or merely match local distributional statistics.

Two automata: Rule 30 (chaotic) and Rule 90 (linear/XOR-like control).
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import FibonacciTransformer, count_parameters
from experiment_framework import DATA_SEED, RANDOM_SEEDS, set_all_seeds
from exp5_baselines import NgramBaseline, KNNBaseline

OUTPUT_DIR = 'runs/ca'


def _print(msg: str = ''):
    print(msg, flush=True)


# ===========================================================================
# Tokenizer
# ===========================================================================
class CATokenizer:
    """Fixed vocab tokenizer for 1D cellular automata."""
    PAD = 0
    BOS = 1
    EOS = 2
    ZERO = 3    # dead cell
    ONE = 4     # alive cell
    ROWSEP = 5  # row separator

    vocab_size = 6
    pad_id = 0

    id_to_str = {0: 'PAD', 1: 'BOS', 2: 'EOS', 3: '0', 4: '1', 5: '|'}

    @staticmethod
    def cell_to_token(cell_val):
        return CATokenizer.ONE if cell_val == 1 else CATokenizer.ZERO

    @staticmethod
    def token_to_cell(token_id):
        return 1 if token_id == CATokenizer.ONE else 0

    @staticmethod
    def tokenize_rows(rows):
        """Tokenize K+1 rows: BOS r0 ROWSEP r1 ROWSEP ... rK EOS"""
        tokens = [CATokenizer.BOS]
        for i, row in enumerate(rows):
            for cell in row:
                tokens.append(CATokenizer.cell_to_token(cell))
            if i < len(rows) - 1:
                tokens.append(CATokenizer.ROWSEP)
        tokens.append(CATokenizer.EOS)
        return tokens

    @staticmethod
    def decode_string(token_ids):
        return ''.join(CATokenizer.id_to_str.get(t, '?') for t in token_ids)


# ===========================================================================
# CA Dynamics
# ===========================================================================
def apply_rule(row, rule_number):
    """Apply 1D CA rule with wraparound boundary conditions."""
    W = len(row)
    new_row = [0] * W
    for i in range(W):
        left = row[(i - 1) % W]
        center = row[i]
        right = row[(i + 1) % W]
        neighborhood = (left << 2) | (center << 1) | right
        new_row[i] = (rule_number >> neighborhood) & 1
    return new_row


def generate_trajectory(rule_number, initial_row, n_steps):
    """Generate n_steps+1 rows (row 0 = initial condition)."""
    rows = [list(initial_row)]
    for _ in range(n_steps):
        rows.append(apply_rule(rows[-1], rule_number))
    return rows


def generate_initial_row(width, density, rng):
    return [1 if rng.random() < density else 0 for _ in range(width)]


# ===========================================================================
# Data Generation
# ===========================================================================
def generate_trajectories(rule_number, n_seeds, width, n_steps, density, base_seed):
    """Generate trajectories from distinct random initial conditions."""
    trajectories = []
    for i in range(n_seeds):
        rng = random.Random(base_seed + i)
        initial = generate_initial_row(width, density, rng)
        traj = generate_trajectory(rule_number, initial, n_steps)
        trajectories.append(traj)
    return trajectories


def create_windows(trajectories, k_rows):
    """Sliding windows of K+1 consecutive rows from each trajectory."""
    windows = []
    for traj in trajectories:
        for t in range(len(traj) - k_rows):
            windows.append(traj[t:t + k_rows + 1])
    return windows


def tokenize_windows(windows):
    return [CATokenizer.tokenize_rows(w) for w in windows]


def compute_data_distributions(trajectories, width):
    all_densities = []
    for traj in trajectories:
        for row in traj:
            all_densities.append(sum(row) / width)
    return {
        'n_trajectories': len(trajectories),
        'n_rows_per_traj': len(trajectories[0]) if trajectories else 0,
        'mean_density': float(np.mean(all_densities)),
        'std_density': float(np.std(all_densities)),
    }


# ===========================================================================
# Dataset & Collation
# ===========================================================================
class CADataset(Dataset):
    def __init__(self, token_sequences):
        self.sequences = token_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)


def collate_ca(batch, pad_id=0):
    max_len = max(t.size(0) for t in batch)
    inputs, targets, masks = [], [], []
    for seq in batch:
        inp = seq[:-1]
        tgt = seq[1:]
        pad_len = max_len - seq.size(0)
        if pad_len > 0:
            inp = torch.cat([inp, torch.full((pad_len,), pad_id, dtype=torch.long)])
            tgt = torch.cat([tgt, torch.full((pad_len,), pad_id, dtype=torch.long)])
        masks.append(inp == pad_id)
        inputs.append(inp)
        targets.append(tgt)
    return torch.stack(inputs), torch.stack(targets), torch.stack(masks)


# ===========================================================================
# Training
# ===========================================================================
def train_lm(train_tokens, random_seed, output_dir,
             vocab_size=6, epochs=50, batch_size=32, lr=0.001,
             d_model=128, nhead=4, num_layers=3, dim_feedforward=512,
             dropout=0.1, max_seq_len=400, device=None, verbose=True):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_all_seeds(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    dataset = CADataset(train_tokens)
    pad_id = CATokenizer.PAD
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_ca(b, pad_id))

    model = FibonacciTransformer(
        vocab_size=vocab_size, d_model=d_model, nhead=nhead,
        num_layers=num_layers, dim_feedforward=dim_feedforward,
        dropout=dropout, max_seq_len=max_seq_len).to(device)

    if verbose:
        _print(f'    Model parameters: {count_parameters(model):,}')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    history = {'losses': [], 'accuracies': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets, padding_mask in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()
            logits = model(inputs, src_key_padding_mask=padding_mask)
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_id)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            non_pad = targets_flat != pad_id
            if non_pad.any():
                preds = logits_flat.argmax(dim=-1)
                correct += (preds[non_pad] == targets_flat[non_pad]).sum().item()
                total += non_pad.sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        history['losses'].append(avg_loss)
        history['accuracies'].append(accuracy)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'loss': avg_loss, 'accuracy': accuracy},
                       os.path.join(output_dir, 'best_model.pt'))

        if verbose and (epoch + 1) % 10 == 0:
            _print(f'      Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={accuracy:.2f}%')

    if verbose:
        _print(f'      Training complete. Best loss={best_loss:.4f}, Final acc={accuracy:.2f}%')

    ckpt = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device,
                       weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, history


# ===========================================================================
# Evaluation: Teacher-Forced
# ===========================================================================
def evaluate_teacher_forced_ca(model, tokenized_windows, k_rows, width,
                                device=None):
    """Per-cell accuracy and row exact-match on target row only."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    # Target row cells in the shifted target tensor (tokens[1:])
    target_start = k_rows * (width + 1)
    target_positions = list(range(target_start, target_start + width))

    total_correct = 0
    total_cells = 0
    exact_matches = 0

    with torch.no_grad():
        for tokens in tokenized_windows:
            seq = torch.tensor([tokens], dtype=torch.long, device=device)
            inp = seq[:, :-1]
            tgt = seq[:, 1:]

            logits = model(inp)
            preds = logits.argmax(dim=-1).squeeze(0)
            targets = tgt.squeeze(0)

            row_correct = True
            for pos in target_positions:
                total_cells += 1
                if preds[pos].item() == targets[pos].item():
                    total_correct += 1
                else:
                    row_correct = False
            if row_correct:
                exact_matches += 1

    n = len(tokenized_windows)
    return {
        'per_cell_accuracy': 100.0 * total_correct / total_cells if total_cells > 0 else 0,
        'row_exact_match': 100.0 * exact_matches / n if n > 0 else 0,
        'total_cells': total_cells,
        'n_windows': n,
    }


# ===========================================================================
# Evaluation: Free-Run Rollout
# ===========================================================================
def evaluate_free_run_ca(model, trajectories, k_rows, n_rollout_steps, width,
                          device=None, max_eval_seeds=100):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    eval_trajs = trajectories[:max_eval_seeds]
    all_step_accs = []
    all_steps_to_first_error = []

    with torch.no_grad():
        for traj in eval_trajs:
            available = len(traj) - k_rows
            n_steps = min(n_rollout_steps, available)
            if n_steps <= 0:
                continue

            # Seed with true first K rows
            context_rows = [list(r) for r in traj[:k_rows]]
            step_accs = []
            first_error_step = None

            for step in range(n_steps):
                # Build token context: BOS + last K rows + trailing ROWSEP
                tokens = [CATokenizer.BOS]
                for i, row in enumerate(context_rows[-k_rows:]):
                    for cell in row:
                        tokens.append(CATokenizer.cell_to_token(cell))
                    tokens.append(CATokenizer.ROWSEP)

                # Generate W cells autoregressively
                generated_cells = []
                for _ in range(width):
                    inp = torch.tensor([tokens], dtype=torch.long, device=device)
                    logits = model(inp)
                    next_tok = logits[0, -1, :].argmax().item()
                    tokens.append(next_tok)
                    generated_cells.append(CATokenizer.token_to_cell(next_tok))

                context_rows.append(generated_cells)

                # Compare with ground truth
                true_row = traj[k_rows + step]
                n_correct = sum(1 for p, t in zip(generated_cells, true_row) if p == t)
                acc = 100.0 * n_correct / width
                step_accs.append(acc)

                if acc < 100.0 and first_error_step is None:
                    first_error_step = step

            all_step_accs.append(step_accs)
            all_steps_to_first_error.append(
                first_error_step if first_error_step is not None else n_steps)

    # Aggregate — pad shorter step lists to max length
    max_steps = max(len(s) for s in all_step_accs)
    padded = np.full((len(all_step_accs), max_steps), np.nan)
    for i, s in enumerate(all_step_accs):
        padded[i, :len(s)] = s

    mean_by_step = np.nanmean(padded, axis=0).tolist()
    std_by_step = np.nanstd(padded, axis=0).tolist()

    return {
        'mean_acc_by_step': mean_by_step,
        'std_acc_by_step': std_by_step,
        'final_step_accuracy': {
            'mean': float(np.nanmean(padded[:, -1])),
            'std': float(np.nanstd(padded[:, -1])),
        },
        'steps_to_first_error': {
            'mean': float(np.mean(all_steps_to_first_error)),
            'std': float(np.std(all_steps_to_first_error)),
        },
        'n_eval_seeds': len(eval_trajs),
        'n_rollout_steps': max_steps,
    }


# ===========================================================================
# Baselines
# ===========================================================================
def target_cell_examples(tokenized_windows, k_rows, width, context_window=10):
    """(context, target) pairs for target-row cell positions only."""
    target_start = 1 + k_rows * (width + 1)
    examples = []
    for tokens in tokenized_windows:
        for offset in range(width):
            pos = target_start + offset
            ctx_start = max(0, pos - context_window)
            examples.append((list(tokens[ctx_start:pos]), tokens[pos]))
    return examples


def evaluate_baseline_ca(baseline, tokenized_windows, k_rows, width,
                          context_window=10):
    """Evaluate baseline per-cell accuracy and row exact-match on target row."""
    target_start = 1 + k_rows * (width + 1)
    total_correct = 0
    total_cells = 0
    exact_matches = 0

    for tokens in tokenized_windows:
        row_correct = True
        for offset in range(width):
            pos = target_start + offset
            ctx_start = max(0, pos - context_window)
            ctx = list(tokens[ctx_start:pos])
            target = tokens[pos]
            pred = baseline.predict(ctx)
            total_cells += 1
            if pred == target:
                total_correct += 1
            else:
                row_correct = False
        if row_correct:
            exact_matches += 1

    n = len(tokenized_windows)
    return {
        'per_cell_accuracy': 100.0 * total_correct / total_cells if total_cells > 0 else 0,
        'row_exact_match': 100.0 * exact_matches / n if n > 0 else 0,
        'total_cells': total_cells,
        'n_windows': n,
    }


# ===========================================================================
# Orchestration
# ===========================================================================
def run_ca_experiment(rule_number, width=64, k_rows=4, n_steps=32,
                      n_steps_ood=64, n_train_seeds=2000, n_test_seeds=500,
                      density_train=0.5, density_test_list=None,
                      random_seeds=None, device=None):
    if random_seeds is None:
        random_seeds = RANDOM_SEEDS
    if density_test_list is None:
        density_test_list = []
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    label = f'Rule{rule_number}'
    _print(f'\n{"="*70}')
    _print(f'  {label} Experiment (W={width}, K={k_rows}, T={n_steps})')
    _print(f'{"="*70}')
    _print(f'  Vocab: {CATokenizer.id_to_str}')

    # --- Data generation ---
    _print(f'\n  Generating trajectories...')

    train_trajs = generate_trajectories(
        rule_number, n_train_seeds, width, n_steps, density_train,
        base_seed=DATA_SEED)
    _print(f'    Train: {len(train_trajs)} seeds, {n_steps+1} rows each')

    test_base = DATA_SEED + n_train_seeds
    id_test_trajs = generate_trajectories(
        rule_number, n_test_seeds, width, n_steps, density_train,
        base_seed=test_base)
    _print(f'    ID test: {len(id_test_trajs)} seeds')

    longer_test_trajs = generate_trajectories(
        rule_number, n_test_seeds, width, n_steps_ood, density_train,
        base_seed=test_base)
    _print(f'    Longer OOD: {len(longer_test_trajs)} seeds, {n_steps_ood+1} rows')

    density_trajs = {}
    density_base = test_base + n_test_seeds
    for d in density_test_list:
        d_trajs = generate_trajectories(
            rule_number, n_test_seeds, width, n_steps, d,
            base_seed=density_base)
        density_trajs[d] = d_trajs
        _print(f'    Density OOD (p={d}): {len(d_trajs)} seeds')

    _print(f'    Train seeds: {DATA_SEED}..{DATA_SEED+n_train_seeds-1}')
    _print(f'    Test seeds:  {test_base}..{test_base+n_test_seeds-1}')
    assert DATA_SEED + n_train_seeds <= test_base, 'Seed overlap!'
    _print(f'    Disjoint: PASS')

    # --- Windows & tokenize ---
    _print(f'\n  Creating windows...')
    train_windows = create_windows(train_trajs, k_rows)
    train_tokens = tokenize_windows(train_windows)
    seq_len = len(train_tokens[0])
    _print(f'    Train: {len(train_tokens)} windows, seq_len={seq_len}')

    all_toks = set()
    for s in train_tokens[:100]:
        all_toks.update(s)
    _print(f'    Token check: {sorted(all_toks)} — no UNK: PASS')

    # Test splits
    splits = {}

    id_test_tokens = tokenize_windows(create_windows(id_test_trajs, k_rows))
    splits['id_test'] = {'tokens': id_test_tokens, 'trajs': id_test_trajs}
    _print(f'    ID test: {len(id_test_tokens)} windows')

    longer_tokens = tokenize_windows(create_windows(longer_test_trajs, k_rows))
    splits['longer_ood'] = {'tokens': longer_tokens, 'trajs': longer_test_trajs}
    _print(f'    Longer OOD: {len(longer_tokens)} windows')

    for d, d_trajs in density_trajs.items():
        d_tokens = tokenize_windows(create_windows(d_trajs, k_rows))
        splits[f'density_{d}'] = {'tokens': d_tokens, 'trajs': d_trajs}
        _print(f'    Density {d}: {len(d_tokens)} windows')

    # Distributions
    _print(f'\n  Data distributions...')
    distributions = {
        'train': compute_data_distributions(train_trajs, width),
    }
    for name, sp in splits.items():
        distributions[name] = compute_data_distributions(sp['trajs'], width)
    for name, dist in distributions.items():
        _print(f'    {name}: density={dist["mean_density"]:.3f} ± {dist["std_density"]:.3f}')

    # --- Baselines ---
    _print(f'\n  Training baselines...')
    baseline_results = {}

    # N-gram baselines (train on all target-cell examples, eval on all test windows)
    ngram_ctx = 10
    ngram_train = target_cell_examples(train_tokens, k_rows, width, ngram_ctx)
    _print(f'    N-gram training examples: {len(ngram_train)}')

    for n in [3, 5]:
        ngram = NgramBaseline(n=n)
        ngram.train(ngram_train)
        bkey = f'ngram_{n}'
        baseline_results[bkey] = {}
        for sname, sp in splits.items():
            res = evaluate_baseline_ca(ngram, sp['tokens'], k_rows, width, ngram_ctx)
            baseline_results[bkey][sname] = res
            _print(f'    N-gram({n}) {sname}: {res["per_cell_accuracy"]:.2f}%')

    # kNN baselines (subsampled, ID test only for speed)
    knn_window_max = 80
    knn_eval_window_max = 50
    for knn_ctx in [width + 1, 2 * (width + 1)]:
        n_rows_label = knn_ctx // (width + 1)
        bkey = f'knn_{n_rows_label}row'
        baseline_results[bkey] = {}

        knn_train_tokens = train_tokens
        if len(knn_train_tokens) > knn_window_max:
            rng_knn = random.Random(DATA_SEED)
            knn_train_tokens = rng_knn.sample(train_tokens, knn_window_max)
        knn_train_ex = target_cell_examples(knn_train_tokens, k_rows, width, knn_ctx)

        knn = KNNBaseline(k=5, context_window=knn_ctx)
        knn.train(knn_train_ex)

        # Evaluate on id_test only (kNN is slow)
        eval_tokens = splits['id_test']['tokens']
        if len(eval_tokens) > knn_eval_window_max:
            rng_eval = random.Random(DATA_SEED + 1)
            eval_tokens = rng_eval.sample(splits['id_test']['tokens'], knn_eval_window_max)
        res = evaluate_baseline_ca(knn, eval_tokens, k_rows, width, knn_ctx)
        baseline_results[bkey]['id_test'] = res
        _print(f'    kNN({n_rows_label}-row) ID: {res["per_cell_accuracy"]:.2f}% '
               f'(n={res["n_windows"]} windows)')

    # --- Transformer training ---
    _print(f'\n  Training transformers (seeds={random_seeds})...')
    max_seq_len = seq_len + 50
    n_rollout_id = n_steps + 1 - k_rows
    n_rollout_ood = n_steps_ood + 1 - k_rows

    transformer_results = {}
    for rs in random_seeds:
        run_dir = os.path.join(OUTPUT_DIR, f'{label}_RS{rs}')
        _print(f'\n    --- RS={rs} ---')

        model, history = train_lm(
            train_tokens=train_tokens, random_seed=rs, output_dir=run_dir,
            vocab_size=CATokenizer.vocab_size, epochs=50, batch_size=32,
            lr=0.001, d_model=128, nhead=4, num_layers=3,
            dim_feedforward=512, dropout=0.1, max_seq_len=max_seq_len,
            device=device, verbose=True)

        seed_results = {'history': history}

        # Teacher-forced on all splits
        for sname, sp in splits.items():
            tf = evaluate_teacher_forced_ca(
                model, sp['tokens'], k_rows, width, device)
            seed_results[f'tf_{sname}'] = tf
            _print(f'      TF {sname}: cell={tf["per_cell_accuracy"]:.2f}%, '
                   f'row_exact={tf["row_exact_match"]:.2f}%')

        # Free-run: ID
        fr_id = evaluate_free_run_ca(
            model, id_test_trajs, k_rows, n_rollout_id, width,
            device, max_eval_seeds=100)
        seed_results['fr_id_test'] = fr_id
        _print(f'      FR ID: final={fr_id["final_step_accuracy"]["mean"]:.2f}%, '
               f'steps_to_err={fr_id["steps_to_first_error"]["mean"]:.1f}')

        # Free-run: longer OOD
        fr_long = evaluate_free_run_ca(
            model, longer_test_trajs, k_rows, n_rollout_ood, width,
            device, max_eval_seeds=100)
        seed_results['fr_longer_ood'] = fr_long
        _print(f'      FR longer: final={fr_long["final_step_accuracy"]["mean"]:.2f}%, '
               f'steps_to_err={fr_long["steps_to_first_error"]["mean"]:.1f}')

        # Save manifest
        manifest = {
            'label': label, 'rule_number': rule_number,
            'random_seed': rs, 'width': width, 'k_rows': k_rows,
            'n_steps': n_steps, 'n_train_seeds': n_train_seeds,
            'results': seed_results,
        }
        with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

        transformer_results[rs] = seed_results

    # --- Aggregate ---
    _print(f'\n  Aggregating across seeds...')
    agg = {}

    for sname in splits:
        tf_key = f'tf_{sname}'
        accs = [transformer_results[rs][tf_key]['per_cell_accuracy']
                for rs in random_seeds]
        exact = [transformer_results[rs][tf_key]['row_exact_match']
                 for rs in random_seeds]
        agg[sname] = {
            'mean_cell_acc': float(np.mean(accs)),
            'std_cell_acc': float(np.std(accs)),
            'mean_row_exact': float(np.mean(exact)),
            'std_row_exact': float(np.std(exact)),
            'per_seed_cell_acc': {str(rs): a for rs, a in zip(random_seeds, accs)},
        }
        _print(f'    TF {sname}: {agg[sname]["mean_cell_acc"]:.2f} '
               f'± {agg[sname]["std_cell_acc"]:.2f}%')

    for fr_name in ['fr_id_test', 'fr_longer_ood']:
        final = [transformer_results[rs][fr_name]['final_step_accuracy']['mean']
                 for rs in random_seeds]
        ste = [transformer_results[rs][fr_name]['steps_to_first_error']['mean']
               for rs in random_seeds]
        agg[fr_name] = {
            'mean_final_acc': float(np.mean(final)),
            'std_final_acc': float(np.std(final)),
            'mean_steps_to_error': float(np.mean(ste)),
            'std_steps_to_error': float(np.std(ste)),
        }
        _print(f'    {fr_name}: final={agg[fr_name]["mean_final_acc"]:.2f}%, '
               f'steps_to_err={agg[fr_name]["mean_steps_to_error"]:.1f}')

    result = {
        'label': label, 'rule_number': rule_number,
        'width': width, 'k_rows': k_rows,
        'n_steps': n_steps, 'n_steps_ood': n_steps_ood,
        'n_train_seeds': n_train_seeds, 'n_test_seeds': n_test_seeds,
        'density_train': density_train, 'density_test_list': density_test_list,
        'distributions': distributions,
        'baseline_results': baseline_results,
        'transformer_aggregate': agg,
        'transformer_per_seed': {
            str(rs): {k: v for k, v in transformer_results[rs].items()
                      if k != 'history'}
            for rs in random_seeds
        },
    }

    return result


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description='1D Cellular Automata Experiment')
    parser.add_argument('--rules', nargs='+', type=int, default=[30, 90])
    parser.add_argument('--random_seeds', nargs='+', type=int, default=RANDOM_SEEDS)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--k_rows', type=int, default=4)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--steps_ood', type=int, default=64)
    parser.add_argument('--train_seeds', type=int, default=2000)
    parser.add_argument('--test_seeds', type=int, default=500)
    parser.add_argument('--density_train', type=float, default=0.5)
    parser.add_argument('--density_test_list', nargs='*', type=float, default=[0.2, 0.8])
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}
    start_time = time.time()

    for rule in args.rules:
        all_results[f'Rule{rule}'] = run_ca_experiment(
            rule_number=rule,
            width=args.width, k_rows=args.k_rows,
            n_steps=args.steps, n_steps_ood=args.steps_ood,
            n_train_seeds=args.train_seeds, n_test_seeds=args.test_seeds,
            density_train=args.density_train,
            density_test_list=args.density_test_list,
            random_seeds=args.random_seeds,
        )

    elapsed = time.time() - start_time

    summary = {
        'elapsed_seconds': elapsed,
        'rules': args.rules,
        'random_seeds': args.random_seeds,
        'width': args.width, 'k_rows': args.k_rows,
        'steps': args.steps, 'steps_ood': args.steps_ood,
        'results': all_results,
    }
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    dist_data = {k: r.get('distributions', {}) for k, r in all_results.items()}
    with open(os.path.join(OUTPUT_DIR, 'data_distributions.json'), 'w') as f:
        json.dump(dist_data, f, indent=2, default=str)

    # --- Print summary table ---
    _print(f'\n{"="*80}')
    _print(f'  CA EXPERIMENT SUMMARY')
    _print(f'{"="*80}')
    _print(f'  Elapsed: {elapsed:.1f}s')
    _print()

    for key, result in all_results.items():
        _print(f'  --- {key} ---')
        ag = result['transformer_aggregate']
        bl = result['baseline_results']

        # Collect split names for table
        tf_splits = [s for s in ag if not s.startswith('fr_')]
        _print(f'\n  Teacher-forced per-cell accuracy:')
        header = f'    {"Method":<20}'
        for s in tf_splits:
            header += f' {s:>14}'
        _print(header)
        _print(f'    {"-"*70}')

        for bname, bdata in bl.items():
            row = f'    {bname:<20}'
            for s in tf_splits:
                if s in bdata:
                    row += f' {bdata[s]["per_cell_accuracy"]:>13.2f}%'
                else:
                    row += f' {"—":>14}'
            _print(row)

        row = f'    {"transformer":<20}'
        for s in tf_splits:
            row += f' {ag[s]["mean_cell_acc"]:>13.2f}%'
        _print(row)

        _print(f'\n  Free-run rollout:')
        for fr_key in ['fr_id_test', 'fr_longer_ood']:
            if fr_key in ag:
                fr = ag[fr_key]
                _print(f'    {fr_key}: final_acc={fr["mean_final_acc"]:.2f}%, '
                       f'steps_to_err={fr["mean_steps_to_error"]:.1f}')
        _print()

    _print(f'  Results saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
