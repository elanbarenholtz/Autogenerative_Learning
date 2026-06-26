"""
Parity CA Experiment with Pattern Holdout

Tests whether transformers learn the parity rule (radius-5, 11-bit neighborhood,
2048 possible patterns) or merely memorize seen neighborhood-to-output mappings.

Training rows contain only neighborhoods from set A (40% of 2048).
OOD test rows contain neighborhoods from set B (the remaining 60%).
If the transformer generalizes to B, it learned parity as a rule;
if accuracy collapses on B, it memorized A-pattern mappings.

A symbolic oracle proves the task is unambiguous (100% on all splits).
"""
import argparse
import functools
import json
import operator
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import FibonacciTransformer, count_parameters
from experiment_framework import DATA_SEED, RANDOM_SEEDS, set_all_seeds
from exp5_baselines import NgramBaseline, KNNBaseline

OUTPUT_DIR = 'runs/parity_ca'

RADIUS = 5
NEIGHBORHOOD_SIZE = 2 * RADIUS + 1  # 11
N_PATTERNS = 2 ** NEIGHBORHOOD_SIZE  # 2048


def _print(msg: str = ''):
    print(msg, flush=True)


# ===========================================================================
# Tokenizer (same as exp_ca.py)
# ===========================================================================
class CATokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    ZERO = 3
    ONE = 4
    ROWSEP = 5

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
# Parity Rule & Neighborhood Extraction
# ===========================================================================
def apply_parity_rule(row, radius=RADIUS):
    """Apply radius-5 parity CA: each cell = XOR of its 11-cell neighborhood."""
    W = len(row)
    return [functools.reduce(operator.xor,
            [row[(i + d) % W] for d in range(-radius, radius + 1)])
            for i in range(W)]


def get_neighborhood(row, i, radius=RADIUS):
    """Return 11-bit neighborhood of cell i as integer."""
    W = len(row)
    bits = 0
    for d in range(-radius, radius + 1):
        bits = (bits << 1) | row[(i + d) % W]
    return bits


def get_all_neighborhoods(row, radius=RADIUS):
    return [get_neighborhood(row, i, radius) for i in range(len(row))]


# ===========================================================================
# A/B Set Selection
# ===========================================================================
def select_allowed_set(n_patterns=N_PATTERNS, fraction=0.4, seed=DATA_SEED):
    rng = random.Random(seed)
    all_patterns = list(range(n_patterns))
    rng.shuffle(all_patterns)
    n_a = int(n_patterns * fraction)
    return frozenset(all_patterns[:n_a]), frozenset(all_patterns[n_a:])


# ===========================================================================
# Constraint Graph & A-Only Row Generation
# ===========================================================================
def build_constraint_graph(allowed_set):
    """Build directed graph: edges are allowed 11-bit patterns.

    Nodes are 10-bit strings (overlapping prefix/suffix of consecutive
    neighborhoods). Each allowed 11-bit pattern p creates an edge from
    (p >> 1) to (p & 0x3FF) with new_bit = p & 1.
    """
    overlap = NEIGHBORHOOD_SIZE - 1  # 10
    mask = (1 << overlap) - 1
    graph = defaultdict(list)
    for p in allowed_set:
        src = p >> 1
        dst = p & mask
        graph[src].append((dst, p & 1))
    return graph


def _build_allowed_lookup(allowed_set):
    """Precompute: for each 10-bit prefix, which bits (0/1) yield an allowed 11-bit pattern."""
    lookup = {}  # 10-bit prefix -> set of valid next bits
    for prefix in range(1024):
        valid = []
        for bit in [0, 1]:
            pattern = (prefix << 1) | bit
            if pattern in allowed_set:
                valid.append(bit)
        if valid:
            lookup[prefix] = valid
    return lookup


def generate_a_only_row(width, allowed_set, graph, rng, max_attempts=500,
                        _lookup_cache={}):
    """Generate a row where all neighborhoods are in A.

    Uses backtracking search with precomputed valid-bit lookup tables for speed.
    """
    # Cache the lookup table (keyed by frozenset id)
    cache_key = id(allowed_set)
    if cache_key not in _lookup_cache:
        _lookup_cache[cache_key] = _build_allowed_lookup(allowed_set)
    lookup = _lookup_cache[cache_key]

    overlap = NEIGHBORHOOD_SIZE - 1  # 10

    for attempt in range(max_attempts):
        row = [0] * width
        for i in range(overlap):
            row[i] = rng.randint(0, 1)

        if _fill_row_backtrack(row, overlap, width, allowed_set, lookup, rng):
            return row

    raise RuntimeError(f"Failed to generate A-only row after {max_attempts} attempts "
                       f"(width={width}, |A|={len(allowed_set)})")


def _fill_row_backtrack(row, pos, width, allowed_set, lookup, rng,
                        max_backtracks=500000):
    """Fill row[pos:] such that all neighborhoods are in allowed_set.

    Uses iterative backtracking with precomputed lookup for valid next bits.
    The lookup maps 10-bit prefixes to valid continuation bits, avoiding
    expensive get_neighborhood calls during the inner loop.
    """
    radius = RADIUS
    nbr_size = NEIGHBORHOOD_SIZE
    stack = []  # (position, bits_tried)
    bits_tried = set()
    backtracks = 0

    while pos < width:
        if backtracks > max_backtracks:
            return False

        # Get the 10-bit prefix (the 10 cells ending at pos-1) to look up valid bits
        center = pos - radius
        if radius <= center <= width - radius - 1:
            # Non-wraparound: compute the 10-bit prefix from row[center-radius..center+radius-1]
            prefix = 0
            for d in range(-radius, radius):
                prefix = (prefix << 1) | row[center + d]
            candidates = lookup.get(prefix, [])
        else:
            candidates = [0, 1]

        rng.shuffle(candidates)
        found = False

        for bit in candidates:
            if bit in bits_tried:
                continue
            row[pos] = bit

            ok = True
            # At the last position, check all wraparound neighborhoods
            if pos == width - 1:
                for c in list(range(radius)) + list(range(width - radius, width)):
                    if get_neighborhood(row, c, radius) not in allowed_set:
                        ok = False
                        break

            if ok:
                stack.append((pos, bits_tried))
                bits_tried = set()
                pos += 1
                found = True
                break
            else:
                bits_tried.add(bit)

        if not found:
            backtracks += 1
            if not stack:
                return False
            pos, bits_tried = stack.pop()
            bits_tried.add(row[pos])

    return True


# ===========================================================================
# Data Generation
# ===========================================================================
def generate_training_data(n_rows, width, allowed_set, graph, seed):
    """A-only input rows + parity targets."""
    rng = random.Random(seed)
    windows = []
    for _ in range(n_rows):
        input_row = generate_a_only_row(width, allowed_set, graph, rng)
        target_row = apply_parity_rule(input_row)
        windows.append((input_row, target_row))
    return windows


def generate_test_data(n_rows, width, seed):
    """Random (unconstrained) input rows + parity targets."""
    rng = random.Random(seed)
    windows = []
    for _ in range(n_rows):
        input_row = [rng.randint(0, 1) for _ in range(width)]
        target_row = apply_parity_rule(input_row)
        windows.append((input_row, target_row))
    return windows


def generate_id_test_data(n_rows, width, allowed_set, graph, seed):
    """A-only input rows (different seed from training)."""
    return generate_training_data(n_rows, width, allowed_set, graph, seed)


def generate_max_b_test_data(n_rows, width, allowed_set, seed,
                              min_b_fraction=0.5, max_attempts=500000):
    """Rejection-sample rows with above-median B neighborhood fraction.

    With a_fraction=0.6, random rows have ~40% B neighborhoods (mean).
    Rows with >50% B are in the top ~7% tail, giving a meaningful enrichment.
    The actual achieved B-fraction is logged in the output.
    """
    rng = random.Random(seed)
    windows = []
    b_fracs_accepted = []
    attempts = 0
    while len(windows) < n_rows and attempts < max_attempts:
        row = [rng.randint(0, 1) for _ in range(width)]
        nbrs = get_all_neighborhoods(row)
        b_frac = sum(1 for n in nbrs if n not in allowed_set) / width
        if b_frac >= min_b_fraction:
            windows.append((row, apply_parity_rule(row)))
            b_fracs_accepted.append(b_frac)
        attempts += 1

    if len(windows) < n_rows:
        _print(f'    WARNING: only generated {len(windows)}/{n_rows} max-B rows '
               f'(threshold={min_b_fraction}) after {max_attempts} attempts.')

    if b_fracs_accepted:
        _print(f'    Max-B stats: mean_b_frac={np.mean(b_fracs_accepted):.3f}, '
               f'min={min(b_fracs_accepted):.3f}, max={max(b_fracs_accepted):.3f}')

    return windows


def tokenize_windows(windows, k_rows=1):
    """Tokenize (input_row, target_row) pairs into token sequences."""
    return [CATokenizer.tokenize_rows([inp, tgt]) for inp, tgt in windows]


# ===========================================================================
# Neighborhood Distribution Stats
# ===========================================================================
def compute_neighborhood_stats(windows, allowed_set, width):
    """Compute A/B neighborhood fractions for a set of windows."""
    a_count = 0
    b_count = 0
    for input_row, _ in windows:
        nbrs = get_all_neighborhoods(input_row)
        for n in nbrs:
            if n in allowed_set:
                a_count += 1
            else:
                b_count += 1
    total = a_count + b_count
    return {
        'a_count': a_count, 'b_count': b_count, 'total': total,
        'a_fraction': a_count / total if total > 0 else 0,
        'b_fraction': b_count / total if total > 0 else 0,
        'n_windows': len(windows),
    }


# ===========================================================================
# Dataset & Collation (from exp_ca.py)
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
# Training (from exp_ca.py)
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
# Symbolic Oracle
# ===========================================================================
def evaluate_oracle(windows, width, allowed_set, radius=RADIUS):
    """Deterministic parity rule -- must get 100% on all splits."""
    counters = {'overall': [0, 0], 'A': [0, 0], 'B': [0, 0]}  # [correct, total]
    exact = 0
    for input_row, target_row in windows:
        predicted = apply_parity_rule(input_row, radius)
        row_ok = True
        for i in range(width):
            nbr = get_neighborhood(input_row, i, radius)
            cls = 'A' if nbr in allowed_set else 'B'
            ok = (predicted[i] == target_row[i])
            counters['overall'][1] += 1
            counters[cls][1] += 1
            if ok:
                counters['overall'][0] += 1
                counters[cls][0] += 1
            else:
                row_ok = False
        if row_ok:
            exact += 1

    def pct(c):
        return 100.0 * c[0] / c[1] if c[1] > 0 else 0.0

    return {
        'overall_cell_acc': pct(counters['overall']),
        'a_cell_acc': pct(counters['A']),
        'b_cell_acc': pct(counters['B']),
        'row_exact_match': 100.0 * exact / len(windows) if windows else 0.0,
        'a_total': counters['A'][1],
        'b_total': counters['B'][1],
    }


# ===========================================================================
# Evaluation: Teacher-Forced with A/B Breakdown
# ===========================================================================
def evaluate_teacher_forced_parity(model, token_seqs, raw_windows,
                                    width, allowed_set, k_rows=1,
                                    radius=RADIUS, device=None):
    """Per-cell accuracy broken down by A vs B neighborhood class."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # In shifted tensor (tokens[1:]), target row starts after k_rows*(width+1)
    # because each input row = width cells + 1 ROWSEP, preceded by BOS
    target_start = k_rows * (width + 1)
    target_positions = list(range(target_start, target_start + width))

    counters = {'overall': [0, 0], 'A': [0, 0], 'B': [0, 0]}
    exact = 0

    model.eval()
    with torch.no_grad():
        for tokens, (input_row, _) in zip(token_seqs, raw_windows):
            seq = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(seq[:, :-1])
            preds = logits.argmax(dim=-1).squeeze(0)
            tgt = seq[:, 1:].squeeze(0)

            row_ok = True
            for cell_i, pos in enumerate(target_positions):
                nbr = get_neighborhood(input_row, cell_i, radius)
                cls = 'A' if nbr in allowed_set else 'B'
                ok = (preds[pos].item() == tgt[pos].item())
                counters['overall'][1] += 1
                counters[cls][1] += 1
                if ok:
                    counters['overall'][0] += 1
                    counters[cls][0] += 1
                else:
                    row_ok = False
            if row_ok:
                exact += 1

    def pct(c):
        return 100.0 * c[0] / c[1] if c[1] > 0 else 0.0

    return {
        'overall_cell_acc': pct(counters['overall']),
        'a_cell_acc': pct(counters['A']),
        'b_cell_acc': pct(counters['B']),
        'row_exact_match': 100.0 * exact / len(token_seqs) if token_seqs else 0.0,
        'a_total': counters['A'][1],
        'b_total': counters['B'][1],
    }


# ===========================================================================
# Evaluation: Baselines with A/B Breakdown
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


def evaluate_baseline_parity(baseline, tokenized_windows, raw_windows,
                              k_rows, width, allowed_set,
                              context_window=10, radius=RADIUS):
    """Evaluate baseline per-cell accuracy with A/B breakdown."""
    target_start = 1 + k_rows * (width + 1)

    counters = {'overall': [0, 0], 'A': [0, 0], 'B': [0, 0]}
    exact = 0

    for tokens, (input_row, _) in zip(tokenized_windows, raw_windows):
        row_ok = True
        for cell_i in range(width):
            pos = target_start + cell_i
            ctx_start = max(0, pos - context_window)
            ctx = list(tokens[ctx_start:pos])
            target = tokens[pos]
            pred = baseline.predict(ctx)

            nbr = get_neighborhood(input_row, cell_i, radius)
            cls = 'A' if nbr in allowed_set else 'B'
            ok = (pred == target)

            counters['overall'][1] += 1
            counters[cls][1] += 1
            if ok:
                counters['overall'][0] += 1
                counters[cls][0] += 1
            else:
                row_ok = False
        if row_ok:
            exact += 1

    def pct(c):
        return 100.0 * c[0] / c[1] if c[1] > 0 else 0.0

    n = len(tokenized_windows)
    return {
        'overall_cell_acc': pct(counters['overall']),
        'a_cell_acc': pct(counters['A']),
        'b_cell_acc': pct(counters['B']),
        'row_exact_match': 100.0 * exact / n if n > 0 else 0.0,
        'a_total': counters['A'][1],
        'b_total': counters['B'][1],
    }


# ===========================================================================
# Evaluation: Free-Run Rollout with A/B Breakdown
# ===========================================================================
def evaluate_free_run_parity(model, raw_windows, k_rows, width, allowed_set,
                              n_rollout_steps=20, radius=RADIUS, device=None,
                              max_eval=100):
    """Multi-step autoregressive rollout with A/B accuracy breakdown."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    eval_windows = raw_windows[:max_eval]
    all_step_accs = []
    all_step_a_accs = []
    all_step_b_accs = []

    with torch.no_grad():
        for input_row, _ in eval_windows:
            current_row = list(input_row)
            step_accs = []
            step_a_accs = []
            step_b_accs = []

            for step in range(n_rollout_steps):
                true_target = apply_parity_rule(current_row, radius)

                # Build token context: BOS + current_row + ROWSEP
                tokens = [CATokenizer.BOS]
                for cell in current_row:
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

                # Per-cell accuracy with A/B breakdown
                a_correct, a_total = 0, 0
                b_correct, b_total = 0, 0
                total_correct = 0
                for i in range(width):
                    nbr = get_neighborhood(current_row, i, radius)
                    ok = (generated_cells[i] == true_target[i])
                    if ok:
                        total_correct += 1
                    if nbr in allowed_set:
                        a_total += 1
                        if ok:
                            a_correct += 1
                    else:
                        b_total += 1
                        if ok:
                            b_correct += 1

                step_accs.append(100.0 * total_correct / width)
                step_a_accs.append(100.0 * a_correct / a_total if a_total > 0 else float('nan'))
                step_b_accs.append(100.0 * b_correct / b_total if b_total > 0 else float('nan'))

                # Use predicted row as next input
                current_row = generated_cells

            all_step_accs.append(step_accs)
            all_step_a_accs.append(step_a_accs)
            all_step_b_accs.append(step_b_accs)

    def agg_steps(all_lists):
        max_steps = max(len(s) for s in all_lists)
        padded = np.full((len(all_lists), max_steps), np.nan)
        for i, s in enumerate(all_lists):
            padded[i, :len(s)] = s
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return {
                'mean_by_step': np.nanmean(padded, axis=0).tolist(),
                'std_by_step': np.nanstd(padded, axis=0).tolist(),
                'final_mean': float(np.nanmean(padded[:, -1])),
                'final_std': float(np.nanstd(padded[:, -1])),
            }

    return {
        'overall': agg_steps(all_step_accs),
        'a_neighborhoods': agg_steps(all_step_a_accs),
        'b_neighborhoods': agg_steps(all_step_b_accs),
        'n_eval': len(eval_windows),
        'n_rollout_steps': n_rollout_steps,
    }


# ===========================================================================
# Orchestration
# ===========================================================================
def run_parity_ca_experiment(width=64, k_rows=1, a_fraction=0.4,
                              n_train=5000, n_test=1000,
                              random_seeds=None, epochs=50, batch_size=32,
                              device=None):
    if random_seeds is None:
        random_seeds = RANDOM_SEEDS
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _print(f'\n{"="*70}')
    _print(f'  Parity CA Experiment (W={width}, K={k_rows}, '
           f'A_frac={a_fraction}, R={RADIUS})')
    _print(f'{"="*70}')
    _print(f'  Neighborhood size: {NEIGHBORHOOD_SIZE} bits')
    _print(f'  Total patterns: {N_PATTERNS}')
    _print(f'  Device: {device}')

    # --- 1. Select A/B sets ---
    _print(f'\n  Selecting A/B pattern sets...')
    allowed_set, blocked_set = select_allowed_set(N_PATTERNS, a_fraction, DATA_SEED)
    _print(f'    |A| = {len(allowed_set)} ({100*len(allowed_set)/N_PATTERNS:.1f}%)')
    _print(f'    |B| = {len(blocked_set)} ({100*len(blocked_set)/N_PATTERNS:.1f}%)')

    # --- 2. Build constraint graph ---
    _print(f'\n  Building constraint graph...')
    graph = build_constraint_graph(allowed_set)
    nodes_with_edges = sum(1 for n in graph if graph[n])
    total_edges = sum(len(edges) for edges in graph.values())
    avg_degree = total_edges / nodes_with_edges if nodes_with_edges > 0 else 0
    _print(f'    Nodes with outgoing edges: {nodes_with_edges}/1024')
    _print(f'    Total edges: {total_edges}')
    _print(f'    Avg out-degree: {avg_degree:.2f}')

    if nodes_with_edges < 100:
        raise RuntimeError(f"Graph too sparse ({nodes_with_edges} nodes with edges). "
                           f"Adjust --a_fraction (currently {a_fraction}).")

    # --- 3. Generate training data ---
    _print(f'\n  Generating training data ({n_train} A-only rows)...')
    t0 = time.time()
    train_windows = generate_training_data(n_train, width, allowed_set, graph, DATA_SEED)
    _print(f'    Generated {len(train_windows)} windows in {time.time()-t0:.1f}s')

    # --- 4. Generate test data ---
    _print(f'\n  Generating test data...')
    test_seed_base = DATA_SEED + 100000

    id_test_windows = generate_id_test_data(
        n_test, width, allowed_set, graph, test_seed_base)
    _print(f'    ID test (A-only): {len(id_test_windows)} windows')

    ood_b_windows = generate_test_data(n_test, width, test_seed_base + 1)
    _print(f'    OOD B-inclusive (random): {len(ood_b_windows)} windows')

    t0 = time.time()
    max_b_windows = generate_max_b_test_data(
        n_test, width, allowed_set, test_seed_base + 2)
    _print(f'    OOD max-B (>90% B): {len(max_b_windows)} windows in {time.time()-t0:.1f}s')

    # --- 5. Tokenize ---
    _print(f'\n  Tokenizing...')
    train_tokens = tokenize_windows(train_windows, k_rows)
    seq_len = len(train_tokens[0])
    _print(f'    Seq length: {seq_len} tokens')

    splits = {
        'id_test': {'windows': id_test_windows,
                     'tokens': tokenize_windows(id_test_windows, k_rows)},
        'ood_b_inclusive': {'windows': ood_b_windows,
                            'tokens': tokenize_windows(ood_b_windows, k_rows)},
        'ood_max_b': {'windows': max_b_windows,
                       'tokens': tokenize_windows(max_b_windows, k_rows)},
    }

    # --- 6. Neighborhood distribution stats ---
    _print(f'\n  Neighborhood distributions:')
    distributions = {
        'train': compute_neighborhood_stats(train_windows, allowed_set, width),
    }
    for sname, sp in splits.items():
        distributions[sname] = compute_neighborhood_stats(sp['windows'], allowed_set, width)

    for name, dist in distributions.items():
        _print(f'    {name}: A={dist["a_fraction"]:.3f}, B={dist["b_fraction"]:.3f} '
               f'({dist["n_windows"]} windows)')

    # --- 7. Symbolic oracle ---
    _print(f'\n  Running symbolic oracle...')
    oracle_results = {}
    for sname in ['train'] + list(splits.keys()):
        windows = train_windows if sname == 'train' else splits[sname]['windows']
        oracle = evaluate_oracle(windows, width, allowed_set)
        oracle_results[sname] = oracle
        _print(f'    Oracle {sname}: overall={oracle["overall_cell_acc"]:.1f}%, '
               f'A={oracle["a_cell_acc"]:.1f}%, B={oracle["b_cell_acc"]:.1f}%, '
               f'row_exact={oracle["row_exact_match"]:.1f}%')
        assert oracle['overall_cell_acc'] == 100.0, \
            f"Oracle failed on {sname}! ({oracle['overall_cell_acc']:.2f}% != 100%)"
    _print(f'    Oracle 100% on all splits: PASS')

    # --- 8. Baselines ---
    _print(f'\n  Training baselines...')
    baseline_results = {}

    ngram_ctx = 10
    ngram_train = target_cell_examples(train_tokens, k_rows, width, ngram_ctx)
    _print(f'    N-gram training examples: {len(ngram_train)}')

    for n in [3, 5]:
        ngram = NgramBaseline(n=n)
        ngram.train(ngram_train)
        bkey = f'ngram_{n}'
        baseline_results[bkey] = {}
        for sname, sp in splits.items():
            res = evaluate_baseline_parity(
                ngram, sp['tokens'], sp['windows'], k_rows, width,
                allowed_set, ngram_ctx)
            baseline_results[bkey][sname] = res
            _print(f'    N-gram({n}) {sname}: overall={res["overall_cell_acc"]:.2f}%, '
                   f'A={res["a_cell_acc"]:.2f}%, B={res["b_cell_acc"]:.2f}%')

    # kNN (subsampled for speed)
    knn_window_max = 80
    knn_eval_window_max = 50
    knn_ctx = width + 1
    bkey = 'knn_1row'
    baseline_results[bkey] = {}

    knn_train_tokens = train_tokens
    if len(knn_train_tokens) > knn_window_max:
        rng_knn = random.Random(DATA_SEED)
        knn_train_tokens = rng_knn.sample(train_tokens, knn_window_max)
    knn_train_ex = target_cell_examples(knn_train_tokens, k_rows, width, knn_ctx)

    knn = KNNBaseline(k=5, context_window=knn_ctx)
    knn.train(knn_train_ex)

    # kNN eval on id_test only (slow)
    eval_tokens = splits['id_test']['tokens']
    eval_windows_knn = splits['id_test']['windows']
    if len(eval_tokens) > knn_eval_window_max:
        rng_eval = random.Random(DATA_SEED + 1)
        indices = rng_eval.sample(range(len(eval_tokens)), knn_eval_window_max)
        eval_tokens = [eval_tokens[i] for i in indices]
        eval_windows_knn = [eval_windows_knn[i] for i in indices]

    res = evaluate_baseline_parity(
        knn, eval_tokens, eval_windows_knn, k_rows, width,
        allowed_set, knn_ctx)
    baseline_results[bkey]['id_test'] = res
    _print(f'    kNN(1-row) ID: overall={res["overall_cell_acc"]:.2f}%, '
           f'A={res["a_cell_acc"]:.2f}%')

    # --- 9. Transformer training ---
    _print(f'\n  Training transformers (seeds={random_seeds})...')
    max_seq_len = seq_len + 50

    transformer_results = {}
    for rs in random_seeds:
        run_dir = os.path.join(OUTPUT_DIR, f'ParityCA_RS{rs}')
        _print(f'\n    --- RS={rs} ---')

        model, history = train_lm(
            train_tokens=train_tokens, random_seed=rs, output_dir=run_dir,
            vocab_size=CATokenizer.vocab_size, epochs=epochs, batch_size=batch_size,
            lr=0.001, d_model=128, nhead=4, num_layers=3,
            dim_feedforward=512, dropout=0.1, max_seq_len=max_seq_len,
            device=device, verbose=True)

        seed_results = {'history': history}

        # Teacher-forced on all splits
        for sname, sp in splits.items():
            tf = evaluate_teacher_forced_parity(
                model, sp['tokens'], sp['windows'], width, allowed_set,
                k_rows, RADIUS, device)
            seed_results[f'tf_{sname}'] = tf
            _print(f'      TF {sname}: overall={tf["overall_cell_acc"]:.2f}%, '
                   f'A={tf["a_cell_acc"]:.2f}%, B={tf["b_cell_acc"]:.2f}%, '
                   f'row_exact={tf["row_exact_match"]:.2f}%')

        # Free-run on ood_b_inclusive
        fr = evaluate_free_run_parity(
            model, splits['ood_b_inclusive']['windows'], k_rows, width,
            allowed_set, n_rollout_steps=20, radius=RADIUS, device=device)
        seed_results['fr_ood_b_inclusive'] = fr
        _print(f'      FR ood_b: overall_final={fr["overall"]["final_mean"]:.2f}%, '
               f'A_final={fr["a_neighborhoods"]["final_mean"]:.2f}%, '
               f'B_final={fr["b_neighborhoods"]["final_mean"]:.2f}%')

        # Save manifest
        manifest = {
            'label': 'ParityCA', 'random_seed': rs,
            'width': width, 'k_rows': k_rows,
            'a_fraction': a_fraction, 'radius': RADIUS,
            'n_train': n_train, 'n_test': n_test,
            'results': seed_results,
        }
        with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

        transformer_results[rs] = seed_results

    # --- 10. Aggregate ---
    _print(f'\n  Aggregating across seeds...')
    agg = {}

    for sname in splits:
        tf_key = f'tf_{sname}'
        overall_accs = [transformer_results[rs][tf_key]['overall_cell_acc']
                        for rs in random_seeds]
        a_accs = [transformer_results[rs][tf_key]['a_cell_acc']
                  for rs in random_seeds]
        b_accs = [transformer_results[rs][tf_key]['b_cell_acc']
                  for rs in random_seeds]
        exact = [transformer_results[rs][tf_key]['row_exact_match']
                 for rs in random_seeds]

        # A-B gap (only meaningful when B cells exist)
        b_totals = [transformer_results[rs][tf_key]['b_total']
                    for rs in random_seeds]
        has_b = all(bt > 0 for bt in b_totals)
        if has_b:
            gaps = [a - b for a, b in zip(a_accs, b_accs)]
        else:
            gaps = [float('nan')] * len(random_seeds)

        agg[sname] = {
            'mean_overall_acc': float(np.mean(overall_accs)),
            'std_overall_acc': float(np.std(overall_accs)),
            'mean_a_acc': float(np.mean(a_accs)),
            'std_a_acc': float(np.std(a_accs)),
            'mean_b_acc': float(np.mean(b_accs)) if has_b else float('nan'),
            'std_b_acc': float(np.std(b_accs)) if has_b else float('nan'),
            'mean_ab_gap': float(np.mean(gaps)) if has_b else float('nan'),
            'std_ab_gap': float(np.std(gaps)) if has_b else float('nan'),
            'mean_row_exact': float(np.mean(exact)),
            'std_row_exact': float(np.std(exact)),
            'has_b_neighborhoods': has_b,
            'per_seed_overall': {str(rs): a for rs, a in zip(random_seeds, overall_accs)},
            'per_seed_a': {str(rs): a for rs, a in zip(random_seeds, a_accs)},
            'per_seed_b': {str(rs): b for rs, b in zip(random_seeds, b_accs)},
        }

    for sname, ag in agg.items():
        b_str = f'B={ag["mean_b_acc"]:.2f}%' if ag.get('has_b_neighborhoods') else 'B=N/A'
        gap_str = (f'A-B gap={ag["mean_ab_gap"]:.2f} +/- {ag["std_ab_gap"]:.2f}%'
                   if ag.get('has_b_neighborhoods') else 'A-B gap=N/A (no B cells)')
        _print(f'    TF {sname}: overall={ag["mean_overall_acc"]:.2f}%, '
               f'A={ag["mean_a_acc"]:.2f}%, {b_str}, {gap_str}')

    # Free-run aggregate
    fr_overall = [transformer_results[rs]['fr_ood_b_inclusive']['overall']['final_mean']
                  for rs in random_seeds]
    fr_a = [transformer_results[rs]['fr_ood_b_inclusive']['a_neighborhoods']['final_mean']
            for rs in random_seeds]
    fr_b = [transformer_results[rs]['fr_ood_b_inclusive']['b_neighborhoods']['final_mean']
            for rs in random_seeds]
    agg['fr_ood_b_inclusive'] = {
        'mean_overall': float(np.mean(fr_overall)),
        'mean_a': float(np.mean(fr_a)),
        'mean_b': float(np.mean(fr_b)),
    }
    _print(f'    FR ood_b: overall={agg["fr_ood_b_inclusive"]["mean_overall"]:.2f}%, '
           f'A={agg["fr_ood_b_inclusive"]["mean_a"]:.2f}%, '
           f'B={agg["fr_ood_b_inclusive"]["mean_b"]:.2f}%')

    # --- 11. Save results ---
    result = {
        'label': 'ParityCA',
        'width': width, 'k_rows': k_rows,
        'a_fraction': a_fraction, 'radius': RADIUS,
        'n_patterns': N_PATTERNS,
        'n_a': len(allowed_set), 'n_b': len(blocked_set),
        'n_train': n_train, 'n_test': n_test,
        'random_seeds': random_seeds,
        'distributions': distributions,
        'oracle_results': oracle_results,
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
    parser = argparse.ArgumentParser(
        description='Parity CA Experiment with Pattern Holdout')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--k_rows', type=int, default=1)
    parser.add_argument('--a_fraction', type=float, default=0.6)
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--random_seeds', nargs='+', type=int, default=RANDOM_SEEDS)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    result = run_parity_ca_experiment(
        width=args.width, k_rows=args.k_rows, a_fraction=args.a_fraction,
        n_train=args.n_train, n_test=args.n_test,
        random_seeds=args.random_seeds,
        epochs=args.epochs, batch_size=args.batch_size,
    )

    elapsed = time.time() - start_time

    summary = {
        'elapsed_seconds': elapsed,
        **result,
    }
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    with open(os.path.join(OUTPUT_DIR, 'data_distributions.json'), 'w') as f:
        json.dump(result['distributions'], f, indent=2, default=str)

    # --- Print summary table ---
    _print(f'\n{"="*80}')
    _print(f'  PARITY CA EXPERIMENT SUMMARY')
    _print(f'{"="*80}')
    _print(f'  Elapsed: {elapsed:.1f}s')
    _print(f'  Width={args.width}, K={args.k_rows}, A_frac={args.a_fraction}')
    _print(f'  |A|={result["n_a"]}, |B|={result["n_b"]} (of {N_PATTERNS})')
    _print()

    ag = result['transformer_aggregate']
    bl = result['baseline_results']

    _print(f'  Teacher-forced per-cell accuracy:')
    header = f'    {"Method":<20} {"Split":<18} {"Overall":>8} {"A-nbr":>8} {"B-nbr":>8} {"A-B gap":>8}'
    _print(header)
    _print(f'    {"-"*74}')

    # Oracle
    for sname in ['id_test', 'ood_b_inclusive', 'ood_max_b']:
        o = result['oracle_results'].get(sname, {})
        if o:
            _print(f'    {"oracle":<20} {sname:<18} '
                   f'{o["overall_cell_acc"]:>7.1f}% {o["a_cell_acc"]:>7.1f}% '
                   f'{o["b_cell_acc"]:>7.1f}% {"0.0":>7}%')

    # Baselines
    for bname, bdata in bl.items():
        for sname, bres in bdata.items():
            gap = bres['a_cell_acc'] - bres['b_cell_acc'] if bres['b_total'] > 0 else float('nan')
            gap_str = f'{gap:>7.1f}%' if not np.isnan(gap) else f'{"N/A":>8}'
            b_str = f'{bres["b_cell_acc"]:>7.1f}%' if bres['b_total'] > 0 else f'{"N/A":>8}'
            _print(f'    {bname:<20} {sname:<18} '
                   f'{bres["overall_cell_acc"]:>7.1f}% {bres["a_cell_acc"]:>7.1f}% '
                   f'{b_str} {gap_str}')

    # Transformer
    for sname in ['id_test', 'ood_b_inclusive', 'ood_max_b']:
        if sname in ag:
            a = ag[sname]
            has_b = a.get('has_b_neighborhoods', False)
            b_str = f'{a["mean_b_acc"]:>7.1f}%' if has_b else f'{"N/A":>8}'
            gap_str = f'{a["mean_ab_gap"]:>7.1f}%' if has_b else f'{"N/A":>8}'
            _print(f'    {"transformer":<20} {sname:<18} '
                   f'{a["mean_overall_acc"]:>7.1f}% {a["mean_a_acc"]:>7.1f}% '
                   f'{b_str} {gap_str}')

    _print()
    _print(f'  Primary metric: A-B generalization gap')
    for sname in ['ood_b_inclusive', 'ood_max_b']:
        if sname in ag and ag[sname].get('has_b_neighborhoods'):
            _print(f'    {sname}: {ag[sname]["mean_ab_gap"]:.2f} +/- '
                   f'{ag[sname]["std_ab_gap"]:.2f}%')

    _print()
    _print(f'  Results saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
