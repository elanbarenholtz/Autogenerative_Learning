"""
Dyck Language Experiment — Balanced Parentheses as Formal Language Control

Tests whether transformers can learn stack-state tracking (a fundamentally
compositional operation) from next-token prediction on Dyck-1 and Dyck-2
languages.

Dyck-1: single bracket type ()
Dyck-2: two bracket types () and []

If the transformer fails (or matches n-gram baselines), it strengthens the
"distributional patterns only" thesis from the recurrence-relation experiments.
If it succeeds, it reveals the boundary between learnable and unlearnable structure.

Key design: right-pad + attention mask (not left-pad), standard autoregressive LM
training with loss on all non-PAD positions. This requires the src_key_padding_mask
support added to model.py.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import FibonacciTransformer, count_parameters
from experiment_framework import DATA_SEED, RANDOM_SEEDS, set_all_seeds
from exp5_baselines import NgramBaseline, KNNBaseline

OUTPUT_DIR = 'runs/dyck'


def _print(msg: str = ''):
    """Print with immediate flush."""
    print(msg, flush=True)


# ===========================================================================
# Tokenizer
# ===========================================================================
class DyckTokenizer:
    """Tokenizer for Dyck-1 and Dyck-2 languages."""

    def __init__(self, bracket_types: int = 1):
        assert bracket_types in (1, 2), "Only Dyck-1 and Dyck-2 supported"
        self.bracket_types = bracket_types

        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.OPEN_PAREN = 3
        self.CLOSE_PAREN = 4

        self.id_to_str = {0: 'PAD', 1: 'BOS', 2: 'EOS', 3: '(', 4: ')'}
        self.str_to_id = {'PAD': 0, 'BOS': 1, 'EOS': 2, '(': 3, ')': 4}

        if bracket_types == 2:
            self.OPEN_BRACKET = 5
            self.CLOSE_BRACKET = 6
            self.id_to_str[5] = '['
            self.id_to_str[6] = ']'
            self.str_to_id['['] = 5
            self.str_to_id[']'] = 6
            self.vocab_size = 7
        else:
            self.vocab_size = 5

        self.pad_id = self.PAD

    def open_ids(self) -> List[int]:
        ids = [self.OPEN_PAREN]
        if self.bracket_types == 2:
            ids.append(self.OPEN_BRACKET)
        return ids

    def close_ids(self) -> List[int]:
        ids = [self.CLOSE_PAREN]
        if self.bracket_types == 2:
            ids.append(self.CLOSE_BRACKET)
        return ids

    def matching_close(self, open_id: int) -> int:
        if open_id == self.OPEN_PAREN:
            return self.CLOSE_PAREN
        if self.bracket_types == 2 and open_id == self.OPEN_BRACKET:
            return self.CLOSE_BRACKET
        raise ValueError(f"Not an open bracket ID: {open_id}")

    def matching_open(self, close_id: int) -> int:
        if close_id == self.CLOSE_PAREN:
            return self.OPEN_PAREN
        if self.bracket_types == 2 and close_id == self.CLOSE_BRACKET:
            return self.OPEN_BRACKET
        raise ValueError(f"Not a close bracket ID: {close_id}")

    def is_open(self, token_id: int) -> bool:
        return token_id in self.open_ids()

    def is_close(self, token_id: int) -> bool:
        return token_id in self.close_ids()

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.str_to_id[t] for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id_to_str.get(i, '?') for i in ids]

    def decode_string(self, ids: List[int]) -> str:
        return ''.join(self.decode(ids))


# ===========================================================================
# Data Generation
# ===========================================================================
def generate_dyck_string(
    tokenizer: DyckTokenizer,
    n_pairs: int,
    max_depth: int,
    rng: random.Random,
) -> List[int]:
    """
    Generate a valid Dyck string using a stack-based online algorithm.

    Returns token IDs including BOS and EOS.
    """
    tokens = [tokenizer.BOS]
    stack = []
    pairs_remaining = n_pairs

    while pairs_remaining > 0 or len(stack) > 0:
        depth = len(stack)

        if depth >= max_depth or pairs_remaining == 0:
            # Must close
            tokens.append(tokenizer.matching_close(stack.pop()))
        elif depth == 0:
            # Must open (stack empty, pairs remaining)
            open_id = rng.choice(tokenizer.open_ids())
            tokens.append(open_id)
            stack.append(open_id)
            pairs_remaining -= 1
        else:
            # Choose open or close probabilistically
            p_open = pairs_remaining / (pairs_remaining + depth)
            if rng.random() < p_open:
                open_id = rng.choice(tokenizer.open_ids())
                tokens.append(open_id)
                stack.append(open_id)
                pairs_remaining -= 1
            else:
                tokens.append(tokenizer.matching_close(stack.pop()))

    tokens.append(tokenizer.EOS)
    return tokens


def generate_split(
    tokenizer: DyckTokenizer,
    n_sequences: int,
    min_pairs: int,
    max_pairs: int,
    max_depth: int,
    seed: int,
) -> List[List[int]]:
    """Generate a dataset split."""
    rng = random.Random(seed)
    sequences = []
    for _ in range(n_sequences):
        n_pairs = rng.randint(min_pairs, max_pairs)
        seq = generate_dyck_string(tokenizer, n_pairs, max_depth, rng)
        sequences.append(seq)
    return sequences


def compute_data_distributions(sequences: List[List[int]], tokenizer: DyckTokenizer) -> Dict:
    """Compute length, depth, and open/close ratio distributions."""
    lengths = []
    max_depths = []
    open_counts_by_pos = defaultdict(int)
    close_counts_by_pos = defaultdict(int)
    total_by_pos = defaultdict(int)

    for seq in sequences:
        # Length excluding BOS/EOS
        content = seq[1:-1]  # strip BOS and EOS
        lengths.append(len(content))

        # Max depth
        depth = 0
        max_d = 0
        for tok in content:
            if tokenizer.is_open(tok):
                depth += 1
                max_d = max(max_d, depth)
            elif tokenizer.is_close(tok):
                depth -= 1
        max_depths.append(max_d)

        # Per-position open/close
        for i, tok in enumerate(content):
            total_by_pos[i] += 1
            if tokenizer.is_open(tok):
                open_counts_by_pos[i] += 1
            elif tokenizer.is_close(tok):
                close_counts_by_pos[i] += 1

    # Summarize
    length_hist = {}
    for l in lengths:
        length_hist[l] = length_hist.get(l, 0) + 1

    depth_hist = {}
    for d in max_depths:
        depth_hist[d] = depth_hist.get(d, 0) + 1

    # Open ratio for first 80 positions
    open_ratio_by_pos = {}
    for pos in range(min(80, max(total_by_pos.keys()) + 1 if total_by_pos else 0)):
        if total_by_pos[pos] > 0:
            open_ratio_by_pos[pos] = open_counts_by_pos[pos] / total_by_pos[pos]

    return {
        'n_sequences': len(sequences),
        'mean_length': float(np.mean(lengths)) if lengths else 0,
        'std_length': float(np.std(lengths)) if lengths else 0,
        'min_length': int(min(lengths)) if lengths else 0,
        'max_length': int(max(lengths)) if lengths else 0,
        'mean_max_depth': float(np.mean(max_depths)) if max_depths else 0,
        'std_max_depth': float(np.std(max_depths)) if max_depths else 0,
        'length_histogram': {str(k): v for k, v in sorted(length_hist.items())},
        'depth_histogram': {str(k): v for k, v in sorted(depth_hist.items())},
        'open_ratio_by_position': {str(k): round(v, 3) for k, v in sorted(open_ratio_by_pos.items())},
    }


# ===========================================================================
# Validation
# ===========================================================================
def is_valid_dyck(
    token_ids: List[int],
    tokenizer: DyckTokenizer,
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate a Dyck string.

    Returns (is_valid, first_error_position, error_type).
    error_type is one of: 'underflow', 'wrong_type', 'unclosed', 'unexpected_token', None
    """
    stack = []
    # Expect BOS at position 0
    for i, tok in enumerate(token_ids):
        if tok == tokenizer.BOS:
            continue
        if tok == tokenizer.EOS:
            if len(stack) > 0:
                return False, i, 'unclosed'
            return True, None, None
        if tok == tokenizer.PAD:
            continue
        if tokenizer.is_open(tok):
            stack.append(tok)
        elif tokenizer.is_close(tok):
            if len(stack) == 0:
                return False, i, 'underflow'
            expected_close = tokenizer.matching_close(stack[-1])
            if tok != expected_close:
                return False, i, 'wrong_type'
            stack.pop()
        else:
            return False, i, 'unexpected_token'

    # If we never saw EOS
    if len(stack) > 0:
        return False, len(token_ids), 'unclosed'
    return True, None, None


# ===========================================================================
# Oracle
# ===========================================================================
def oracle_next_tokens(
    token_ids: List[int],
    tokenizer: DyckTokenizer,
    max_pairs_in_sequence: int = 40,
) -> List[Set[int]]:
    """
    For each position, return the set of legal next tokens given the stack state.

    This is a deterministic baseline showing what a perfect stack-tracking
    model could predict. Used for:
    - Data sanity checks
    - Oracle accuracy ceiling (fraction of forced moves)
    - Per-position legality checking of model predictions
    """
    legal_sets = []
    stack = []
    total_opens = 0
    total_closes = 0

    for i, tok in enumerate(token_ids[:-1]):  # don't need legal set after last token
        if tok == tokenizer.BOS:
            stack_state = list(stack)
        elif tokenizer.is_open(tok):
            stack.append(tok)
            total_opens += 1
        elif tokenizer.is_close(tok):
            if stack:
                stack.pop()
            total_closes += 1

        # What's legal at position i+1?
        depth = len(stack)
        pairs_remaining_estimate = max_pairs_in_sequence - total_opens

        legal = set()
        # Can always open if there's room
        if depth < 20 and total_opens < max_pairs_in_sequence:
            legal.update(tokenizer.open_ids())
        # Can close if stack is non-empty
        if depth > 0:
            legal.add(tokenizer.matching_close(stack[-1]))
        # Can end if stack is empty and at least one pair was made
        if depth == 0 and total_opens > 0:
            legal.add(tokenizer.EOS)

        legal_sets.append(legal)

    return legal_sets


def compute_oracle_stats(
    sequences: List[List[int]],
    tokenizer: DyckTokenizer,
) -> Dict:
    """Compute oracle statistics: fraction of forced moves, etc."""
    total_positions = 0
    forced_positions = 0
    ambiguous_positions = 0

    for seq in sequences:
        legal_sets = oracle_next_tokens(seq, tokenizer, max_pairs_in_sequence=40)
        for legal_set in legal_sets:
            total_positions += 1
            if len(legal_set) == 1:
                forced_positions += 1
            else:
                ambiguous_positions += 1

    return {
        'total_positions': total_positions,
        'forced_positions': forced_positions,
        'ambiguous_positions': ambiguous_positions,
        'forced_fraction': forced_positions / total_positions if total_positions > 0 else 0,
    }


# ===========================================================================
# LM Dataset
# ===========================================================================
class DyckLMDataset(Dataset):
    """Dataset for autoregressive LM training on Dyck sequences."""

    def __init__(self, sequences: List[List[int]]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)


def collate_right_pad(batch: List[torch.Tensor], pad_id: int = 0):
    """Right-pad sequences to batch-max length. Returns (input, target, padding_mask)."""
    max_len = max(t.size(0) for t in batch)
    inputs = []
    targets = []
    masks = []

    for seq in batch:
        seq_len = seq.size(0)
        # Input: all but last token
        inp = seq[:-1]
        tgt = seq[1:]
        pad_len = max_len - seq_len

        if pad_len > 0:
            inp = torch.cat([inp, torch.full((pad_len,), pad_id, dtype=torch.long)])
            tgt = torch.cat([tgt, torch.full((pad_len,), pad_id, dtype=torch.long)])

        # Padding mask: True where PAD (for input positions)
        mask = (inp == pad_id)

        inputs.append(inp)
        targets.append(tgt)
        masks.append(mask)

    return torch.stack(inputs), torch.stack(targets), torch.stack(masks)


# ===========================================================================
# Training
# ===========================================================================
def train_lm(
    train_sequences: List[List[int]],
    tokenizer: DyckTokenizer,
    random_seed: int,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 3,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    max_seq_len: int = 256,
    device: str = None,
    verbose: bool = True,
) -> Tuple[FibonacciTransformer, Dict]:
    """
    Train an autoregressive LM on Dyck sequences.

    Uses right-padding with attention mask and loss on all non-PAD positions.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_all_seeds(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    dataset = DyckLMDataset(train_sequences)
    pad_id = tokenizer.PAD
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_right_pad(b, pad_id),
    )

    model = FibonacciTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_seq_len,
    ).to(device)

    if verbose:
        _print(f"    Model parameters: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

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

            # Flatten for cross-entropy, ignore PAD positions
            logits_flat = logits.reshape(-1, tokenizer.vocab_size)
            targets_flat = targets.reshape(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_id)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy on non-PAD positions
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
            }, os.path.join(output_dir, 'best_model.pt'))

        if verbose and (epoch + 1) % 10 == 0:
            _print(f"      Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={accuracy:.2f}%")

    if verbose:
        _print(f"      Training complete. Best loss={best_loss:.4f}, Final acc={accuracy:.2f}%")

    # Load best model
    ckpt = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, history


# ===========================================================================
# Evaluation: Teacher-Forced
# ===========================================================================
def evaluate_teacher_forced_dyck(
    model: FibonacciTransformer,
    sequences: List[List[int]],
    tokenizer: DyckTokenizer,
    device: str = None,
) -> Dict:
    """
    Teacher-forced evaluation on Dyck sequences.

    Returns per-token accuracy, exact match rate, per-depth accuracy,
    open vs close accuracy, illegal prediction rate, and error type breakdown.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()

    total_correct = 0
    total_tokens = 0
    exact_matches = 0
    first_errors = []

    # Per-depth tracking
    depth_correct = defaultdict(int)
    depth_total = defaultdict(int)

    # Open vs close accuracy
    open_correct = 0
    open_total = 0
    close_correct = 0
    close_total = 0

    # Illegal prediction tracking
    illegal_predictions = 0
    underflow_predictions = 0
    wrong_type_predictions = 0

    with torch.no_grad():
        for seq in sequences:
            seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
            inp = seq_tensor[:, :-1]
            tgt = seq_tensor[:, 1:]

            logits = model(inp)
            preds = logits.argmax(dim=-1).squeeze(0)  # (seq_len-1,)
            targets = tgt.squeeze(0)

            # Track depth via stack simulation
            stack = []
            seq_correct = True
            first_err = None

            for pos in range(targets.size(0)):
                target_tok = targets[pos].item()
                pred_tok = preds[pos].item()

                if target_tok == tokenizer.PAD:
                    continue

                # Current depth (before processing this target)
                depth = len(stack)

                total_tokens += 1
                depth_total[depth] += 1

                if pred_tok == target_tok:
                    total_correct += 1
                    depth_correct[depth] += 1
                else:
                    if seq_correct:
                        first_err = pos
                        seq_correct = False

                # Track open/close accuracy
                if tokenizer.is_open(target_tok):
                    open_total += 1
                    if pred_tok == target_tok:
                        open_correct += 1
                elif tokenizer.is_close(target_tok):
                    close_total += 1
                    if pred_tok == target_tok:
                        close_correct += 1

                # Check if prediction is illegal
                if tokenizer.is_close(pred_tok):
                    if depth == 0:
                        illegal_predictions += 1
                        underflow_predictions += 1
                    elif tokenizer.matching_close(stack[-1]) != pred_tok:
                        illegal_predictions += 1
                        wrong_type_predictions += 1

                # Update stack with TRUE target (teacher forcing)
                if tokenizer.is_open(target_tok):
                    stack.append(target_tok)
                elif tokenizer.is_close(target_tok):
                    if stack:
                        stack.pop()

            if seq_correct:
                exact_matches += 1
            if first_err is not None:
                first_errors.append(first_err)

    n_sequences = len(sequences)
    results = {
        'per_token_accuracy': 100.0 * total_correct / total_tokens if total_tokens > 0 else 0,
        'exact_match_rate': 100.0 * exact_matches / n_sequences if n_sequences > 0 else 0,
        'mean_first_error': float(np.mean(first_errors)) if first_errors else None,
        'per_depth_accuracy': {
            d: 100.0 * depth_correct[d] / depth_total[d]
            for d in sorted(depth_total.keys())
        },
        'open_accuracy': 100.0 * open_correct / open_total if open_total > 0 else 0,
        'close_accuracy': 100.0 * close_correct / close_total if close_total > 0 else 0,
        'illegal_prediction_rate': 100.0 * illegal_predictions / total_tokens if total_tokens > 0 else 0,
        'underflow_rate': 100.0 * underflow_predictions / total_tokens if total_tokens > 0 else 0,
        'wrong_type_rate': 100.0 * wrong_type_predictions / total_tokens if total_tokens > 0 else 0,
        'total_tokens': total_tokens,
        'n_sequences': n_sequences,
    }

    return results


# ===========================================================================
# Evaluation: Free-Run
# ===========================================================================
def evaluate_free_run(
    model: FibonacciTransformer,
    tokenizer: DyckTokenizer,
    n_samples: int = 200,
    max_len: int = 100,
    device: str = None,
    temperature: float = 0.0,
) -> Dict:
    """
    Free-run generation evaluation.

    Starts from BOS, generates autoregressively until EOS or max_len.
    Returns validity rate, mean length, error analysis, and samples.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()

    valid_count = 0
    lengths = []
    error_types = defaultdict(int)
    first_errors = []
    samples = []

    with torch.no_grad():
        for i in range(n_samples):
            generated = [tokenizer.BOS]

            for step in range(max_len):
                inp = torch.tensor([generated], dtype=torch.long, device=device)
                logits = model(inp)
                next_logits = logits[0, -1, :]

                if temperature == 0:
                    next_tok = next_logits.argmax().item()
                else:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, 1).item()

                generated.append(next_tok)

                if next_tok == tokenizer.EOS:
                    break

            # Validate
            valid, err_pos, err_type = is_valid_dyck(generated, tokenizer)
            if valid:
                valid_count += 1
            else:
                if err_type:
                    error_types[err_type] += 1
                if err_pos is not None:
                    first_errors.append(err_pos)

            content_len = len(generated) - 2  # exclude BOS/EOS
            lengths.append(content_len)

            if i < 20:
                samples.append({
                    'ids': generated,
                    'string': tokenizer.decode_string(generated),
                    'valid': valid,
                    'error_type': err_type,
                    'error_pos': err_pos,
                })

    results = {
        'validity_rate': 100.0 * valid_count / n_samples if n_samples > 0 else 0,
        'mean_length': float(np.mean(lengths)) if lengths else 0,
        'std_length': float(np.std(lengths)) if lengths else 0,
        'error_type_counts': dict(error_types),
        'mean_first_error': float(np.mean(first_errors)) if first_errors else None,
        'n_samples': n_samples,
        'temperature': temperature,
        'samples': samples,
    }

    return results


# ===========================================================================
# Baselines
# ===========================================================================
class BigramBaseline:
    """Trivial bigram baseline: P(next | last_token)."""

    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def train(self, examples: List[Tuple[List[int], int]]):
        for ctx, tgt in examples:
            last_tok = ctx[-1] if ctx else 0
            self.counts[last_tok][tgt] += 1
            self.vocab.add(tgt)

    def predict(self, context: List[int]) -> int:
        last_tok = context[-1] if context else 0
        if last_tok in self.counts and self.counts[last_tok]:
            return max(self.counts[last_tok], key=self.counts[last_tok].get)
        if self.vocab:
            return next(iter(self.vocab))
        return 0


def sequences_to_ngram_examples(
    sequences: List[List[int]],
    context_window: int = 10,
) -> List[Tuple[List[int], int]]:
    """Convert sequences to (context, target) pairs for baseline training."""
    examples = []
    for seq in sequences:
        for i in range(1, len(seq)):
            ctx_start = max(0, i - context_window)
            ctx = seq[ctx_start:i]
            tgt = seq[i]
            examples.append((ctx, tgt))
    return examples


def evaluate_baseline_dyck(
    baseline,
    sequences: List[List[int]],
    tokenizer: DyckTokenizer,
    context_window: int = 10,
) -> Dict:
    """Evaluate a baseline model on Dyck sequences (teacher-forced)."""
    total_correct = 0
    total_tokens = 0
    exact_matches = 0

    for seq in sequences:
        seq_correct = True
        for i in range(1, len(seq)):
            ctx_start = max(0, i - context_window)
            ctx = seq[ctx_start:i]
            target = seq[i]

            if target == tokenizer.PAD:
                continue

            pred = baseline.predict(ctx)
            total_tokens += 1
            if pred == target:
                total_correct += 1
            else:
                seq_correct = False

        if seq_correct:
            exact_matches += 1

    return {
        'per_token_accuracy': 100.0 * total_correct / total_tokens if total_tokens > 0 else 0,
        'exact_match_rate': 100.0 * exact_matches / len(sequences) if sequences else 0,
        'total_tokens': total_tokens,
    }


# ===========================================================================
# Orchestration
# ===========================================================================
def run_dyck_experiment(
    bracket_types: int = 1,
    n_train: int = 10000,
    n_test: int = 2000,
    random_seeds: List[int] = None,
    pilot: bool = False,
    device: str = None,
) -> Dict:
    """Run the full Dyck experiment for one language (D1 or D2)."""
    if random_seeds is None:
        random_seeds = RANDOM_SEEDS
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    label = f"D{bracket_types}"
    _print(f"\n{'='*70}")
    _print(f"  Dyck-{bracket_types} Experiment")
    _print(f"{'='*70}")

    tokenizer = DyckTokenizer(bracket_types=bracket_types)
    _print(f"  Vocab size: {tokenizer.vocab_size}")
    _print(f"  Tokens: {tokenizer.id_to_str}")

    if pilot:
        n_train = 2000
        n_test = 500

    # --- Generate data ---
    _print(f"\n  Generating data (train={n_train}, test={n_test} per split)...")

    # Use sub-seeds derived from DATA_SEED for each split
    train_seqs = generate_split(tokenizer, n_train, 1, 20, 6, seed=DATA_SEED)
    id_test_seqs = generate_split(tokenizer, n_test, 1, 20, 6, seed=DATA_SEED + 1)
    longer_ood_seqs = generate_split(tokenizer, n_test, 30, 40, 14, seed=DATA_SEED + 2)
    deeper_ood_seqs = generate_split(tokenizer, n_test, 1, 20, 14, seed=DATA_SEED + 3)
    combined_ood_seqs = generate_split(tokenizer, n_test, 30, 40, 14, seed=DATA_SEED + 4)

    splits = {
        'train': train_seqs,
        'id_test': id_test_seqs,
        'longer_ood': longer_ood_seqs,
        'deeper_ood': deeper_ood_seqs,
        'combined_ood': combined_ood_seqs,
    }

    # --- Validate all data ---
    _print("  Validating all generated sequences...")
    for split_name, seqs in splits.items():
        invalid = 0
        for seq in seqs:
            valid, _, _ = is_valid_dyck(seq, tokenizer)
            if not valid:
                invalid += 1
        assert invalid == 0, f"FATAL: {invalid} invalid sequences in {split_name}"
        _print(f"    {split_name}: {len(seqs)} sequences, all valid")

    # --- Data distributions ---
    _print("  Computing data distributions...")
    distributions = {}
    for split_name, seqs in splits.items():
        distributions[split_name] = compute_data_distributions(seqs, tokenizer)
        d = distributions[split_name]
        _print(f"    {split_name}: mean_len={d['mean_length']:.1f}, "
               f"mean_depth={d['mean_max_depth']:.1f}")

    # --- Oracle stats ---
    _print("  Computing oracle statistics...")
    oracle_stats = {}
    for split_name, seqs in splits.items():
        oracle_stats[split_name] = compute_oracle_stats(seqs, tokenizer)
        o = oracle_stats[split_name]
        _print(f"    {split_name}: forced={o['forced_fraction']:.3f} "
               f"({o['forced_positions']}/{o['total_positions']})")

    # --- Baselines ---
    _print("\n  Training baselines...")
    ctx_win = 10
    train_examples = sequences_to_ngram_examples(train_seqs, context_window=ctx_win)

    baseline_results = {}

    # Bigram (fast — evaluate on all splits)
    bigram = BigramBaseline()
    bigram.train(train_examples)
    for split_name, seqs in splits.items():
        if split_name == 'train':
            continue
        res = evaluate_baseline_dyck(bigram, seqs, tokenizer, context_window=ctx_win)
        baseline_results.setdefault('bigram', {})[split_name] = res
    _print(f"    Bigram ID: {baseline_results['bigram']['id_test']['per_token_accuracy']:.2f}%")

    # N-gram (3 and 5) — fast, evaluate on all splits
    for n in [3, 5]:
        ngram = NgramBaseline(n=n)
        ngram.train(train_examples)
        for split_name, seqs in splits.items():
            if split_name == 'train':
                continue
            res = evaluate_baseline_dyck(ngram, seqs, tokenizer, context_window=ctx_win)
            baseline_results.setdefault(f'ngram_{n}', {})[split_name] = res
        _print(f"    N-gram({n}) ID: {baseline_results[f'ngram_{n}']['id_test']['per_token_accuracy']:.2f}%")

    # kNN at multiple context lengths — slow O(n*m), subsample for efficiency
    knn_eval_max = 100  # max sequences per split for kNN eval
    knn_train_max = 5000  # max training examples for kNN
    for knn_ctx in [10, 20, 40]:
        knn_examples = sequences_to_ngram_examples(train_seqs, context_window=knn_ctx)
        if len(knn_examples) > knn_train_max:
            rng_knn = random.Random(DATA_SEED)
            knn_examples = rng_knn.sample(knn_examples, knn_train_max)
        knn = KNNBaseline(k=5, context_window=knn_ctx)
        knn.train(knn_examples)
        for split_name, seqs in splits.items():
            if split_name == 'train':
                continue
            eval_seqs = seqs[:knn_eval_max]
            res = evaluate_baseline_dyck(knn, eval_seqs, tokenizer, context_window=knn_ctx)
            baseline_results.setdefault(f'knn_ctx{knn_ctx}', {})[split_name] = res
            _print(f"    kNN(ctx={knn_ctx}) {split_name}: "
                   f"{res['per_token_accuracy']:.2f}% (n={len(eval_seqs)})")
        _print(f"    kNN(ctx={knn_ctx}) ID: "
               f"{baseline_results[f'knn_ctx{knn_ctx}']['id_test']['per_token_accuracy']:.2f}%")

    # --- Transformer training ---
    _print(f"\n  Training transformers (seeds={random_seeds})...")
    transformer_results = {}

    for rs in random_seeds:
        run_dir = os.path.join(OUTPUT_DIR, f'{label}_RS{rs}')
        _print(f"\n    --- RS={rs} ---")

        model, history = train_lm(
            train_sequences=train_seqs,
            tokenizer=tokenizer,
            random_seed=rs,
            output_dir=run_dir,
            epochs=50,
            batch_size=32,
            lr=0.001,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            max_seq_len=256,
            device=device,
            verbose=True,
        )

        # Teacher-forced eval on all splits
        seed_results = {'history': history}
        for split_name, seqs in splits.items():
            if split_name == 'train':
                continue
            tf_res = evaluate_teacher_forced_dyck(model, seqs, tokenizer, device)
            seed_results[f'teacher_forced_{split_name}'] = tf_res
            _print(f"      TF {split_name}: token_acc={tf_res['per_token_accuracy']:.2f}%, "
                   f"exact={tf_res['exact_match_rate']:.2f}%, "
                   f"illegal={tf_res['illegal_prediction_rate']:.2f}%")

        # Free-run eval (greedy)
        fr_greedy = evaluate_free_run(model, tokenizer, n_samples=200, max_len=100,
                                       device=device, temperature=0.0)
        seed_results['free_run_greedy'] = fr_greedy
        _print(f"      Free-run greedy: validity={fr_greedy['validity_rate']:.2f}%, "
               f"mean_len={fr_greedy['mean_length']:.1f}")

        # Free-run eval (sampled)
        fr_sampled = evaluate_free_run(model, tokenizer, n_samples=200, max_len=100,
                                        device=device, temperature=0.8)
        seed_results['free_run_sampled'] = fr_sampled
        _print(f"      Free-run sampled: validity={fr_sampled['validity_rate']:.2f}%, "
               f"mean_len={fr_sampled['mean_length']:.1f}")

        # Save manifest
        manifest = {
            'label': label,
            'bracket_types': bracket_types,
            'random_seed': rs,
            'vocab_size': tokenizer.vocab_size,
            'n_train': len(train_seqs),
            'results': seed_results,
        }
        with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

        transformer_results[rs] = seed_results

    # --- Aggregate across seeds ---
    _print(f"\n  Aggregating transformer results across seeds...")
    agg_transformer = {}
    for split_name in ['id_test', 'longer_ood', 'deeper_ood', 'combined_ood']:
        tf_key = f'teacher_forced_{split_name}'
        accs = [transformer_results[rs][tf_key]['per_token_accuracy'] for rs in random_seeds]
        exact = [transformer_results[rs][tf_key]['exact_match_rate'] for rs in random_seeds]
        illegal = [transformer_results[rs][tf_key]['illegal_prediction_rate'] for rs in random_seeds]
        underflow = [transformer_results[rs][tf_key]['underflow_rate'] for rs in random_seeds]
        wrong_type = [transformer_results[rs][tf_key]['wrong_type_rate'] for rs in random_seeds]
        agg_transformer[split_name] = {
            'mean_token_acc': float(np.mean(accs)),
            'std_token_acc': float(np.std(accs)),
            'mean_exact_match': float(np.mean(exact)),
            'std_exact_match': float(np.std(exact)),
            'mean_illegal_rate': float(np.mean(illegal)),
            'std_illegal_rate': float(np.std(illegal)),
            'mean_underflow_rate': float(np.mean(underflow)),
            'std_underflow_rate': float(np.std(underflow)),
            'mean_wrong_type_rate': float(np.mean(wrong_type)),
            'std_wrong_type_rate': float(np.std(wrong_type)),
            'per_seed_token_acc': {str(rs): a for rs, a in zip(random_seeds, accs)},
        }
        _print(f"    {split_name}: token_acc={agg_transformer[split_name]['mean_token_acc']:.2f} "
               f"± {agg_transformer[split_name]['std_token_acc']:.2f}%, "
               f"illegal={agg_transformer[split_name]['mean_illegal_rate']:.2f}%")

    # Free-run aggregation
    greedy_validities = [transformer_results[rs]['free_run_greedy']['validity_rate']
                         for rs in random_seeds]
    sampled_validities = [transformer_results[rs]['free_run_sampled']['validity_rate']
                          for rs in random_seeds]
    agg_transformer['free_run_greedy'] = {
        'mean_validity': float(np.mean(greedy_validities)),
        'std_validity': float(np.std(greedy_validities)),
    }
    agg_transformer['free_run_sampled'] = {
        'mean_validity': float(np.mean(sampled_validities)),
        'std_validity': float(np.std(sampled_validities)),
    }

    # --- Compile full results ---
    experiment_result = {
        'label': label,
        'bracket_types': bracket_types,
        'n_train': len(train_seqs),
        'n_test_per_split': n_test,
        'pilot': pilot,
        'distributions': distributions,
        'oracle_stats': oracle_stats,
        'baseline_results': baseline_results,
        'transformer_aggregate': agg_transformer,
        'transformer_per_seed': {
            str(rs): {k: v for k, v in transformer_results[rs].items() if k != 'history'}
            for rs in random_seeds
        },
    }

    # --- Save qualitative samples ---
    samples_file = os.path.join(OUTPUT_DIR, f'qualitative_samples_{label}.txt')
    with open(samples_file, 'w') as f:
        f.write(f"Qualitative Samples: {label}\n")
        f.write(f"{'='*60}\n\n")
        for rs in random_seeds:
            f.write(f"--- RS={rs} (Greedy) ---\n")
            for s in transformer_results[rs]['free_run_greedy']['samples']:
                status = "VALID" if s['valid'] else f"INVALID ({s['error_type']} @ {s['error_pos']})"
                f.write(f"  {s['string']}  [{status}]\n")
            f.write(f"\n--- RS={rs} (Sampled T=0.8) ---\n")
            for s in transformer_results[rs]['free_run_sampled']['samples']:
                status = "VALID" if s['valid'] else f"INVALID ({s['error_type']} @ {s['error_pos']})"
                f.write(f"  {s['string']}  [{status}]\n")
            f.write("\n")
    _print(f"  Saved qualitative samples to {samples_file}")

    return experiment_result


def main():
    parser = argparse.ArgumentParser(description='Dyck Language Experiment')
    parser.add_argument('--pilot', action='store_true',
                        help='Run pilot (2000 train / 500 test)')
    parser.add_argument('--dyck1-only', action='store_true',
                        help='Only run Dyck-1')
    parser.add_argument('--dyck2-only', action='store_true',
                        help='Only run Dyck-2')
    parser.add_argument('--random_seeds', nargs='+', type=int, default=RANDOM_SEEDS,
                        help='Random seeds for model init')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}
    start_time = time.time()

    if not args.dyck2_only:
        d1_result = run_dyck_experiment(
            bracket_types=1,
            random_seeds=args.random_seeds,
            pilot=args.pilot,
        )
        all_results['D1'] = d1_result

    if not args.dyck1_only:
        d2_result = run_dyck_experiment(
            bracket_types=2,
            random_seeds=args.random_seeds,
            pilot=args.pilot,
        )
        all_results['D2'] = d2_result

    elapsed = time.time() - start_time

    # --- Save combined summary ---
    summary = {
        'elapsed_seconds': elapsed,
        'pilot': args.pilot,
        'random_seeds': args.random_seeds,
        'results': all_results,
    }
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save data distributions
    dist_data = {}
    for key, result in all_results.items():
        dist_data[key] = result.get('distributions', {})
    with open(os.path.join(OUTPUT_DIR, 'data_distributions.json'), 'w') as f:
        json.dump(dist_data, f, indent=2, default=str)

    # --- Print summary table ---
    _print(f"\n{'='*80}")
    _print(f"  DYCK EXPERIMENT SUMMARY")
    _print(f"{'='*80}")
    _print(f"  Elapsed: {elapsed:.1f}s")
    _print(f"  Pilot: {args.pilot}")
    _print()

    for key, result in all_results.items():
        _print(f"  --- {key} ---")
        agg = result['transformer_aggregate']
        baselines = result['baseline_results']

        _print(f"  {'Method':<20} {'ID Test':>10} {'Longer OOD':>12} "
               f"{'Deeper OOD':>12} {'Combined OOD':>14}")
        _print(f"  {'-'*70}")

        # Baselines
        for bname, bdata in baselines.items():
            row = f"  {bname:<20}"
            for split in ['id_test', 'longer_ood', 'deeper_ood', 'combined_ood']:
                if split in bdata:
                    row += f" {bdata[split]['per_token_accuracy']:>10.2f}%"
                else:
                    row += f" {'N/A':>10}"
            _print(row)

        # Transformer
        row = f"  {'transformer':<20}"
        for split in ['id_test', 'longer_ood', 'deeper_ood', 'combined_ood']:
            if split in agg:
                row += f" {agg[split]['mean_token_acc']:>10.2f}%"
            else:
                row += f" {'N/A':>10}"
        _print(row)

        # Illegal rate
        row = f"  {'illegal rate':<20}"
        for split in ['id_test', 'longer_ood', 'deeper_ood', 'combined_ood']:
            if split in agg:
                row += f" {agg[split]['mean_illegal_rate']:>10.2f}%"
            else:
                row += f" {'N/A':>10}"
        _print(row)

        # Free-run
        if 'free_run_greedy' in agg:
            _print(f"  Free-run greedy validity: {agg['free_run_greedy']['mean_validity']:.2f}%")
        if 'free_run_sampled' in agg:
            _print(f"  Free-run sampled validity: {agg['free_run_sampled']['mean_validity']:.2f}%")
        _print()

    _print(f"  Results saved to {OUTPUT_DIR}/")
    _print(f"  summary.json, data_distributions.json, qualitative_samples_*.txt")


if __name__ == '__main__':
    main()
