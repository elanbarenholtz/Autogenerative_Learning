"""
Shared infrastructure for the 5-experiment research framework.

Provides: seeding, naming, training, evaluation, metrics, logging, and data generation.
"""
import os
import json
import csv
import hashlib
import subprocess
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import FibonacciTransformer, NumberTokenizer, count_parameters
from recurrence_relations import RecurrenceRelation

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
DATA_SEED = 0          # Fixed seed for ALL data generation (seeds, splits)
RANDOM_SEEDS = [42, 123, 7]  # 3 model-init seeds; only model weights vary


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
def make_run_id(rel: str, rep: str, exp: str, cfg: str, rs: int) -> str:
    """Create a canonical run ID string."""
    return f"REL={rel}__REP={rep}__EXP={exp}__CFG={cfg}__RS={rs}"


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    vocab_size: int
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    batch_size: int = 32
    lr: float = 0.001
    epochs: int = 50
    context_window: int = 10
    max_seq_len: int = 50


# ---------------------------------------------------------------------------
# Training Dataset (left-padded)
# ---------------------------------------------------------------------------
class PaddedDataset(Dataset):
    """Dataset that stores pre-encoded (context_ids, target_id) pairs."""
    def __init__(self, examples: List[Tuple[List[int], int]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ctx_ids, tgt_id = self.examples[idx]
        return torch.tensor(ctx_ids, dtype=torch.long), torch.tensor(tgt_id, dtype=torch.long)


def _left_pad_collate(batch, pad_id: int, context_window: int):
    """Collate with left-padding to context_window length."""
    contexts, targets = zip(*batch)
    padded = []
    for ctx in contexts:
        seq_len = ctx.size(0)
        if seq_len < context_window:
            pad = torch.full((context_window - seq_len,), pad_id, dtype=torch.long)
            ctx = torch.cat([pad, ctx])
        elif seq_len > context_window:
            ctx = ctx[-context_window:]
        padded.append(ctx)
    return torch.stack(padded), torch.stack(list(targets))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    train_examples: List[Tuple[List[int], int]],
    tokenizer,
    config: TrainConfig,
    random_seed: int,
    output_dir: str,
    device: str = None,
    verbose: bool = True,
) -> Tuple[FibonacciTransformer, Dict]:
    """
    Train a FibonacciTransformer on (context, target) pairs.

    Left-pads all contexts to config.context_window so PAD tokens are in the
    training distribution.

    Returns (model, history_dict).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_all_seeds(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    # Encode examples
    encoded = []
    for ctx, tgt in train_examples:
        ctx_ids = tokenizer.encode(ctx) if not isinstance(ctx[0], (torch.Tensor,)) else ctx
        tgt_id = tokenizer.encode([tgt])[0] if not isinstance(tgt, int) or isinstance(tgt, bool) else tokenizer.encode([tgt])[0]
        encoded.append((ctx_ids, tgt_id))

    dataset = PaddedDataset(encoded)
    pad_id = tokenizer.pad_id
    ctx_win = config.context_window

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: _left_pad_collate(b, pad_id, ctx_win),
    )

    model = FibonacciTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    history = {'losses': [], 'accuracies': []}

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for contexts, targets in loader:
            contexts = contexts.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(contexts)
            logits_last = logits[:, -1, :]
            loss = criterion(logits_last, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits_last.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total
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
                'config': asdict(config),
            }, os.path.join(output_dir, 'best_model.pt'))

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config.epochs}: loss={avg_loss:.4f}, acc={accuracy:.2f}%")

    if verbose:
        print(f"  Training complete. Best loss={best_loss:.4f}, Final acc={accuracy:.2f}%")

    return model, history


def load_trained_model(output_dir: str, config: TrainConfig, device: str = None) -> FibonacciTransformer:
    """Load a trained model from output_dir/best_model.pt."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
    model = FibonacciTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=0.0,
        max_seq_len=config.max_seq_len,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    seed: tuple
    accuracy: float
    per_position_correct: List[bool]
    first_error_index: Optional[int]
    predictions: List[int]
    targets: List[int]


def evaluate_teacher_forced(
    model: FibonacciTransformer,
    tokenizer,
    relation: RecurrenceRelation,
    seeds: List[tuple],
    seq_len: int,
    ctx_win: int,
    device: str = None,
) -> List[EvalResult]:
    """
    Teacher-forced evaluation: at each position, predict next token from true context.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    results = []
    pad_id = tokenizer.pad_id

    with torch.no_grad():
        for seed in seeds:
            seq = relation.generate_sequence(seed, seq_len)
            preds = []
            targets = []
            per_pos = []

            for i in range(ctx_win, len(seq)):
                context = seq[max(0, i - ctx_win):i]
                target = seq[i]

                ctx_ids = tokenizer.encode(context)
                # Left-pad
                if len(ctx_ids) < ctx_win:
                    ctx_ids = [pad_id] * (ctx_win - len(ctx_ids)) + ctx_ids

                src = torch.tensor([ctx_ids], dtype=torch.long, device=device)
                logits = model(src)
                pred_id = logits[0, -1, :].argmax().item()
                pred_num = tokenizer.id_to_token.get(pred_id, -999)

                preds.append(pred_num)
                targets.append(target)
                per_pos.append(pred_num == target)

            n_correct = sum(per_pos)
            accuracy = 100.0 * n_correct / len(per_pos) if per_pos else 0.0
            first_err = None
            for idx, c in enumerate(per_pos):
                if not c:
                    first_err = idx
                    break

            results.append(EvalResult(
                seed=seed,
                accuracy=accuracy,
                per_position_correct=per_pos,
                first_error_index=first_err,
                predictions=preds,
                targets=targets,
            ))

    return results


def evaluate_autoregressive(
    model: FibonacciTransformer,
    tokenizer,
    relation: RecurrenceRelation,
    seeds: List[tuple],
    seq_len: int,
    prefix_len: int,
    ctx_win: int,
    device: str = None,
) -> List[EvalResult]:
    """
    Free-running evaluation: give prefix_len true tokens, then model feeds its own predictions.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    results = []
    pad_id = tokenizer.pad_id

    with torch.no_grad():
        for seed in seeds:
            seq = relation.generate_sequence(seed, seq_len)
            if prefix_len >= len(seq):
                # Not enough sequence to evaluate
                results.append(EvalResult(
                    seed=seed, accuracy=0.0, per_position_correct=[],
                    first_error_index=None, predictions=[], targets=[],
                ))
                continue

            # Start with true prefix
            generated = list(seq[:prefix_len])
            preds = []
            targets = []
            per_pos = []

            for i in range(prefix_len, len(seq)):
                context = generated[max(0, len(generated) - ctx_win):]
                target = seq[i]

                ctx_ids = tokenizer.encode(context)
                if len(ctx_ids) < ctx_win:
                    ctx_ids = [pad_id] * (ctx_win - len(ctx_ids)) + ctx_ids

                src = torch.tensor([ctx_ids], dtype=torch.long, device=device)
                logits = model(src)
                pred_id = logits[0, -1, :].argmax().item()
                pred_num = tokenizer.id_to_token.get(pred_id, -999)

                generated.append(pred_num)
                preds.append(pred_num)
                targets.append(target)
                per_pos.append(pred_num == target)

            n_correct = sum(per_pos)
            accuracy = 100.0 * n_correct / len(per_pos) if per_pos else 0.0
            first_err = None
            for idx, c in enumerate(per_pos):
                if not c:
                    first_err = idx
                    break

            results.append(EvalResult(
                seed=seed,
                accuracy=accuracy,
                per_position_correct=per_pos,
                first_error_index=first_err,
                predictions=preds,
                targets=targets,
            ))

    return results


# ---------------------------------------------------------------------------
# Metrics & Aggregation
# ---------------------------------------------------------------------------
def aggregate_results(results: List[EvalResult]) -> Dict:
    """Aggregate evaluation results across seeds."""
    accs = [r.accuracy for r in results]
    first_errs = [r.first_error_index for r in results if r.first_error_index is not None]
    return {
        'mean_accuracy': float(np.mean(accs)) if accs else 0.0,
        'std_accuracy': float(np.std(accs)) if accs else 0.0,
        'per_seed_accuracy': {str(r.seed): r.accuracy for r in results},
        'first_error_indices': first_errs,
        'n_seeds': len(results),
    }


def compute_confusion_matrix(results: List[EvalResult], vocab_size: int) -> np.ndarray:
    """Compute confusion matrix from evaluation results."""
    cm = np.zeros((vocab_size, vocab_size), dtype=int)
    for r in results:
        for pred, tgt in zip(r.predictions, r.targets):
            if 0 <= pred < vocab_size and 0 <= tgt < vocab_size:
                cm[tgt, pred] += 1
    return cm


# ---------------------------------------------------------------------------
# Manifest & Logging
# ---------------------------------------------------------------------------
def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def _dataset_hash(train_seeds, test_seeds, config_dict) -> str:
    data = json.dumps({
        'train_seeds': [list(s) for s in train_seeds],
        'test_seeds': [list(s) for s in test_seeds],
        'config': config_dict,
    }, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


def save_run(
    run_id: str,
    output_dir: str,
    config: TrainConfig,
    train_seeds: List[tuple],
    test_seeds: List[tuple],
    train_history: Dict,
    eval_results: List[EvalResult],
    extra_meta: Dict = None,
):
    """Save run manifest and append to all_runs.csv."""
    os.makedirs(output_dir, exist_ok=True)
    agg = aggregate_results(eval_results)
    config_dict = asdict(config)

    manifest = {
        'run_id': run_id,
        'train_seeds': [list(s) for s in train_seeds],
        'test_seeds': [list(s) for s in test_seeds],
        'dataset_hash': _dataset_hash(train_seeds, test_seeds, config_dict),
        'git_commit': _get_git_commit(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'total_training_examples': len(train_history.get('losses', [])) and None,  # set externally
        'config': config_dict,
        'train_history': train_history,
        'eval_summary': agg,
    }
    if extra_meta:
        manifest.update(extra_meta)

    # Compute actual training example count from train_seeds if available
    manifest['total_training_examples'] = extra_meta.get('total_training_examples') if extra_meta else None

    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    # Append to all_runs.csv
    csv_path = os.path.join(output_dir, 'all_runs.csv')
    file_exists = os.path.exists(csv_path)
    row = {
        'run_id': run_id,
        'mean_accuracy': agg['mean_accuracy'],
        'std_accuracy': agg['std_accuracy'],
        'n_seeds': agg['n_seeds'],
        'best_train_loss': min(train_history.get('losses', [float('inf')])),
        'final_train_acc': train_history.get('accuracies', [0.0])[-1] if train_history.get('accuracies') else 0.0,
    }
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return manifest


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------
def generate_examples(
    relation: RecurrenceRelation,
    seeds: List[tuple],
    seq_len: int,
    ctx_win: int,
    max_value: int = 100000,
) -> Tuple[List[Tuple[List[int], int]], List[int]]:
    """
    Generate (context, target) examples and vocabulary from seeds.

    Returns (examples_list, sorted_vocabulary).
    """
    all_examples = []
    all_numbers = set()

    for seed in seeds:
        sequence = relation.generate_sequence(seed, seq_len)

        # Truncate if values exceed max_value
        valid_len = len(sequence)
        for i, val in enumerate(sequence):
            if abs(val) > max_value:
                valid_len = i
                break
        sequence = sequence[:valid_len]

        # Sliding window
        for i in range(ctx_win, len(sequence)):
            context = sequence[i - ctx_win:i]
            target = sequence[i]
            all_examples.append((context, target))

        all_numbers.update(sequence)

    return all_examples, sorted(list(all_numbers))


# ---------------------------------------------------------------------------
# Multi-run helper
# ---------------------------------------------------------------------------
def multi_run(run_fn: Callable[[int], Dict], random_seeds: List[int] = None) -> List[Dict]:
    """Call run_fn(rs) for each random seed, collect results."""
    if random_seeds is None:
        random_seeds = RANDOM_SEEDS
    results = []
    for rs in random_seeds:
        results.append(run_fn(rs))
    return results
