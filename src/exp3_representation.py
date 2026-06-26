"""
Experiment 3 — Representation / Tokenization Control

Goal: Eliminate "tokenization did this" objection.

Three representations:
  (A) Digit  — DigitTokenizer, multi-token per number
  (B) Integer — NumberTokenizer, one token per number
  (C) Symbol — SymbolTokenizer, clean bijection (bounded relations only)

Same model architecture, same seeds, same number of terms per sequence.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from recurrence_relations import get_relation
from model import FibonacciTransformer, NumberTokenizer
from digit_tokenizer import DigitTokenizer
from symbol_tokenizer import SymbolTokenizer
from experiment_framework import (
    DATA_SEED, RANDOM_SEEDS, set_all_seeds, make_run_id,
    TrainConfig, train_model, load_trained_model,
    evaluate_teacher_forced, aggregate_results, save_run,
    generate_examples,
)

OUTPUT_DIR = 'runs/recurrence/exp3_representation'
SEQUENCE_LENGTH = 25
EPOCHS = 50


# ---------------------------------------------------------------------------
# Digit-based training (adapted from digit_train.py)
# ---------------------------------------------------------------------------
class DigitDataset(Dataset):
    """Dataset for digit-wise training. Stores (context_ids, target_ids) pairs."""
    def __init__(self, examples: List[Tuple[List[int], List[int]]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ctx_ids, tgt_ids = self.examples[idx]
        return (
            torch.tensor(ctx_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def _digit_collate(batch):
    """Collate digit batches with padding."""
    contexts, targets = zip(*batch)
    contexts_padded = pad_sequence(contexts, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return contexts_padded, targets_padded


def train_digit_model(
    train_examples: List[Tuple[List[int], int]],
    tokenizer: DigitTokenizer,
    random_seed: int,
    output_dir: str,
    epochs: int = EPOCHS,
    device: str = None,
    verbose: bool = True,
) -> Tuple[FibonacciTransformer, Dict]:
    """Train using digit-wise encoding (same approach as digit_train.py)."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_all_seeds(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    # Encode: (context_numbers, target_number) -> (context_token_ids, target_token_ids)
    encoded = []
    for ctx_nums, tgt_num in train_examples:
        ctx_ids = tokenizer.encode_sequence(ctx_nums)
        tgt_ids = tokenizer.encode_sequence([tgt_num])
        encoded.append((ctx_ids, tgt_ids))

    dataset = DigitDataset(encoded)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=_digit_collate)

    model = FibonacciTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128, nhead=4, num_layers=3, dim_feedforward=512,
        dropout=0.1, max_seq_len=100,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    history = {'losses': [], 'accuracies': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_numbers = 0
        total_numbers = 0

        for contexts, targets in loader:
            contexts = contexts.to(device)
            targets = targets.to(device)

            batch_size = contexts.size(0)
            target_len = targets.size(1)

            optimizer.zero_grad()

            # Autoregressive: concat context + target, predict shifted
            full_seq = torch.cat([contexts, targets], dim=1)
            logits = model(full_seq)

            context_len = contexts.size(1)
            pred_logits = logits[:, context_len - 1:context_len + target_len - 1, :]

            pred_flat = pred_logits.reshape(-1, pred_logits.size(-1))
            tgt_flat = targets.reshape(-1)

            mask = tgt_flat != 0
            if mask.sum() > 0:
                loss = criterion(pred_flat[mask], tgt_flat[mask])
            else:
                loss = torch.tensor(0.0, device=device)

            if loss.item() > 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Number-level accuracy
            with torch.no_grad():
                for b in range(batch_size):
                    pred_digits = []
                    for i in range(target_len):
                        if targets[b, i] == 0:
                            break
                        pred_digits.append(pred_logits[b, i].argmax().item())
                    target_digits = targets[b][targets[b] != 0].tolist()
                    if pred_digits == target_digits:
                        correct_numbers += 1
                    total_numbers += 1

        avg_loss = total_loss / max(len(loader), 1)
        accuracy = 100.0 * correct_numbers / max(total_numbers, 1)
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
            print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, num_acc={accuracy:.2f}%")

    if verbose:
        print(f"    Digit training done. Best loss={best_loss:.4f}, Final acc={accuracy:.2f}%")

    return model, history


def evaluate_digit_teacher_forced(
    model: FibonacciTransformer,
    tokenizer: DigitTokenizer,
    relation,
    seeds: List[tuple],
    seq_len: int,
    ctx_win: int,
    device: str = None,
) -> Dict:
    """
    Teacher-forced digit evaluation.
    Reports both token accuracy and number exact-match accuracy.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    total_tokens_correct = 0
    total_tokens = 0
    total_numbers_correct = 0
    total_numbers = 0

    with torch.no_grad():
        for seed in seeds:
            seq = relation.generate_sequence(seed, seq_len)

            for i in range(ctx_win, len(seq)):
                context = seq[max(0, i - ctx_win):i]
                target = seq[i]

                ctx_ids = tokenizer.encode_sequence(context)
                tgt_ids = tokenizer.encode_sequence([target])

                full_ids = ctx_ids + tgt_ids
                src = torch.tensor([full_ids], dtype=torch.long, device=device)
                logits = model(src)

                ctx_len = len(ctx_ids)
                pred_logits = logits[0, ctx_len - 1:ctx_len + len(tgt_ids) - 1, :]

                # Token accuracy
                for j in range(len(tgt_ids)):
                    pred_id = pred_logits[j].argmax().item()
                    if pred_id == tgt_ids[j]:
                        total_tokens_correct += 1
                    total_tokens += 1

                # Number exact-match
                pred_digits = [pred_logits[j].argmax().item() for j in range(len(tgt_ids))]
                if pred_digits == tgt_ids:
                    total_numbers_correct += 1
                total_numbers += 1

    return {
        'token_accuracy': 100.0 * total_tokens_correct / max(total_tokens, 1),
        'number_exact_match': 100.0 * total_numbers_correct / max(total_numbers, 1),
        'total_tokens': total_tokens,
        'total_numbers': total_numbers,
    }


# ---------------------------------------------------------------------------
# Run one representation
# ---------------------------------------------------------------------------
def run_representation(
    relation_name: str,
    rep_name: str,
    random_seed: int,
) -> Dict:
    """Train + evaluate one representation variant."""
    relation = get_relation(relation_name)
    ctx_win = relation.order + 8

    train_seeds = relation.get_training_seeds()
    test_seeds = relation.get_test_seeds()

    # Generate number-level examples (shared across reps)
    train_examples, vocabulary = generate_examples(
        relation, train_seeds, SEQUENCE_LENGTH, ctx_win,
    )
    _, test_vocab = generate_examples(
        relation, test_seeds, SEQUENCE_LENGTH, ctx_win,
    )

    run_id = make_run_id(relation_name, rep_name, 'representation', 'default', random_seed)
    run_dir = os.path.join(OUTPUT_DIR, run_id)

    result = {
        'run_id': run_id,
        'relation': relation_name,
        'representation': rep_name,
        'random_seed': random_seed,
        'n_train_examples': len(train_examples),
    }

    if rep_name == 'digit':
        tokenizer = DigitTokenizer()
        model, history = train_digit_model(
            train_examples, tokenizer, random_seed, run_dir,
        )
        eval_res = evaluate_digit_teacher_forced(
            model, tokenizer, relation, test_seeds, SEQUENCE_LENGTH, ctx_win,
        )
        result['token_accuracy'] = eval_res['token_accuracy']
        result['number_exact_match'] = eval_res['number_exact_match']
        result['mean_accuracy'] = eval_res['number_exact_match']  # Primary metric
        # Minimal history for save_run
        _save_rep_manifest(run_id, run_dir, train_seeds, test_seeds, history, result)

    elif rep_name == 'integer':
        full_vocab = sorted(set(vocabulary) | set(test_vocab))
        tokenizer = NumberTokenizer(full_vocab)
        config = TrainConfig(
            vocab_size=tokenizer.vocab_size,
            context_window=ctx_win,
            epochs=EPOCHS,
        )
        model, history = train_model(
            train_examples, tokenizer, config, random_seed, run_dir, verbose=True,
        )
        eval_results = evaluate_teacher_forced(
            model, tokenizer, relation, test_seeds, SEQUENCE_LENGTH, ctx_win,
        )
        agg = aggregate_results(eval_results)
        result['mean_accuracy'] = agg['mean_accuracy']
        result['std_accuracy'] = agg['std_accuracy']
        save_run(run_id, OUTPUT_DIR, config, train_seeds, test_seeds, history, eval_results,
                 extra_meta={'experiment': 'representation', 'representation': rep_name,
                             'total_training_examples': len(train_examples)})

    elif rep_name == 'symbol':
        # Only valid for bounded relations
        modulus = getattr(relation, 'modulus', None)
        if modulus is None:
            print(f"  SKIP: symbol rep not applicable to {relation_name} (unbounded)")
            result['mean_accuracy'] = None
            return result

        tokenizer = SymbolTokenizer(modulus)
        config = TrainConfig(
            vocab_size=tokenizer.vocab_size,
            context_window=ctx_win,
            epochs=EPOCHS,
        )
        model, history = train_model(
            train_examples, tokenizer, config, random_seed, run_dir, verbose=True,
        )
        eval_results = evaluate_teacher_forced(
            model, tokenizer, relation, test_seeds, SEQUENCE_LENGTH, ctx_win,
        )
        agg = aggregate_results(eval_results)
        result['mean_accuracy'] = agg['mean_accuracy']
        result['std_accuracy'] = agg['std_accuracy']
        save_run(run_id, OUTPUT_DIR, config, train_seeds, test_seeds, history, eval_results,
                 extra_meta={'experiment': 'representation', 'representation': rep_name,
                             'total_training_examples': len(train_examples)})
    else:
        raise ValueError(f"Unknown representation: {rep_name}")

    print(f"    {rep_name}: accuracy={result.get('mean_accuracy', 'N/A')}")
    return result


def _save_rep_manifest(run_id, run_dir, train_seeds, test_seeds, history, result):
    """Save a minimal manifest for digit experiments."""
    os.makedirs(os.path.join(OUTPUT_DIR, run_id), exist_ok=True)
    manifest = {
        'run_id': run_id,
        'train_seeds': [list(s) for s in train_seeds],
        'test_seeds': [list(s) for s in test_seeds],
        'history': history,
        'result': result,
    }
    with open(os.path.join(OUTPUT_DIR, run_id, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------
def run_exp3(
    relations: List[str] = None,
    representations: List[str] = None,
    random_seeds: List[int] = None,
):
    """Run the full Experiment 3."""
    if relations is None:
        relations = ['modular_arithmetic']
    if representations is None:
        representations = ['digit', 'integer', 'symbol']
    if random_seeds is None:
        random_seeds = RANDOM_SEEDS

    print(f"=== Experiment 3: Representation Control ===")
    print(f"Relations: {relations}")
    print(f"Representations: {representations}")
    print(f"Random seeds: {random_seeds}")

    all_results = []
    for rel_name in relations:
        print(f"\n--- {rel_name} ---")
        for rep_name in representations:
            print(f"  Representation: {rep_name}")
            for rs in random_seeds:
                result = run_representation(rel_name, rep_name, rs)
                all_results.append(result)

    # Save summary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Relation':<25} {'Rep':<10} {'Accuracy (mean±std)':<25}")
    print(f"{'='*60}")
    for rel_name in relations:
        for rep_name in representations:
            rel_results = [r for r in all_results
                           if r['relation'] == rel_name and r['representation'] == rep_name
                           and r.get('mean_accuracy') is not None]
            if rel_results:
                accs = [r['mean_accuracy'] for r in rel_results]
                print(f"{rel_name:<25} {rep_name:<10} "
                      f"{np.mean(accs):.2f} ± {np.std(accs):.2f}")

    print(f"\nExperiment 3 complete. {len(all_results)} runs saved to {OUTPUT_DIR}")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 3: Representation Control')
    parser.add_argument('--relations', nargs='+', default=['modular_arithmetic'])
    parser.add_argument('--representations', nargs='+', default=['digit', 'integer', 'symbol'])
    parser.add_argument('--random_seeds', nargs='+', type=int, default=RANDOM_SEEDS)
    args = parser.parse_args()

    run_exp3(
        relations=args.relations,
        representations=args.representations,
        random_seeds=args.random_seeds,
    )
