"""
Experiment 5 — Baselines (N-gram / kNN)  [no-UNK rewrite]

Goal: Show transformer doesn't exceed distributional matching on failing tasks.

No-UNK guarantee:
  - Modular arithmetic: SymbolTokenizer (one token per residue, UNK impossible)
  - Unbounded relations (Fibonacci, linear): closed integer vocabulary built from
    ALL values appearing in generated train + test sequences (no max_value truncation)
  - Every run verifies zero UNK tokens in both inputs and targets before reporting

Baselines:
  (A) N-gram next-token model with backoff + Laplace smoothing
  (B) kNN retrieval baseline with Hamming distance

Comparison against transformer on same test seeds and teacher-forced protocol.
"""
import argparse
import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np

from recurrence_relations import get_relation, ModularArithmetic
from model import NumberTokenizer, FibonacciTransformer
from symbol_tokenizer import SymbolTokenizer
from experiment_framework import (
    DATA_SEED, RANDOM_SEEDS, make_run_id,
    TrainConfig, train_model,
    evaluate_teacher_forced, aggregate_results, save_run,
    EvalResult,
)

OUTPUT_DIR = 'runs/baselines'
SEQUENCE_LENGTH = 25


# ---------------------------------------------------------------------------
# N-gram Baseline
# ---------------------------------------------------------------------------
class NgramBaseline:
    """N-gram next-token model with backoff and Laplace smoothing."""

    def __init__(self, n: int, smoothing: float = 1.0):
        self.n = n
        self.smoothing = smoothing
        self.counts = {}       # {order: {context_tuple: {token: count}}}
        self.vocab = set()

    def train(self, examples: List[Tuple[List[int], int]]):
        """Build n-gram counts from (context, target) pairs."""
        for ctx, tgt in examples:
            self.vocab.add(tgt)
            self.vocab.update(ctx)
            for order in range(self.n, 0, -1):
                if order not in self.counts:
                    self.counts[order] = defaultdict(lambda: defaultdict(int))
                if len(ctx) >= order:
                    key = tuple(ctx[-order:])
                    self.counts[order][key][tgt] += 1
        if 0 not in self.counts:
            self.counts[0] = defaultdict(lambda: defaultdict(int))
        for ctx, tgt in examples:
            self.counts[0][()][tgt] += 1

    def predict(self, context: List[int]) -> int:
        """Predict next token using max-likelihood with backoff."""
        for order in range(self.n, -1, -1):
            if order not in self.counts:
                continue
            if len(context) >= order:
                key = tuple(context[-order:]) if order > 0 else ()
                if key in self.counts[order] and self.counts[order][key]:
                    token_counts = self.counts[order][key]
                    V = len(self.vocab)
                    best_token = None
                    best_score = -1
                    for token in self.vocab:
                        score = (token_counts.get(token, 0) + self.smoothing) / (
                            sum(token_counts.values()) + self.smoothing * V
                        )
                        if score > best_score:
                            best_score = score
                            best_token = token
                    if best_token is not None:
                        return best_token
        if self.vocab:
            return max(self.vocab, key=lambda t: self.counts.get(0, {}).get((), {}).get(t, 0))
        return 0


# ---------------------------------------------------------------------------
# kNN Baseline
# ---------------------------------------------------------------------------
class KNNBaseline:
    """kNN retrieval baseline using Hamming distance on context window."""

    def __init__(self, k: int = 5, context_window: int = 10):
        self.k = k
        self.context_window = context_window
        self.contexts = []
        self.targets = []

    def train(self, examples: List[Tuple[List[int], int]]):
        """Store all (context, target) pairs."""
        for ctx, tgt in examples:
            if len(ctx) > self.context_window:
                ctx = ctx[-self.context_window:]
            elif len(ctx) < self.context_window:
                ctx = [0] * (self.context_window - len(ctx)) + ctx
            self.contexts.append(ctx)
            self.targets.append(tgt)

    def _hamming_distance(self, a: List[int], b: List[int]) -> int:
        return sum(1 for x, y in zip(a, b) if x != y)

    def predict(self, context: List[int]) -> int:
        if len(context) > self.context_window:
            context = context[-self.context_window:]
        elif len(context) < self.context_window:
            context = [0] * (self.context_window - len(context)) + context

        dists = [(self._hamming_distance(context, c), t)
                 for c, t in zip(self.contexts, self.targets)]
        dists.sort(key=lambda x: x[0])

        top_k = [t for _, t in dists[:self.k]]
        vote_counts = defaultdict(int)
        for t in top_k:
            vote_counts[t] += 1
        return max(vote_counts, key=vote_counts.get)


# ---------------------------------------------------------------------------
# Evaluate a baseline (operates on raw numbers, tokenizer-agnostic)
# ---------------------------------------------------------------------------
def evaluate_baseline(
    baseline,
    relation,
    test_seeds: List[tuple],
    seq_len: int,
    ctx_win: int,
) -> List[EvalResult]:
    """Teacher-forced evaluation of a baseline model."""
    results = []
    for seed in test_seeds:
        seq = relation.generate_sequence(seed, seq_len)
        preds, targets, per_pos = [], [], []

        for i in range(ctx_win, len(seq)):
            context = seq[max(0, i - ctx_win):i]
            target = seq[i]
            pred = baseline.predict(context)
            preds.append(pred)
            targets.append(target)
            per_pos.append(pred == target)

        n_correct = sum(per_pos)
        accuracy = 100.0 * n_correct / len(per_pos) if per_pos else 0.0
        first_err = next((idx for idx, c in enumerate(per_pos) if not c), None)

        results.append(EvalResult(
            seed=seed, accuracy=accuracy,
            per_position_correct=per_pos,
            first_error_index=first_err,
            predictions=preds, targets=targets,
        ))
    return results


# ---------------------------------------------------------------------------
# Vocabulary & tokenizer helpers (no-UNK guarantee)
# ---------------------------------------------------------------------------
def _generate_all_sequences(relation, seeds, seq_len):
    """Generate raw sequences with NO truncation."""
    return [relation.generate_sequence(seed, seq_len) for seed in seeds]


def _build_examples_from_sequences(sequences, ctx_win):
    """Build (context, target) pairs from raw sequences (no max_value filter)."""
    examples = []
    for seq in sequences:
        for i in range(ctx_win, len(seq)):
            examples.append((seq[i - ctx_win:i], seq[i]))
    return examples


def _build_tokenizer(relation, train_seqs, test_seqs):
    """
    Build the appropriate tokenizer for a relation.

    - ModularArithmetic → SymbolTokenizer (bijection, UNK impossible by construction)
    - Unbounded relations → NumberTokenizer with closed vocabulary from ALL values
      in train + test sequences (guarantees zero UNK at eval time)
    """
    if isinstance(relation, ModularArithmetic):
        return SymbolTokenizer(relation.modulus), 'symbol'

    # Closed integer vocabulary: union of all values in train + test sequences
    all_numbers = set()
    for seq in train_seqs + test_seqs:
        all_numbers.update(seq)
    return NumberTokenizer(sorted(all_numbers)), 'integer_closed'


def _verify_no_unk(tokenizer, examples, label):
    """
    Verify zero UNK tokens in encoded examples.
    Returns unk_count (should be 0).
    """
    unk_id = tokenizer.unk_id
    unk_count = 0
    for ctx, tgt in examples:
        for tok_id in tokenizer.encode(ctx):
            if tok_id == unk_id:
                unk_count += 1
        if tokenizer.encode([tgt])[0] == unk_id:
            unk_count += 1
    return unk_count


def _seq_stats(sequences):
    """Compute max and min value across a list of sequences."""
    all_vals = [v for seq in sequences for v in seq]
    return {'max_value': max(all_vals), 'min_value': min(all_vals),
            'n_unique': len(set(all_vals))}


# ---------------------------------------------------------------------------
# Run one relation
# ---------------------------------------------------------------------------
def run_relation(
    relation_name: str,
    random_seeds: List[int],
) -> Dict:
    """Run all baselines + transformer for one relation (no-UNK)."""
    relation = get_relation(relation_name)
    ctx_win = relation.order + 8

    train_seeds = relation.get_training_seeds()
    test_seeds = relation.get_test_seeds()

    # Generate raw sequences (NO truncation)
    train_seqs = _generate_all_sequences(relation, train_seeds, SEQUENCE_LENGTH)
    test_seqs = _generate_all_sequences(relation, test_seeds, SEQUENCE_LENGTH)

    # Build examples from raw sequences
    train_examples = _build_examples_from_sequences(train_seqs, ctx_win)
    test_examples = _build_examples_from_sequences(test_seqs, ctx_win)

    # Build tokenizer (no-UNK guarantee)
    tokenizer, tok_type = _build_tokenizer(relation, train_seqs, test_seqs)

    # Verify no UNK in training data
    train_unk = _verify_no_unk(tokenizer, train_examples, 'train')
    test_unk = _verify_no_unk(tokenizer, test_examples, 'test')
    assert train_unk == 0, f"FATAL: {train_unk} UNK tokens in training data for {relation_name}"
    assert test_unk == 0, f"FATAL: {test_unk} UNK tokens in test data for {relation_name}"

    # Sequence stats
    train_stats = _seq_stats(train_seqs)
    test_stats = _seq_stats(test_seqs)
    print(f"  Tokenizer: {tok_type}, vocab_size={tokenizer.vocab_size}")
    print(f"  Train: {len(train_examples)} examples, max_value={train_stats['max_value']}, "
          f"{train_stats['n_unique']} unique values")
    print(f"  Test:  {len(test_examples)} examples, max_value={test_stats['max_value']}, "
          f"{test_stats['n_unique']} unique values")
    print(f"  UNK check: train={train_unk}, test={test_unk} (verified zero)")

    # N-gram orders match context window
    n_list = [max(2, ctx_win - 2), ctx_win, ctx_win + 2]

    results = {
        'relation': relation_name,
        'tokenizer': tok_type,
        'vocab_size': tokenizer.vocab_size,
        'train_stats': train_stats,
        'test_stats': test_stats,
        'unk_train': train_unk,
        'unk_test': test_unk,
        'n_train_examples': len(train_examples),
        'n_test_examples': len(test_examples),
        'baselines': {},
        'transformer': {},
    }

    # --- N-gram baselines (operate on raw numbers, tokenizer-agnostic) ---
    for n in n_list:
        ngram = NgramBaseline(n=n)
        ngram.train(train_examples)
        eval_results = evaluate_baseline(ngram, relation, test_seeds, SEQUENCE_LENGTH, ctx_win)
        agg = aggregate_results(eval_results)
        key = f'ngram_{n}'
        results['baselines'][key] = {
            'type': 'ngram', 'n': n,
            'mean_accuracy': agg['mean_accuracy'],
            'std_accuracy': agg['std_accuracy'],
        }
        print(f"    {key}: acc={agg['mean_accuracy']:.2f}%")

    # --- kNN baseline ---
    knn = KNNBaseline(k=5, context_window=ctx_win)
    knn.train(train_examples)
    eval_results = evaluate_baseline(knn, relation, test_seeds, SEQUENCE_LENGTH, ctx_win)
    agg = aggregate_results(eval_results)
    results['baselines']['knn_5'] = {
        'type': 'knn', 'k': 5,
        'mean_accuracy': agg['mean_accuracy'],
        'std_accuracy': agg['std_accuracy'],
    }
    print(f"    kNN(k=5): acc={agg['mean_accuracy']:.2f}%")

    # --- Transformer ---
    config = TrainConfig(
        vocab_size=tokenizer.vocab_size,
        context_window=ctx_win,
        epochs=50,
    )

    transformer_accs = []
    for rs in random_seeds:
        rep_label = tok_type
        run_id = make_run_id(relation_name, rep_label, 'baselines', 'transformer', rs)
        run_dir = os.path.join(OUTPUT_DIR, run_id)

        print(f"    Transformer RS={rs}...", end=' ')
        model, history = train_model(
            train_examples, tokenizer, config, rs, run_dir, verbose=False,
        )
        eval_results = evaluate_teacher_forced(
            model, tokenizer, relation, test_seeds, SEQUENCE_LENGTH, ctx_win,
        )
        agg = aggregate_results(eval_results)
        transformer_accs.append(agg['mean_accuracy'])
        print(f"acc={agg['mean_accuracy']:.2f}%")

        save_run(run_id, OUTPUT_DIR, config, train_seeds, test_seeds, history, eval_results,
                 extra_meta={
                     'experiment': 'baselines',
                     'relation': relation_name,
                     'tokenizer': tok_type,
                     'total_training_examples': len(train_examples),
                     'unk_train': train_unk,
                     'unk_test': test_unk,
                     'train_stats': train_stats,
                     'test_stats': test_stats,
                 })

    results['transformer'] = {
        'mean_accuracy': float(np.mean(transformer_accs)),
        'std_accuracy': float(np.std(transformer_accs)),
        'per_seed_accs': transformer_accs,
    }

    return results


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------
def run_exp5(
    relations: List[str] = None,
    random_seeds: List[int] = None,
):
    """Run the full Experiment 5 (no-UNK)."""
    if relations is None:
        relations = ['fibonacci', 'modular_arithmetic']
    if random_seeds is None:
        random_seeds = RANDOM_SEEDS

    print(f"=== Experiment 5: Baselines (no-UNK) ===")
    print(f"Relations: {relations}")
    print(f"Random seeds (transformer): {random_seeds}")

    all_results = []
    for rel_name in relations:
        print(f"\n--- {rel_name} ---")
        result = run_relation(rel_name, random_seeds)
        all_results.append(result)

    # Save summary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  Baseline Comparison (no-UNK)")
    print(f"{'='*70}")
    print(f"{'Relation':<25} {'Method':<15} {'Accuracy':<20}")
    print(f"{'-'*70}")
    for result in all_results:
        rel = result['relation']
        tok = result['tokenizer']
        print(f"  [{tok}, vocab={result['vocab_size']}, "
              f"UNK: train={result['unk_train']}, test={result['unk_test']}]")
        for key, val in result['baselines'].items():
            print(f"{rel:<25} {key:<15} {val['mean_accuracy']:6.2f}%")
        t = result['transformer']
        print(f"{rel:<25} {'transformer':<15} {t['mean_accuracy']:6.2f} ± {t['std_accuracy']:.2f}%")
        print()

    print(f"Experiment 5 complete. Results saved to {OUTPUT_DIR}")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 5: Baselines (no-UNK)')
    parser.add_argument('--relations', nargs='+',
                        default=['fibonacci', 'modular_arithmetic'],
                        help='Relations to test')
    parser.add_argument('--random_seeds', nargs='+', type=int, default=RANDOM_SEEDS,
                        help='Transformer init seeds')
    args = parser.parse_args()

    run_exp5(relations=args.relations, random_seeds=args.random_seeds)
