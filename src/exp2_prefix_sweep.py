"""
Experiment 2 — Prefix-Length Sweep

Goal: Is failure due to insufficient context, or inability to infer the rule?

Design:
  - Train once per relation (standard 20 seeds)
  - Evaluate at multiple prefix lengths
  - Two evaluation modes: teacher-forced and free-running
  - Training uses left-padding so PAD tokens are in training distribution
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np

from recurrence_relations import get_relation
from model import NumberTokenizer
from experiment_framework import (
    DATA_SEED, RANDOM_SEEDS, make_run_id,
    TrainConfig, train_model, load_trained_model,
    evaluate_teacher_forced, evaluate_autoregressive,
    aggregate_results, save_run, generate_examples,
)

OUTPUT_DIR = 'runs/recurrence/exp2_prefix_sweep'

# Prefix grids per relation type
PREFIX_GRIDS = {
    'unbounded': [2, 4, 8, 16, 22],   # seq_len=25
    'bounded': [2, 4, 8, 16, 32, 64],  # seq_len=128
}

RELATION_CONFIG = {
    'fibonacci':               {'type': 'unbounded', 'seq_len': 25},
    'modular_arithmetic':      {'type': 'bounded',   'seq_len': 128},
    'tribonacci':              {'type': 'unbounded', 'seq_len': 25},
    'linear':                  {'type': 'unbounded', 'seq_len': 25},
}


# ---------------------------------------------------------------------------
# Run one relation
# ---------------------------------------------------------------------------
def run_relation(relation_name: str, random_seed: int) -> Dict:
    """Train one model, evaluate at all prefix lengths."""
    relation = get_relation(relation_name)
    rcfg = RELATION_CONFIG[relation_name]
    seq_len = rcfg['seq_len']
    prefix_grid = PREFIX_GRIDS[rcfg['type']]
    ctx_win = relation.order + 8

    # Data generation (fixed DATA_SEED)
    train_seeds = relation.get_training_seeds()
    test_seeds = relation.get_test_seeds()

    train_examples, vocabulary = generate_examples(
        relation, train_seeds, seq_len, ctx_win,
    )
    _, test_vocab = generate_examples(
        relation, test_seeds, seq_len, ctx_win,
    )
    full_vocab = sorted(set(vocabulary) | set(test_vocab))
    tokenizer = NumberTokenizer(full_vocab)

    max_seq_len = 50 if rcfg['type'] == 'unbounded' else 150
    config = TrainConfig(
        vocab_size=tokenizer.vocab_size,
        context_window=ctx_win,
        epochs=50,
        max_seq_len=max_seq_len,
    )

    run_id = make_run_id(relation_name, 'integer', 'prefix_sweep', 'train', random_seed)
    run_dir = os.path.join(OUTPUT_DIR, run_id)

    # Train
    print(f"  Training {relation_name} RS={random_seed} ({len(train_examples)} examples)...")
    model, history = train_model(
        train_examples, tokenizer, config, random_seed, run_dir, verbose=False,
    )

    # Evaluate at each prefix length
    results_by_prefix = {}
    for prefix_len in prefix_grid:
        # Teacher-forced
        tf_results = evaluate_teacher_forced(
            model, tokenizer, relation, test_seeds, seq_len, ctx_win,
        )
        tf_agg = aggregate_results(tf_results)

        # Free-running
        ar_results = evaluate_autoregressive(
            model, tokenizer, relation, test_seeds, seq_len, prefix_len, ctx_win,
        )
        ar_agg = aggregate_results(ar_results)

        # Sequence exact-match: all predictions correct after prefix
        seq_exact = sum(1 for r in ar_results if all(r.per_position_correct)) / len(ar_results) * 100.0 if ar_results else 0.0

        results_by_prefix[prefix_len] = {
            'prefix_len': prefix_len,
            'teacher_forced': tf_agg,
            'free_running': ar_agg,
            'sequence_exact_match_pct': seq_exact,
            'first_error_indices': ar_agg['first_error_indices'],
        }

        print(f"    prefix={prefix_len}: TF={tf_agg['mean_accuracy']:.2f}%, "
              f"FR={ar_agg['mean_accuracy']:.2f}%, "
              f"SeqEM={seq_exact:.1f}%")

    # Save
    save_run(
        run_id, OUTPUT_DIR, config, train_seeds, test_seeds,
        history, tf_results,
        extra_meta={
            'experiment': 'prefix_sweep',
            'relation': relation_name,
            'random_seed': random_seed,
            'total_training_examples': len(train_examples),
            'prefix_results': results_by_prefix,
        },
    )

    return {
        'run_id': run_id,
        'relation': relation_name,
        'random_seed': random_seed,
        'prefix_results': results_by_prefix,
    }


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------
def run_exp2(
    relations: List[str] = None,
    random_seeds: List[int] = None,
):
    """Run the full Experiment 2."""
    if relations is None:
        relations = ['fibonacci', 'modular_arithmetic']
    if random_seeds is None:
        random_seeds = RANDOM_SEEDS

    print(f"=== Experiment 2: Prefix-Length Sweep ===")
    print(f"Relations: {relations}")
    print(f"Random seeds: {random_seeds}")
    print(f"Total training runs: {len(relations) * len(random_seeds)}")

    all_results = []
    for rel_name in relations:
        rcfg = RELATION_CONFIG[rel_name]
        print(f"\n--- {rel_name} (seq_len={rcfg['seq_len']}, "
              f"prefixes={PREFIX_GRIDS[rcfg['type']]}) ---")
        for rs in random_seeds:
            result = run_relation(rel_name, rs)
            all_results.append(result)

    # Save summary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nExperiment 2 complete. {len(all_results)} runs saved to {OUTPUT_DIR}")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 2: Prefix-Length Sweep')
    parser.add_argument('--relations', nargs='+',
                        default=['fibonacci', 'modular_arithmetic'],
                        help='Relations to test')
    parser.add_argument('--random_seeds', nargs='+', type=int, default=RANDOM_SEEDS)
    args = parser.parse_args()

    run_exp2(relations=args.relations, random_seeds=args.random_seeds)
