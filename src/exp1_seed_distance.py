"""
Experiment 1 — Seed-Distance Generalization Curve

Goal: Does accuracy degrade smoothly with seed distance, or collapse abruptly?

Design:
  - 4 training bands (Core, Near d∈[1,5], Mid d∈[6,15], Far d∈[16,2*S_max])
  - Each band is a complete (non-additive) training set
  - 161 test seeds partitioned by distance quantiles into Near/Mid/Far
  - 4 relations x 4 bands x 3 RS = 48 training runs (default)
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

from recurrence_relations import get_relation, RecurrenceRelation
from model import NumberTokenizer
from experiment_framework import (
    DATA_SEED, RANDOM_SEEDS, set_all_seeds, make_run_id,
    TrainConfig, train_model, load_trained_model,
    evaluate_teacher_forced, aggregate_results, save_run,
    generate_examples, multi_run,
)

OUTPUT_DIR = 'runs/recurrence/exp1_seed_distance'
S_MAX = 50
N_TEST = 161
SEQUENCE_LENGTH = 25


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------
def compute_distance(s1: tuple, s2: tuple, modulus: Optional[int] = None) -> int:
    """
    Compute distance between two seeds.
    - Order-2 (no modulus): L1 = |a1-a2| + |b1-b2|
    - Modular: min(|a-b|, M-|a-b|) per component, then sum
    - Order-1: |a1-a2|
    """
    if modulus is not None:
        total = 0
        for a, b in zip(s1, s2):
            diff = abs(a - b)
            total += min(diff, modulus - diff)
        return total
    return sum(abs(a - b) for a, b in zip(s1, s2))


def distance_to_nearest(seed: tuple, reference_seeds: List[tuple],
                        modulus: Optional[int] = None) -> int:
    """Distance from seed to the nearest reference seed."""
    return min(compute_distance(seed, ref, modulus) for ref in reference_seeds)


# ---------------------------------------------------------------------------
# Seed space and sampling
# ---------------------------------------------------------------------------
def _get_seed_space(relation_name: str) -> List[tuple]:
    """Return all valid seeds for a relation."""
    rel = get_relation(relation_name)
    if relation_name == 'modular_arithmetic':
        return [(a, b) for a in range(31) for b in range(31)]
    elif relation_name == 'geometric':
        return [(a,) for a in range(S_MAX + 1)]
    elif rel.order == 3:
        # Tribonacci: limit range to keep tractable
        return [(a, b, c) for a in range(S_MAX + 1)
                for b in range(S_MAX + 1)
                for c in range(S_MAX + 1)
                if a + b + c <= 2 * S_MAX]  # rough bound
    else:
        # Order-2 (Fibonacci, Fib+1, Linear)
        return [(a, b) for a in range(S_MAX + 1) for b in range(S_MAX + 1)]


def _get_modulus(relation_name: str) -> Optional[int]:
    if relation_name == 'modular_arithmetic':
        return 31
    return None


def sample_seeds_in_band(
    core_seeds: List[tuple],
    d_min: int,
    d_max: int,
    count: int,
    seed_space: List[tuple],
    rng: np.random.RandomState,
    modulus: Optional[int] = None,
    exclude: set = None,
) -> List[tuple]:
    """Sample `count` seeds from seed_space with distance in [d_min, d_max] from nearest core seed."""
    if exclude is None:
        exclude = set()
    candidates = []
    for s in seed_space:
        if s in exclude:
            continue
        d = distance_to_nearest(s, core_seeds, modulus)
        if d_min <= d <= d_max:
            candidates.append(s)
    rng.shuffle(candidates)
    return candidates[:count]


def partition_test_by_quantile(
    test_seeds: List[tuple],
    train_seeds: List[tuple],
    modulus: Optional[int] = None,
) -> Dict[str, List[tuple]]:
    """Partition test seeds into Near/Mid/Far thirds by distance quantiles from nearest train seed."""
    dists = [(s, distance_to_nearest(s, train_seeds, modulus)) for s in test_seeds]
    dists.sort(key=lambda x: x[1])
    n = len(dists)
    third = n // 3
    return {
        'near': [s for s, _ in dists[:third]],
        'mid': [s for s, _ in dists[third:2 * third]],
        'far': [s for s, _ in dists[2 * third:]],
    }


# ---------------------------------------------------------------------------
# Band definitions
# ---------------------------------------------------------------------------
BANDS = {
    'band0_core': (0, 0),     # Placeholder — uses core seeds directly
    'band1_near': (1, 5),
    'band2_mid': (6, 15),
    'band3_far': (16, None),  # None = 2*S_MAX (computed per relation)
}


def _max_distance(relation_name: str) -> int:
    if relation_name == 'geometric':
        return S_MAX
    if relation_name == 'modular_arithmetic':
        return 30  # max modular distance: 15 per component * 2
    return 2 * S_MAX


# ---------------------------------------------------------------------------
# Run one condition
# ---------------------------------------------------------------------------
def run_condition(
    relation_name: str,
    band_name: str,
    n_train: int,
    random_seed: int,
) -> Dict:
    """Train + evaluate one condition. Returns aggregated results dict."""
    relation = get_relation(relation_name)
    modulus = _get_modulus(relation_name)
    seed_space = _get_seed_space(relation_name)
    data_rng = np.random.RandomState(DATA_SEED)

    # Core seeds
    core_seeds = relation.get_training_seeds()[:n_train]

    # Select training seeds for this band
    if band_name == 'band0_core':
        train_seeds = core_seeds
    else:
        d_min, d_max = BANDS[band_name]
        if d_max is None:
            d_max = _max_distance(relation_name)
        train_seeds = sample_seeds_in_band(
            core_seeds, d_min, d_max, n_train, seed_space,
            data_rng, modulus, exclude=set(core_seeds),
        )
        if len(train_seeds) == 0:
            print(f"  SKIP: No seeds available for {band_name} "
                  f"(d_range=[{d_min},{d_max}], relation={relation_name})")
            return {
                'run_id': None, 'relation': relation_name, 'band': band_name,
                'n_train': n_train, 'random_seed': random_seed,
                'mean_accuracy': None, 'std_accuracy': None,
                'partition_accuracy': {}, 'n_train_examples': 0,
                'n_train_seeds': 0, 'skipped': True,
            }
        if len(train_seeds) < n_train:
            print(f"  WARNING: Only found {len(train_seeds)} seeds for {band_name} "
                  f"(requested {n_train})")

    # Test seeds: 161 seeds not in any training set
    all_train_set = set(core_seeds) | set(train_seeds)
    test_candidates = [s for s in seed_space if s not in all_train_set]
    data_rng2 = np.random.RandomState(DATA_SEED + 1)
    data_rng2.shuffle(test_candidates)
    test_seeds = test_candidates[:N_TEST]

    # Context window
    ctx_win = relation.order + 8

    # Generate examples
    train_examples, vocabulary = generate_examples(
        relation, train_seeds, SEQUENCE_LENGTH, ctx_win,
    )

    # Also include test vocabulary for tokenizer completeness
    test_examples_tmp, test_vocab = generate_examples(
        relation, test_seeds, SEQUENCE_LENGTH, ctx_win,
    )
    full_vocab = sorted(set(vocabulary) | set(test_vocab))
    tokenizer = NumberTokenizer(full_vocab)

    # Config
    config = TrainConfig(
        vocab_size=tokenizer.vocab_size,
        context_window=ctx_win,
        epochs=50,
    )

    # Run ID
    cfg_str = f"{band_name}_n{n_train}"
    run_id = make_run_id(relation_name, 'integer', 'seed_distance', cfg_str, random_seed)

    run_dir = os.path.join(OUTPUT_DIR, run_id)

    # Train
    print(f"  Training {run_id} ({len(train_examples)} examples)...")
    model, history = train_model(
        train_examples, tokenizer, config, random_seed, run_dir, verbose=False,
    )

    # Evaluate
    eval_results = evaluate_teacher_forced(
        model, tokenizer, relation, test_seeds, SEQUENCE_LENGTH, ctx_win,
    )
    agg = aggregate_results(eval_results)

    # Partition test by distance quantile
    partition = partition_test_by_quantile(test_seeds, train_seeds, modulus)
    partition_acc = {}
    for bucket, bucket_seeds in partition.items():
        bucket_results = [r for r in eval_results if r.seed in bucket_seeds]
        if bucket_results:
            partition_acc[bucket] = aggregate_results(bucket_results)['mean_accuracy']
        else:
            partition_acc[bucket] = None

    # Save manifest
    save_run(
        run_id, OUTPUT_DIR, config, train_seeds, test_seeds,
        history, eval_results,
        extra_meta={
            'experiment': 'seed_distance',
            'relation': relation_name,
            'band': band_name,
            'n_train': n_train,
            'random_seed': random_seed,
            'total_training_examples': len(train_examples),
            'partition_accuracy': partition_acc,
        },
    )

    return {
        'run_id': run_id,
        'relation': relation_name,
        'band': band_name,
        'n_train': n_train,
        'random_seed': random_seed,
        'mean_accuracy': agg['mean_accuracy'],
        'std_accuracy': agg['std_accuracy'],
        'partition_accuracy': partition_acc,
        'n_train_examples': len(train_examples),
        'n_train_seeds': len(train_seeds),
    }


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------
def run_exp1(
    relations: List[str] = None,
    train_seed_count: int = 20,
    random_seeds: List[int] = None,
):
    """Run the full Experiment 1."""
    if relations is None:
        relations = ['fibonacci', 'modular_arithmetic', 'fibonacci_plus_constant', 'geometric']
    if random_seeds is None:
        random_seeds = RANDOM_SEEDS

    band_names = list(BANDS.keys())
    all_results = []

    print(f"=== Experiment 1: Seed-Distance Generalization ===")
    print(f"Relations: {relations}")
    print(f"Bands: {band_names}")
    print(f"Train seed count: {train_seed_count}")
    print(f"Random seeds: {random_seeds}")
    print(f"Total runs: {len(relations) * len(band_names) * len(random_seeds)}")

    for rel_name in relations:
        print(f"\n--- Relation: {rel_name} ---")
        for band_name in band_names:
            for rs in random_seeds:
                result = run_condition(rel_name, band_name, train_seed_count, rs)
                all_results.append(result)
                if result.get('skipped'):
                    print(f"    {band_name} RS={rs}: SKIPPED")
                else:
                    print(f"    {band_name} RS={rs}: acc={result['mean_accuracy']:.2f}% "
                          f"(near={result['partition_accuracy'].get('near', 'N/A')}, "
                          f"mid={result['partition_accuracy'].get('mid', 'N/A')}, "
                          f"far={result['partition_accuracy'].get('far', 'N/A')})")

    # Save summary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nExperiment 1 complete. {len(all_results)} runs saved to {OUTPUT_DIR}")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 1: Seed-Distance Generalization')
    parser.add_argument('--relations', nargs='+',
                        default=['fibonacci', 'modular_arithmetic',
                                 'fibonacci_plus_constant', 'geometric'],
                        help='Relations to test')
    parser.add_argument('--train_seed_count', type=int, default=20,
                        help='Number of training seeds per band (20 or 100)')
    parser.add_argument('--random_seeds', nargs='+', type=int, default=RANDOM_SEEDS,
                        help='Random seeds for model init')
    args = parser.parse_args()

    run_exp1(
        relations=args.relations,
        train_seed_count=args.train_seed_count,
        random_seeds=args.random_seeds,
    )
