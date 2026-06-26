"""
Experiment 4 — Data Regime Sweep

Goal: Separate "memorizing trajectories" from "learning rules."

Design:
  - Modulus M=97 (primary), optionally M=127
  - 6 conditions: {Unique, Repeats} x {Few, Mid, Many}
  - All conditions use 90 seed-slots → same total training examples
  - Fixed DATA_SEED for data, only model init varies across RS
  - 6 conditions x 3 RS = 18 runs per modulus
"""
import argparse
import json
import os
from typing import List, Dict, Tuple

import numpy as np

from recurrence_relations import get_modular_relation, ModularArithmetic
from model import NumberTokenizer
from experiment_framework import (
    DATA_SEED, RANDOM_SEEDS, make_run_id,
    TrainConfig, train_model,
    evaluate_teacher_forced, aggregate_results, save_run,
    generate_examples,
)

OUTPUT_DIR = 'runs/recurrence/exp4_data_regime'
SEQUENCE_LENGTH = 128
CONTEXT_WINDOW = 10
TOTAL_SLOTS = 90
N_TEST = 161
EPOCHS = 50

# 6 conditions: (name, pool_size, mode)
CONDITIONS = [
    ('unique_few',    10, 'unique'),
    ('unique_mid',    30, 'unique'),
    ('unique_many',   90, 'unique'),
    ('repeats_few',    5, 'repeats'),
    ('repeats_mid',   10, 'repeats'),
    ('repeats_many',  30, 'repeats'),
]


# ---------------------------------------------------------------------------
# Seed pool generation
# ---------------------------------------------------------------------------
def generate_seed_pools(modulus: int) -> Tuple[List[tuple], List[tuple]]:
    """
    Generate deterministic train pool (90) and test seeds (161).
    Uses DATA_SEED for reproducibility.
    """
    rng = np.random.RandomState(DATA_SEED)
    all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    rng.shuffle(all_pairs)
    train_pool = all_pairs[:TOTAL_SLOTS]
    test_seeds = all_pairs[TOTAL_SLOTS:TOTAL_SLOTS + N_TEST]
    return train_pool, test_seeds


def build_training_slots(
    train_pool: List[tuple],
    pool_size: int,
    mode: str,
    rng: np.random.RandomState,
) -> List[tuple]:
    """
    Build 90 training seed-slots from pool.

    - unique: take first pool_size seeds, repeat deterministically to fill 90
    - repeats: draw pool_size seeds, sample with replacement to fill 90
    """
    if mode == 'unique':
        base = train_pool[:pool_size]
        slots = []
        while len(slots) < TOTAL_SLOTS:
            slots.extend(base)
        return slots[:TOTAL_SLOTS]
    elif mode == 'repeats':
        base = train_pool[:pool_size]
        indices = rng.randint(0, len(base), size=TOTAL_SLOTS)
        return [base[i] for i in indices]
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Run one condition
# ---------------------------------------------------------------------------
def run_condition(
    modulus: int,
    condition_name: str,
    pool_size: int,
    mode: str,
    random_seed: int,
) -> Dict:
    """Train + evaluate one data-regime condition."""
    relation = get_modular_relation(modulus)
    train_pool, test_seeds = generate_seed_pools(modulus)

    # Build training slots (deterministic per condition, not per RS)
    slot_rng = np.random.RandomState(DATA_SEED + hash(condition_name) % (2**31))
    train_slots = build_training_slots(train_pool, pool_size, mode, slot_rng)

    unique_seeds = list(set(train_slots))

    # Generate examples
    train_examples, vocabulary = generate_examples(
        relation, train_slots, SEQUENCE_LENGTH, CONTEXT_WINDOW,
    )
    _, test_vocab = generate_examples(
        relation, test_seeds, SEQUENCE_LENGTH, CONTEXT_WINDOW,
    )
    full_vocab = sorted(set(vocabulary) | set(test_vocab))
    tokenizer = NumberTokenizer(full_vocab)

    config = TrainConfig(
        vocab_size=tokenizer.vocab_size,
        context_window=CONTEXT_WINDOW,
        epochs=EPOCHS,
        max_seq_len=SEQUENCE_LENGTH + 10,
    )

    run_id = make_run_id(
        f'mod{modulus}', 'integer', 'data_regime',
        f'{condition_name}_pool{pool_size}', random_seed,
    )
    run_dir = os.path.join(OUTPUT_DIR, run_id)

    # Train
    print(f"  Training {run_id} ({len(train_examples)} examples, "
          f"{len(unique_seeds)} unique seeds)...")
    model, history = train_model(
        train_examples, tokenizer, config, random_seed, run_dir, verbose=False,
    )

    # Evaluate
    eval_results = evaluate_teacher_forced(
        model, tokenizer, relation, test_seeds, SEQUENCE_LENGTH, CONTEXT_WINDOW,
    )
    agg = aggregate_results(eval_results)

    # Also evaluate on training seeds (expect ~100%)
    train_eval = evaluate_teacher_forced(
        model, tokenizer, relation, unique_seeds[:20], SEQUENCE_LENGTH, CONTEXT_WINDOW,
    )
    train_agg = aggregate_results(train_eval)

    # Save
    save_run(
        run_id, OUTPUT_DIR, config,
        train_slots, test_seeds, history, eval_results,
        extra_meta={
            'experiment': 'data_regime',
            'modulus': modulus,
            'condition': condition_name,
            'pool_size': pool_size,
            'mode': mode,
            'random_seed': random_seed,
            'total_training_examples': len(train_examples),
            'unique_seeds_in_training': len(unique_seeds),
            'avg_repeats_per_seed': TOTAL_SLOTS / max(len(unique_seeds), 1),
            'train_accuracy': train_agg['mean_accuracy'],
        },
    )

    result = {
        'run_id': run_id,
        'modulus': modulus,
        'condition': condition_name,
        'pool_size': pool_size,
        'mode': mode,
        'random_seed': random_seed,
        'n_train_examples': len(train_examples),
        'unique_seeds': len(unique_seeds),
        'test_accuracy': agg['mean_accuracy'],
        'test_std': agg['std_accuracy'],
        'train_accuracy': train_agg['mean_accuracy'],
    }

    print(f"    test_acc={agg['mean_accuracy']:.2f}%, train_acc={train_agg['mean_accuracy']:.2f}%")
    return result


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------
def run_exp4(
    moduli: List[int] = None,
    random_seeds: List[int] = None,
):
    """Run the full Experiment 4."""
    if moduli is None:
        moduli = [97]
    if random_seeds is None:
        random_seeds = RANDOM_SEEDS

    print(f"=== Experiment 4: Data Regime Sweep ===")
    print(f"Moduli: {moduli}")
    print(f"Conditions: {[c[0] for c in CONDITIONS]}")
    print(f"Random seeds: {random_seeds}")
    total = len(moduli) * len(CONDITIONS) * len(random_seeds)
    print(f"Total runs: {total}")

    all_results = []
    for modulus in moduli:
        print(f"\n--- Modulus {modulus} ({modulus**2} total pairs) ---")
        for cond_name, pool_size, mode in CONDITIONS:
            print(f"  Condition: {cond_name} (pool={pool_size}, mode={mode})")
            for rs in random_seeds:
                result = run_condition(modulus, cond_name, pool_size, mode, rs)
                all_results.append(result)

    # Save summary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print 2x3 table
    for modulus in moduli:
        print(f"\n{'='*60}")
        print(f"  Data Regime Results — mod-{modulus}")
        print(f"{'='*60}")
        print(f"{'Condition':<18} {'Pool':<6} {'Mode':<10} {'Test Acc (mean±std)':<25} {'Train Acc':<12}")
        print(f"{'-'*60}")
        for cond_name, pool_size, mode in CONDITIONS:
            cond_results = [r for r in all_results
                            if r['modulus'] == modulus and r['condition'] == cond_name]
            if cond_results:
                test_accs = [r['test_accuracy'] for r in cond_results]
                train_accs = [r['train_accuracy'] for r in cond_results]
                print(f"{cond_name:<18} {pool_size:<6} {mode:<10} "
                      f"{np.mean(test_accs):6.2f} ± {np.std(test_accs):5.2f}   "
                      f"{np.mean(train_accs):6.2f}")

    print(f"\nExperiment 4 complete. {len(all_results)} runs saved to {OUTPUT_DIR}")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 4: Data Regime Sweep')
    parser.add_argument('--moduli', nargs='+', type=int, default=[97],
                        help='Moduli to test (97, 127)')
    parser.add_argument('--random_seeds', nargs='+', type=int, default=RANDOM_SEEDS)
    args = parser.parse_args()

    run_exp4(moduli=args.moduli, random_seeds=args.random_seeds)
