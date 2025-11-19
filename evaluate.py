"""
Comprehensive evaluation of trained Fibonacci transformer.
Tests on novel seeds and extended sequences.
"""
import torch
import json
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm

from model import FibonacciTransformer, NumberTokenizer
from data_generation import generate_fibonacci_sequence


def evaluate_on_sequence(
    model: FibonacciTransformer,
    tokenizer: NumberTokenizer,
    seed: Tuple[int, int],
    sequence_length: int,
    context_window: int,
    device: str
) -> Dict:
    """
    Evaluate model on a single Fibonacci sequence.

    Args:
        model: Trained transformer model
        tokenizer: Number tokenizer
        seed: Fibonacci seed (F_0, F_1)
        sequence_length: Length of sequence to generate
        context_window: Size of context window
        device: Device to run on

    Returns:
        Dictionary with evaluation metrics
    """
    # Generate true Fibonacci sequence
    true_sequence = generate_fibonacci_sequence(seed, sequence_length)

    # Predictions storage
    predictions = []
    targets = []
    errors = []
    exact_matches = 0

    model.eval()

    # Use first context_window numbers as initial context
    context = true_sequence[:context_window]

    # Predict remaining numbers one by one
    for i in range(context_window, len(true_sequence)):
        target = true_sequence[i]

        # Predict next number
        with torch.no_grad():
            predicted = model.predict_next(context, tokenizer, temperature=0.0)

        predictions.append(predicted)
        targets.append(target)

        # Calculate error
        error = abs(predicted - target)
        errors.append(error)

        if predicted == target:
            exact_matches += 1

        # Update context (sliding window)
        context = context[1:] + [target]  # Use true value for next prediction

    # Calculate metrics
    total_predictions = len(predictions)
    exact_match_accuracy = 100.0 * exact_matches / total_predictions if total_predictions > 0 else 0
    mean_absolute_error = np.mean(errors) if errors else 0
    median_absolute_error = np.median(errors) if errors else 0
    max_error = max(errors) if errors else 0

    # Calculate relative errors (percentage)
    relative_errors = [100.0 * e / max(t, 1) for e, t in zip(errors, targets)]
    mean_relative_error = np.mean(relative_errors) if relative_errors else 0

    return {
        'seed': seed,
        'sequence_length': sequence_length,
        'total_predictions': total_predictions,
        'exact_matches': exact_matches,
        'exact_match_accuracy': exact_match_accuracy,
        'mean_absolute_error': mean_absolute_error,
        'median_absolute_error': median_absolute_error,
        'max_error': max_error,
        'mean_relative_error': mean_relative_error,
        'predictions': predictions,
        'targets': targets,
        'errors': errors,
        'true_sequence': true_sequence
    }


def evaluate_autoregressive_generation(
    model: FibonacciTransformer,
    tokenizer: NumberTokenizer,
    seed: Tuple[int, int],
    initial_context_length: int,
    generation_length: int,
    device: str
) -> Dict:
    """
    Evaluate model in fully autoregressive mode (using its own predictions).

    Args:
        model: Trained transformer model
        tokenizer: Number tokenizer
        seed: Fibonacci seed
        initial_context_length: Length of initial true context
        generation_length: Number of tokens to generate
        device: Device to run on

    Returns:
        Dictionary with generation results
    """
    # Generate true sequence for comparison
    true_sequence = generate_fibonacci_sequence(
        seed, initial_context_length + generation_length
    )

    # Start with initial context
    initial_context = true_sequence[:initial_context_length]

    # Generate sequence autoregressively
    generated_sequence = model.predict_sequence(
        initial_context,
        generation_length,
        tokenizer,
        temperature=0.0
    )

    # Calculate errors for generated part
    errors = []
    for i in range(initial_context_length, len(generated_sequence)):
        if i < len(true_sequence):
            error = abs(generated_sequence[i] - true_sequence[i])
            errors.append(error)

    return {
        'seed': seed,
        'initial_context': initial_context,
        'generated_sequence': generated_sequence,
        'true_sequence': true_sequence,
        'errors': errors,
        'mean_error': np.mean(errors) if errors else 0
    }


def main():
    print("=== Fibonacci Transformer Evaluation ===\n")

    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    context_window = 10

    # Load trained model
    print("Loading trained model...")
    checkpoint = torch.load('models/best_model.pt', map_location=device)

    vocabulary = checkpoint['vocabulary']
    tokenizer = NumberTokenizer(vocabulary)

    model = FibonacciTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=checkpoint['config']['d_model'],
        nhead=checkpoint['config']['nhead'],
        num_layers=checkpoint['config']['num_layers'],
        dim_feedforward=checkpoint['config']['dim_feedforward'],
        dropout=0.0  # No dropout during evaluation
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Training accuracy: {checkpoint['accuracy']:.2f}%\n")

    # ===== TEST 1: Novel Seeds (Same Length) =====
    print("=" * 60)
    print("TEST 1: Novel Seeds (Never Seen During Training)")
    print("=" * 60)

    novel_seeds = [
        (17, 29), (23, 37), (13, 21), (31, 50), (19, 31)
    ]

    novel_results = []
    for seed in tqdm(novel_seeds, desc="Evaluating novel seeds"):
        result = evaluate_on_sequence(
            model, tokenizer, seed, 30, context_window, device
        )
        novel_results.append(result)

        print(f"\nSeed {seed}:")
        print(f"  Exact Match Accuracy: {result['exact_match_accuracy']:.2f}%")
        print(f"  Mean Absolute Error: {result['mean_absolute_error']:.2f}")
        print(f"  Mean Relative Error: {result['mean_relative_error']:.2f}%")

    # Aggregate statistics for novel seeds
    novel_avg_accuracy = np.mean([r['exact_match_accuracy'] for r in novel_results])
    novel_avg_mae = np.mean([r['mean_absolute_error'] for r in novel_results])

    print(f"\n{'='*60}")
    print(f"NOVEL SEEDS - AGGREGATE RESULTS:")
    print(f"  Average Exact Match Accuracy: {novel_avg_accuracy:.2f}%")
    print(f"  Average Mean Absolute Error: {novel_avg_mae:.2f}")
    print(f"{'='*60}\n")

    # ===== TEST 2: Training Seeds (Baseline) =====
    print("=" * 60)
    print("TEST 2: Training Seeds (Baseline Performance)")
    print("=" * 60)

    training_seeds = [
        (0, 1), (1, 1), (2, 3), (1, 2), (5, 8)
    ]

    training_results = []
    for seed in tqdm(training_seeds, desc="Evaluating training seeds"):
        result = evaluate_on_sequence(
            model, tokenizer, seed, 30, context_window, device
        )
        training_results.append(result)

        print(f"\nSeed {seed}:")
        print(f"  Exact Match Accuracy: {result['exact_match_accuracy']:.2f}%")
        print(f"  Mean Absolute Error: {result['mean_absolute_error']:.2f}")

    # Aggregate statistics for training seeds
    training_avg_accuracy = np.mean([r['exact_match_accuracy'] for r in training_results])
    training_avg_mae = np.mean([r['mean_absolute_error'] for r in training_results])

    print(f"\n{'='*60}")
    print(f"TRAINING SEEDS - AGGREGATE RESULTS:")
    print(f"  Average Exact Match Accuracy: {training_avg_accuracy:.2f}%")
    print(f"  Average Mean Absolute Error: {training_avg_mae:.2f}")
    print(f"{'='*60}\n")

    # ===== TEST 3: Extended Sequences =====
    print("=" * 60)
    print("TEST 3: Extended Sequences (Length Generalization)")
    print("=" * 60)

    extended_seeds = [(0, 1), (1, 1), (2, 3)]

    extended_results = []
    for seed in tqdm(extended_seeds, desc="Evaluating extended sequences"):
        result = evaluate_on_sequence(
            model, tokenizer, seed, 50, context_window, device
        )
        extended_results.append(result)

        print(f"\nSeed {seed} (length 50):")
        print(f"  Exact Match Accuracy: {result['exact_match_accuracy']:.2f}%")
        print(f"  Mean Absolute Error: {result['mean_absolute_error']:.2f}")

    # ===== TEST 4: Autoregressive Generation =====
    print("\n" + "=" * 60)
    print("TEST 4: Autoregressive Generation (Using Own Predictions)")
    print("=" * 60)

    autoregressive_results = []
    test_seeds = [(0, 1), (17, 29)]  # One training, one novel

    for seed in test_seeds:
        result = evaluate_autoregressive_generation(
            model, tokenizer, seed,
            initial_context_length=10,
            generation_length=20,
            device=device
        )
        autoregressive_results.append(result)

        print(f"\nSeed {seed}:")
        print(f"  Initial context: {result['initial_context']}")
        print(f"  Generated: {result['generated_sequence'][10:20]}")
        print(f"  True:      {result['true_sequence'][10:20]}")
        print(f"  Mean Error: {result['mean_error']:.2f}")

    # ===== SAVE RESULTS =====
    print("\n" + "=" * 60)
    print("Saving evaluation results...")

    results_summary = {
        'novel_seeds': {
            'individual_results': [
                {k: v for k, v in r.items() if k not in ['predictions', 'targets', 'errors', 'true_sequence']}
                for r in novel_results
            ],
            'average_accuracy': float(novel_avg_accuracy),
            'average_mae': float(novel_avg_mae)
        },
        'training_seeds': {
            'individual_results': [
                {k: v for k, v in r.items() if k not in ['predictions', 'targets', 'errors', 'true_sequence']}
                for r in training_results
            ],
            'average_accuracy': float(training_avg_accuracy),
            'average_mae': float(training_avg_mae)
        },
        'extended_sequences': [
            {k: v for k, v in r.items() if k not in ['predictions', 'targets', 'errors', 'true_sequence']}
            for r in extended_results
        ],
        'autoregressive_generation': autoregressive_results
    }

    with open('results/evaluation_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Save detailed results for visualization
    detailed_results = {
        'novel_seeds': novel_results,
        'training_seeds': training_results,
        'extended_sequences': extended_results
    }

    torch.save(detailed_results, 'results/detailed_results.pt')

    print("Results saved to results/evaluation_summary.json")
    print("Detailed results saved to results/detailed_results.pt")

    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Training Seeds Accuracy:  {training_avg_accuracy:.2f}%")
    print(f"Novel Seeds Accuracy:     {novel_avg_accuracy:.2f}%")
    print(f"Degradation:              {training_avg_accuracy - novel_avg_accuracy:.2f} percentage points")
    print(f"\nTraining Seeds MAE:       {training_avg_mae:.2f}")
    print(f"Novel Seeds MAE:          {novel_avg_mae:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
