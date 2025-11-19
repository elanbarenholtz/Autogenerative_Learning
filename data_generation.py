"""
Generate Fibonacci sequences from various seeds for training and testing.
"""
import json
import numpy as np
from typing import List, Tuple, Dict


def generate_fibonacci_sequence(seed: Tuple[int, int], length: int) -> List[int]:
    """
    Generate a Fibonacci sequence from a given seed.

    Args:
        seed: Tuple of (F_0, F_1) initial values
        length: Total number of elements to generate

    Returns:
        List of Fibonacci numbers
    """
    if length <= 0:
        return []
    if length == 1:
        return [seed[0]]

    sequence = [seed[0], seed[1]]
    for i in range(2, length):
        sequence.append(sequence[i-1] + sequence[i-2])

    return sequence


def create_sliding_window_examples(
    sequence: List[int],
    context_window: int = 10
) -> List[Tuple[List[int], int]]:
    """
    Create training examples using sliding window approach.

    Args:
        sequence: Full Fibonacci sequence
        context_window: Number of previous tokens to use as context

    Returns:
        List of (context, target) pairs
    """
    examples = []
    for i in range(context_window, len(sequence)):
        context = sequence[i-context_window:i]
        target = sequence[i]
        examples.append((context, target))

    return examples


def generate_training_data(
    seeds: List[Tuple[int, int]],
    sequence_length: int = 30,
    context_window: int = 10,
    max_value: int = 10000
) -> Tuple[List[Tuple[List[int], int]], Dict]:
    """
    Generate all training examples from multiple seeds.

    Args:
        seeds: List of seed tuples
        sequence_length: Length of sequence to generate per seed
        context_window: Size of context window
        max_value: Maximum value to include (for vocabulary management)

    Returns:
        Tuple of (training_examples, metadata)
    """
    all_examples = []
    sequences = []
    all_numbers = set()

    for seed in seeds:
        # Generate sequence
        sequence = generate_fibonacci_sequence(seed, sequence_length)

        # Filter if values exceed max_value
        valid_length = len(sequence)
        for i, val in enumerate(sequence):
            if val > max_value:
                valid_length = i
                break

        sequence = sequence[:valid_length]
        sequences.append({
            'seed': seed,
            'sequence': sequence,
            'length': len(sequence)
        })

        # Create training examples
        examples = create_sliding_window_examples(sequence, context_window)
        all_examples.extend(examples)

        # Track vocabulary
        all_numbers.update(sequence)

    metadata = {
        'num_seeds': len(seeds),
        'sequence_length': sequence_length,
        'context_window': context_window,
        'total_examples': len(all_examples),
        'vocabulary_size': len(all_numbers),
        'max_value': max(all_numbers) if all_numbers else 0,
        'sequences': sequences
    }

    return all_examples, metadata


def save_data(examples: List[Tuple[List[int], int]], metadata: Dict,
              output_prefix: str):
    """Save training data and metadata to files."""

    # Save examples
    examples_data = {
        'examples': [(ctx, tgt) for ctx, tgt in examples]
    }
    with open(f'{output_prefix}_examples.json', 'w') as f:
        json.dump(examples_data, f, indent=2)

    # Save metadata
    with open(f'{output_prefix}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(examples)} examples to {output_prefix}_examples.json")
    print(f"Vocabulary size: {metadata['vocabulary_size']}")
    print(f"Max value: {metadata['max_value']}")


if __name__ == "__main__":
    # Training seeds as specified
    training_seeds = [
        (0, 1), (1, 1), (2, 3), (1, 2), (5, 8),
        (3, 5), (7, 11), (2, 5), (4, 7), (6, 10),
        (1, 3), (3, 7), (4, 6), (8, 13), (5, 12),
        (9, 14), (11, 18), (7, 12), (10, 16), (12, 19)
    ]

    # Test seeds (novel, never seen in training)
    test_seeds_novel = [
        (17, 29), (23, 37), (13, 21), (31, 50), (19, 31)
    ]

    # Generate training data
    print("Generating training data...")
    train_examples, train_metadata = generate_training_data(
        training_seeds,
        sequence_length=30,
        context_window=10,
        max_value=10000
    )
    save_data(train_examples, train_metadata, 'data/train')

    # Generate test data (novel seeds, same length)
    print("\nGenerating test data (novel seeds)...")
    test_examples, test_metadata = generate_training_data(
        test_seeds_novel,
        sequence_length=30,
        context_window=10,
        max_value=10000
    )
    save_data(test_examples, test_metadata, 'data/test_novel')

    # Generate extended sequences from training seeds for length generalization test
    print("\nGenerating test data (extended sequences)...")
    extended_seeds = [training_seeds[0], training_seeds[4], training_seeds[10]]  # 3 seeds
    extended_examples, extended_metadata = generate_training_data(
        extended_seeds,
        sequence_length=50,
        context_window=10,
        max_value=10000
    )
    save_data(extended_examples, extended_metadata, 'data/test_extended')

    print("\n=== Data Generation Complete ===")
    print(f"Training examples: {len(train_examples)}")
    print(f"Test examples (novel seeds): {len(test_examples)}")
    print(f"Test examples (extended): {len(extended_examples)}")
