"""
Data generation for digit-wise encoding (Experiment 2A).

Uses the same sequence generation as original, but encodes as digits.
"""
import json
import torch
from typing import List, Tuple
from pathlib import Path
from digit_tokenizer import DigitTokenizer
from recurrence_relations import get_relation

def create_digit_training_data(relation_name: str, sequence_length: int = 30):
    """
    Generate training data for a relation using digit-wise encoding.

    Args:
        relation_name: Which recurrence relation to use
        sequence_length: How long to generate sequences

    Returns:
        examples: List of (context_ids, target_ids) pairs
        tokenizer: The digit tokenizer used
        metadata: Information about the data
    """
    print(f"\n{'='*70}")
    print(f"Generating Digit-Wise Training Data: {relation_name}")
    print(f"{'='*70}")

    # Get relation and tokenizer
    relation = get_relation(relation_name)
    tokenizer = DigitTokenizer()

    # Get training seeds
    train_seeds = relation.get_training_seeds()
    print(f"\nTraining seeds: {len(train_seeds)}")

    # Generate sequences and create examples
    examples = []
    sequences_metadata = []

    context_window = 10  # Still 10 numbers (but many more tokens)

    for seed in train_seeds:
        # Generate sequence
        sequence = relation.generate_sequence(seed, sequence_length)

        # Create sliding window examples
        for i in range(context_window, len(sequence)):
            context = sequence[i-context_window:i]
            target = sequence[i]

            # Encode using digit tokenizer
            context_ids, target_ids = tokenizer.encode_for_training(context, target)

            examples.append((context_ids, target_ids))

        # Store metadata
        sequences_metadata.append({
            'seed': seed,
            'length': len(sequence),
            'sequence': sequence[:min(len(sequence), 20)]  # Store first 20 terms
        })

    print(f"\nGenerated {len(examples)} training examples")
    print(f"Vocabulary size: {tokenizer.vocab_size} tokens")
    print(f"Sample context length: ~{len(examples[0][0])} tokens (for 10 numbers)")
    print(f"Sample target length: ~{len(examples[0][1])} tokens")

    # Create metadata
    metadata = {
        'relation_name': relation.name,
        'num_seeds': len(train_seeds),
        'sequence_length': sequence_length,
        'context_window': context_window,
        'total_examples': len(examples),
        'vocab_size': tokenizer.vocab_size,
        'tokenization': 'digit-wise',
        'sequences': sequences_metadata
    }

    return examples, tokenizer, metadata


def save_digit_training_data(relation_name: str, output_dir: str = 'experiments_digit'):
    """
    Generate and save digit-wise training data for a relation.
    """
    # Create output directory
    data_dir = Path(output_dir) / relation_name / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    examples, tokenizer, metadata = create_digit_training_data(relation_name)

    # Save examples
    # Note: Save as lists since they're variable length
    examples_data = {
        'examples': [(ctx, tgt) for ctx, tgt in examples]
    }

    with open(data_dir / 'train_examples.json', 'w') as f:
        json.dump(examples_data, f)

    # Save metadata
    with open(data_dir / 'train_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save tokenizer config
    tokenizer_config = {
        'type': 'digit',
        'vocab_size': tokenizer.vocab_size,
        'vocab': tokenizer.vocab,
        'pad_token': tokenizer.pad_token,
        'sep_token': tokenizer.sep_token
    }
    with open(data_dir / 'tokenizer_config.json', 'w') as f:
        json.dump(tokenizer_config, f, indent=2)

    print(f"\n✓ Saved training data to {data_dir}")
    print(f"  - train_examples.json: {len(examples)} examples")
    print(f"  - train_metadata.json: Metadata")
    print(f"  - tokenizer_config.json: Tokenizer configuration")

    return examples, tokenizer, metadata


def create_digit_test_data(relation_name: str, test_seeds: List[Tuple[int, int]],
                            sequence_length: int = 30):
    """
    Generate test data for novel seeds using digit-wise encoding.
    """
    relation = get_relation(relation_name)
    tokenizer = DigitTokenizer()

    examples = []
    sequences_metadata = []

    context_window = 10

    for seed in test_seeds:
        sequence = relation.generate_sequence(seed, sequence_length)

        # Create examples
        for i in range(context_window, len(sequence)):
            context = sequence[i-context_window:i]
            target = sequence[i]

            context_ids, target_ids = tokenizer.encode_for_training(context, target)
            examples.append((context_ids, target_ids))

        sequences_metadata.append({
            'seed': seed,
            'length': len(sequence),
            'sequence': sequence[:min(len(sequence), 20)]
        })

    metadata = {
        'relation_name': relation.name,
        'num_seeds': len(test_seeds),
        'sequence_length': sequence_length,
        'context_window': context_window,
        'total_examples': len(examples),
        'vocab_size': tokenizer.vocab_size,
        'tokenization': 'digit-wise',
        'sequences': sequences_metadata
    }

    return examples, tokenizer, metadata


def main():
    """Generate digit-wise training data for all relations."""
    relations = ['fibonacci', 'linear', 'tribonacci', 'geometric', 'fibonacci_plus_constant']

    print("="*70)
    print("GENERATING DIGIT-WISE TRAINING DATA FOR ALL RELATIONS")
    print("="*70)

    for relation_name in relations:
        save_digit_training_data(relation_name)
        print()

    print("\n" + "="*70)
    print("ALL DIGIT-WISE TRAINING DATA GENERATED")
    print("="*70)


if __name__ == "__main__":
    main()
