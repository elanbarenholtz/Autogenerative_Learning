"""
Evaluation script for digit-wise models (Experiment 2A).

Tests:
1. Individual prediction: Given correct context, predict next number once
2. Sequential generation: Autoregressive generation using own predictions
"""
import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple

from model import FibonacciTransformer
from digit_tokenizer import DigitTokenizer
from recurrence_relations import get_relation


def predict_next_number_digit(model, tokenizer, context: List[int], device, max_digits=10):
    """
    Predict the next number using digit-wise generation.

    Autoregressively generates digits until SEP token or max_digits reached.
    """
    model.eval()

    # Encode context
    context_ids = tokenizer.encode_sequence(context)
    input_seq = torch.tensor([context_ids], dtype=torch.long).to(device)

    predicted_digits = []

    with torch.no_grad():
        for _ in range(max_digits):
            # Get logits for next token
            logits = model(input_seq)  # [1, seq_len, vocab_size]
            next_token_logits = logits[0, -1, :]  # [vocab_size]

            # Predict next token
            predicted_id = next_token_logits.argmax().item()
            predicted_token = tokenizer.id_to_token[predicted_id]

            # Stop if we hit SEP or PAD
            if predicted_token in [tokenizer.sep_token, tokenizer.pad_token]:
                break

            # If it's a digit, add it
            if predicted_token.startswith('[DIGIT_'):
                predicted_digits.append(predicted_token)
                # Append to input for next prediction
                input_seq = torch.cat([
                    input_seq,
                    torch.tensor([[predicted_id]], dtype=torch.long).to(device)
                ], dim=1)
            else:
                break

    # Convert digits to number
    if not predicted_digits:
        return 0

    predicted_number = tokenizer.tokens_to_number(predicted_digits)
    return predicted_number


def evaluate_individual_predictions(model, tokenizer, test_seeds: List[Tuple[int, int]],
                                     relation_name: str, device, sequence_length: int = 30):
    """
    Test individual predictions: Given correct context, predict next number once.

    This is the easier test - no error accumulation.
    """
    relation = get_relation(relation_name)

    all_correct = 0
    all_total = 0
    all_errors = []

    results_by_seed = []

    for seed in test_seeds:
        sequence = relation.generate_sequence(seed, sequence_length)

        correct = 0
        total = 0
        errors = []
        predictions = []
        targets = []

        # Test predictions from position 10 onwards
        context_window = 10
        for i in range(context_window, len(sequence)):
            context = sequence[i-context_window:i]
            target = sequence[i]

            # Predict
            predicted = predict_next_number_digit(model, tokenizer, context, device)

            predictions.append(predicted)
            targets.append(target)

            if predicted == target:
                correct += 1

            errors.append(abs(predicted - target))
            total += 1

        accuracy = 100.0 * correct / total if total > 0 else 0
        mae = np.mean(errors) if errors else 0

        results_by_seed.append({
            'seed': seed,
            'accuracy': accuracy,
            'mae': mae,
            'correct': correct,
            'total': total,
            'predictions': predictions[:20],  # Store first 20
            'targets': targets[:20]
        })

        all_correct += correct
        all_total += total
        all_errors.extend(errors)

    overall_accuracy = 100.0 * all_correct / all_total if all_total > 0 else 0
    overall_mae = np.mean(all_errors) if all_errors else 0

    return {
        'overall_accuracy': overall_accuracy,
        'overall_mae': overall_mae,
        'by_seed': results_by_seed,
        'total_predictions': all_total
    }


def evaluate_sequential_generation(model, tokenizer, test_seeds: List[Tuple[int, int]],
                                    relation_name: str, device, sequence_length: int = 30):
    """
    Test sequential generation: Use model's own predictions as context.

    This is the harder test - errors compound.
    """
    relation = get_relation(relation_name)

    all_correct = 0
    all_total = 0
    all_errors = []

    results_by_seed = []

    for seed in test_seeds:
        true_sequence = relation.generate_sequence(seed, sequence_length)

        # Start with initial context
        context_window = 10
        context = true_sequence[:context_window]

        predictions = []
        errors = []

        # Generate positions 10 onwards
        for i in range(context_window, len(true_sequence)):
            target = true_sequence[i]

            # Predict next number
            predicted = predict_next_number_digit(model, tokenizer, context, device)

            predictions.append(predicted)
            error = abs(predicted - target)
            errors.append(error)

            if predicted == target:
                all_correct += 1
            all_total += 1

            # Update context with TRUE value (teacher forcing between predictions)
            context = context[1:] + [target]

        accuracy = 100.0 * sum(1 for i, p in enumerate(predictions)
                               if p == true_sequence[context_window + i]) / len(predictions)
        mae = np.mean(errors)

        results_by_seed.append({
            'seed': seed,
            'accuracy': accuracy,
            'mae': mae,
            'predictions': predictions[:20],
            'targets': true_sequence[context_window:context_window+20],
            'errors': errors[:20]
        })

        all_errors.extend(errors)

    overall_accuracy = 100.0 * all_correct / all_total if all_total > 0 else 0
    overall_mae = np.mean(all_errors) if all_errors else 0

    return {
        'overall_accuracy': overall_accuracy,
        'overall_mae': overall_mae,
        'by_seed': results_by_seed,
        'total_predictions': all_total
    }


def evaluate_digit_model(relation_name: str, test_type: str = 'both'):
    """
    Evaluate a trained digit-wise model.

    Args:
        relation_name: Name of relation
        test_type: 'individual', 'sequential', or 'both'
    """
    print(f"\n{'='*70}")
    print(f"Evaluating Digit-Wise Model: {relation_name}")
    print(f"{'='*70}")

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_dir = Path(f'experiments_digit/{relation_name}')

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(exp_dir / 'models' / 'best_model.pt',
                           map_location=device, weights_only=False)

    tokenizer = DigitTokenizer()

    model = FibonacciTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.0,
        max_seq_len=200  # Increased for longer digit sequences
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Training accuracy: {checkpoint['accuracy']:.2f}%")

    # Get test seeds
    relation = get_relation(relation_name)
    novel_seeds = relation.get_test_seeds()
    training_seeds = relation.get_training_seeds()[:5]  # Use subset of training seeds

    print(f"\nTest seeds:")
    print(f"  Training seeds: {len(training_seeds)}")
    print(f"  Novel seeds: {len(novel_seeds)}")

    results = {
        'relation': relation_name,
        'tokenization': 'digit-wise',
        'model_params': sum(p.numel() for p in model.parameters()),
        'vocab_size': tokenizer.vocab_size
    }

    # Individual prediction test
    if test_type in ['individual', 'both']:
        print("\n--- Individual Prediction Test ---")
        print("(Given correct context, predict next number once)")

        print("\n  Testing on training seeds...")
        train_individual = evaluate_individual_predictions(
            model, tokenizer, training_seeds, relation_name, device
        )

        print("\n  Testing on novel seeds...")
        novel_individual = evaluate_individual_predictions(
            model, tokenizer, novel_seeds, relation_name, device
        )

        results['individual'] = {
            'training_seeds': train_individual,
            'novel_seeds': novel_individual
        }

        print(f"\n  Results (Individual):")
        print(f"    Training seeds: {train_individual['overall_accuracy']:.2f}%")
        print(f"    Novel seeds:    {novel_individual['overall_accuracy']:.2f}%")

    # Sequential generation test
    if test_type in ['sequential', 'both']:
        print("\n--- Sequential Generation Test ---")
        print("(Use model's own predictions as context)")

        print("\n  Testing on training seeds...")
        train_sequential = evaluate_sequential_generation(
            model, tokenizer, training_seeds, relation_name, device
        )

        print("\n  Testing on novel seeds...")
        novel_sequential = evaluate_sequential_generation(
            model, tokenizer, novel_seeds, relation_name, device
        )

        results['sequential'] = {
            'training_seeds': train_sequential,
            'novel_seeds': novel_sequential
        }

        print(f"\n  Results (Sequential):")
        print(f"    Training seeds: {train_sequential['overall_accuracy']:.2f}%")
        print(f"    Novel seeds:    {novel_sequential['overall_accuracy']:.2f}%")

    # Save results
    results_path = exp_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")

    return results


def main():
    """Evaluate digit-wise model."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate digit-wise model')
    parser.add_argument('--relation', type=str, default='fibonacci',
                        choices=['fibonacci', 'linear', 'tribonacci', 'geometric', 'fibonacci_plus_constant'],
                        help='Relation to evaluate')
    parser.add_argument('--test-type', type=str, default='both',
                        choices=['individual', 'sequential', 'both'],
                        help='Type of evaluation to run')

    args = parser.parse_args()

    evaluate_digit_model(args.relation, test_type=args.test_type)


if __name__ == "__main__":
    main()
