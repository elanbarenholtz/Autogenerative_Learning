"""
Run experiment for a specific recurrence relation.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path

from recurrence_relations import get_relation
from model import FibonacciTransformer, NumberTokenizer, count_parameters
from train import FibonacciDataset, collate_fn


def generate_data_for_relation(relation_name: str, sequence_length: int = 25):
    """Generate training and test data for a specific relation."""
    relation = get_relation(relation_name)

    print(f"\n{'='*70}")
    print(f"Generating data for: {relation.name}")
    print(f"{'='*70}")

    # Create output directory
    data_dir = Path(f'experiments/{relation_name}/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate training data
    train_seeds = relation.get_training_seeds()
    train_examples = []
    all_numbers = set()

    max_value = 100000  # Vocabulary limit

    for seed in train_seeds:
        sequence = relation.generate_sequence(seed, sequence_length)

        # Truncate if exceeds max value
        valid_length = len(sequence)
        for i, val in enumerate(sequence):
            if val > max_value:
                valid_length = i
                break
        sequence = sequence[:valid_length]

        # Create sliding window examples
        context_window = relation.order + 8  # Give extra context
        for i in range(context_window, len(sequence)):
            context = sequence[i-context_window:i]
            target = sequence[i]
            train_examples.append((context, target))

        all_numbers.update(sequence)

    # Save training data
    with open(data_dir / 'train_examples.json', 'w') as f:
        json.dump({'examples': train_examples}, f)

    print(f"Training examples: {len(train_examples)}")
    print(f"Vocabulary size: {len(all_numbers)}")

    # Generate test data
    test_seeds = relation.get_test_seeds()
    test_examples = []

    for seed in test_seeds:
        sequence = relation.generate_sequence(seed, sequence_length)

        valid_length = len(sequence)
        for i, val in enumerate(sequence):
            if val > max_value:
                valid_length = i
                break
        sequence = sequence[:valid_length]

        context_window = relation.order + 8
        for i in range(context_window, len(sequence)):
            context = sequence[i-context_window:i]
            target = sequence[i]
            test_examples.append((context, target))

    with open(data_dir / 'test_examples.json', 'w') as f:
        json.dump({'examples': test_examples}, f)

    print(f"Test examples: {len(test_examples)}")

    # Save vocabulary
    vocabulary = sorted(list(all_numbers))
    with open(data_dir / 'vocabulary.json', 'w') as f:
        json.dump({'vocabulary': vocabulary}, f)

    return vocabulary, train_examples, test_examples


def train_model_on_relation(relation_name: str, epochs: int = 50):
    """Train a model on a specific relation."""
    relation = get_relation(relation_name)

    print(f"\n{'='*70}")
    print(f"Training model for: {relation.name}")
    print(f"{'='*70}")

    # Setup directories
    exp_dir = Path(f'experiments/{relation_name}')
    model_dir = exp_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(exp_dir / 'data' / 'train_examples.json', 'r') as f:
        train_data = json.load(f)
    train_examples = [tuple(ex) for ex in train_data['examples']]

    with open(exp_dir / 'data' / 'vocabulary.json', 'r') as f:
        vocabulary = json.load(f)['vocabulary']

    # Create tokenizer
    tokenizer = NumberTokenizer(vocabulary)

    # Create dataset
    train_dataset = FibonacciDataset(train_examples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FibonacciTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    best_loss = float('inf')
    history = {'losses': [], 'accuracies': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for contexts, targets in train_loader:
            contexts = contexts.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(contexts)
            logits_last = logits[:, -1, :]
            loss = criterion(logits_last, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = logits_last.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        history['losses'].append(avg_loss)
        history['accuracies'].append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
                'vocabulary': vocabulary
            }
            torch.save(checkpoint, model_dir / 'best_model.pt')

    # Save history
    with open(exp_dir / 'training_history.json', 'w') as f:
        json.dump(history, f)

    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"Final accuracy: {history['accuracies'][-1]:.2f}%")

    return history


def evaluate_model_on_relation(relation_name: str):
    """Evaluate model on test seeds."""
    relation = get_relation(relation_name)

    print(f"\n{'='*70}")
    print(f"Evaluating model for: {relation.name}")
    print(f"{'='*70}")

    exp_dir = Path(f'experiments/{relation_name}')

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(exp_dir / 'models' / 'best_model.pt', map_location=device)

    vocabulary = checkpoint['vocabulary']
    tokenizer = NumberTokenizer(vocabulary)

    model = FibonacciTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.0
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data
    with open(exp_dir / 'data' / 'test_examples.json', 'r') as f:
        test_data = json.load(f)
    test_examples = [tuple(ex) for ex in test_data['examples']]

    # Evaluate
    correct = 0
    total = 0
    errors = []

    with torch.no_grad():
        for context, target in test_examples:
            src = torch.tensor([tokenizer.encode(context)], dtype=torch.long).to(device)
            logits = model(src)
            predicted_id = logits[0, -1, :].argmax().item()
            predicted = tokenizer.id_to_token[predicted_id]

            if predicted == target:
                correct += 1
            errors.append(abs(predicted - target))
            total += 1

    accuracy = 100.0 * correct / total
    mae = np.mean(errors)

    print(f"\nTest Results:")
    print(f"  Exact Match Accuracy: {accuracy:.2f}%")
    print(f"  Mean Absolute Error: {mae:.2f}")

    # Load training data for comparison
    with open(exp_dir / 'data' / 'train_examples.json', 'r') as f:
        train_data = json.load(f)
    train_examples = [tuple(ex) for ex in train_data['examples']]

    # Evaluate on training data (for comparison)
    train_correct = 0
    train_total = 0
    train_errors = []

    with torch.no_grad():
        for context, target in train_examples[:len(test_examples)]:  # Sample same size
            src = torch.tensor([tokenizer.encode(context)], dtype=torch.long).to(device)
            logits = model(src)
            predicted_id = logits[0, -1, :].argmax().item()
            predicted = tokenizer.id_to_token[predicted_id]

            if predicted == target:
                train_correct += 1
            train_errors.append(abs(predicted - target))
            train_total += 1

    train_accuracy = 100.0 * train_correct / train_total
    train_mae = np.mean(train_errors)

    print(f"\nTraining Seeds (for comparison):")
    print(f"  Exact Match Accuracy: {train_accuracy:.2f}%")
    print(f"  Mean Absolute Error: {train_mae:.2f}")

    # Save results
    results = {
        'relation_name': relation.name,
        'test_accuracy': accuracy,
        'test_mae': mae,
        'train_accuracy': train_accuracy,
        'train_mae': train_mae,
        'degradation': train_accuracy - accuracy
    }

    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_relation_experiment.py <relation_name>")
        print(f"Available relations: fibonacci, linear, tribonacci, geometric, fibonacci_plus_constant")
        sys.exit(1)

    relation_name = sys.argv[1]

    # Run complete experiment
    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT: {relation_name}")
    print(f"{'='*70}")

    # 1. Generate data
    vocabulary, train_examples, test_examples = generate_data_for_relation(relation_name)

    # 2. Train model
    history = train_model_on_relation(relation_name, epochs=50)

    # 3. Evaluate
    results = evaluate_model_on_relation(relation_name)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE: {relation_name}")
    print(f"{'='*70}")
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
    print(f"Training Accuracy: {results['train_accuracy']:.2f}%")
    print(f"Degradation: {results['degradation']:.2f} pp")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
