"""
Training script for Fibonacci transformer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import numpy as np

from model import FibonacciTransformer, NumberTokenizer, count_parameters


class FibonacciDataset(Dataset):
    """Dataset for Fibonacci sequence prediction."""

    def __init__(self, examples: List[Tuple[List[int], int]]):
        """
        Args:
            examples: List of (context, target) pairs
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        context, target = self.examples[idx]
        return context, target


def collate_fn(batch, tokenizer):
    """
    Custom collate function to handle variable length sequences.

    Args:
        batch: List of (context, target) pairs
        tokenizer: NumberTokenizer instance

    Returns:
        Tuple of (input_tensor, target_tensor)
    """
    contexts, targets = zip(*batch)

    # Encode contexts
    context_ids = [tokenizer.encode(ctx) for ctx in contexts]

    # Pad contexts to same length
    max_len = max(len(ctx) for ctx in context_ids)
    padded_contexts = []
    for ctx in context_ids:
        padded = ctx + [tokenizer.pad_id] * (max_len - len(ctx))
        padded_contexts.append(padded)

    # Encode targets
    target_ids = [tokenizer.encode([tgt])[0] for tgt in targets]

    return (
        torch.tensor(padded_contexts, dtype=torch.long),
        torch.tensor(target_ids, dtype=torch.long)
    )


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, (contexts, targets) in enumerate(progress_bar):
        contexts = contexts.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(contexts)

        # Get predictions for last position
        logits_last = logits[:, -1, :]

        # Compute loss
        loss = criterion(logits_last, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        predictions = logits_last.argmax(dim=-1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, tokenizer):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for contexts, targets in tqdm(dataloader, desc="Evaluating"):
            contexts = contexts.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(contexts)
            logits_last = logits[:, -1, :]

            # Compute loss
            loss = criterion(logits_last, targets)

            # Track metrics
            total_loss += loss.item()
            predictions = logits_last.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            # Store for detailed analysis
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    # Decode predictions and targets to actual numbers
    pred_numbers = tokenizer.decode(all_predictions)
    target_numbers = tokenizer.decode(all_targets)

    # Compute mean absolute error
    mae = np.mean([abs(p - t) for p, t in zip(pred_numbers, target_numbers)])

    return avg_loss, accuracy, mae, (pred_numbers, target_numbers)


def plot_training_curves(train_losses, train_accs, output_path='results/training_curves.png'):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy curve
    ax2.plot(train_accs, label='Train Accuracy', marker='o', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_path}")


def main():
    # Hyperparameters
    config = {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'd_model': 128,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42
    }

    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    print("=== Fibonacci Transformer Training ===")
    print(f"Device: {config['device']}")
    print(f"Configuration: {json.dumps(config, indent=2)}")

    # Load training data
    print("\nLoading training data...")
    with open('data/train_examples.json', 'r') as f:
        train_data = json.load(f)
    train_examples = [tuple(ex) for ex in train_data['examples']]

    with open('data/train_metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"Loaded {len(train_examples)} training examples")
    print(f"Vocabulary size: {metadata['vocabulary_size']}")

    # Build vocabulary from metadata
    vocabulary = set()
    for seq_info in metadata['sequences']:
        vocabulary.update(seq_info['sequence'])
    vocabulary = sorted(list(vocabulary))

    # Create tokenizer
    tokenizer = NumberTokenizer(vocabulary)
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    # Create datasets and dataloaders
    train_dataset = FibonacciDataset(train_examples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    # Create model
    print("\nInitializing model...")
    model = FibonacciTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    )

    model = model.to(config['device'])
    print(f"Model parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    print("\nStarting training...")
    train_losses = []
    train_accs = []
    best_loss = float('inf')

    for epoch in range(config['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['epochs']} ---")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, config['device']
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(train_loss)

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'accuracy': train_acc,
                'config': config,
                'vocabulary': vocabulary
            }
            torch.save(checkpoint, 'models/best_model.pt')
            print(f"✓ Saved best model (loss: {best_loss:.4f})")

        # Save periodic checkpoints
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'accuracy': train_acc,
                'config': config,
                'vocabulary': vocabulary
            }
            torch.save(checkpoint, f'models/checkpoint_epoch_{epoch+1}.pt')

    # Save final model
    checkpoint = {
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_losses[-1],
        'accuracy': train_accs[-1],
        'config': config,
        'vocabulary': vocabulary
    }
    torch.save(checkpoint, 'models/final_model.pt')

    # Plot training curves
    plot_training_curves(train_losses, train_accs)

    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accs,
        'config': config
    }
    with open('results/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n=== Training Complete ===")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final accuracy: {train_accs[-1]:.2f}%")


if __name__ == "__main__":
    main()
