"""
Training script for digit-wise encoding (Experiment 2A).

Key difference: Targets are now sequences of digit tokens, not single number tokens.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

from model import FibonacciTransformer  # Same architecture!
from digit_tokenizer import DigitTokenizer


class DigitFibonacciDataset(Dataset):
    """Dataset for digit-wise Fibonacci training."""

    def __init__(self, examples):
        """
        Args:
            examples: List of (context_ids, target_ids) tuples
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        context_ids, target_ids = self.examples[idx]
        return (
            torch.tensor(context_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long)
        )


def collate_digit_fn(batch):
    """
    Collate function for digit-wise batches.

    Handles variable-length contexts and targets.
    """
    contexts, targets = zip(*batch)

    # Pad contexts to max length in batch
    contexts_padded = pad_sequence(contexts, batch_first=True, padding_value=0)  # PAD token is 0

    # Pad targets to max length in batch
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)

    return contexts_padded, targets_padded


def train_epoch_digit(model, dataloader, optimizer, criterion, device, tokenizer):
    """Train for one epoch with digit-wise encoding."""
    model.train()
    total_loss = 0
    correct_numbers = 0
    total_numbers = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for contexts, targets in progress_bar:
        contexts = contexts.to(device)
        targets = targets.to(device)

        batch_size = contexts.size(0)
        target_len = targets.size(1)

        optimizer.zero_grad()

        # Autoregressive training: Concatenate context + target, predict shifted
        # Input: [context..., target_0, target_1, ...]
        # Predict: [target_0, target_1, target_2, ...]

        # Create input sequence: context + targets (shifted by 1)
        full_sequence = torch.cat([contexts, targets], dim=1)  # [batch, ctx_len + tgt_len]

        # Forward pass
        logits = model(full_sequence)  # [batch, seq_len, vocab_size]

        # Get predictions for target positions
        # We want to predict target tokens from positions [ctx_len : ctx_len+tgt_len]
        context_len = contexts.size(1)
        pred_logits = logits[:, context_len-1:context_len+target_len-1, :]  # Shift by 1

        # Compute loss
        # Reshape for cross-entropy: [batch*target_len, vocab_size] and [batch*target_len]
        pred_logits_flat = pred_logits.reshape(-1, pred_logits.size(-1))
        targets_flat = targets.reshape(-1)

        # Mask out padding (token ID 0)
        mask = targets_flat != 0
        if mask.sum() > 0:
            loss = criterion(pred_logits_flat[mask], targets_flat[mask])
        else:
            loss = torch.tensor(0.0, device=device)

        # Backward pass
        if loss.item() > 0:
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate accuracy (entire number correct)
        with torch.no_grad():
            for b in range(batch_size):
                # Get predicted digits for this example
                pred_digits = []
                for i in range(target_len):
                    if targets[b, i] == 0:  # PAD
                        break
                    pred_id = pred_logits[b, i].argmax().item()
                    pred_digits.append(pred_id)

                # Get target digits
                target_digits = targets[b][targets[b] != 0].tolist()

                # Check if entire number matches
                if pred_digits == target_digits:
                    correct_numbers += 1
                total_numbers += 1

        # Update progress bar
        acc = 100.0 * correct_numbers / total_numbers if total_numbers > 0 else 0
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.2f}%'
        })

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100.0 * correct_numbers / total_numbers if total_numbers > 0 else 0

    return avg_loss, accuracy


def train_digit_model(relation_name: str, epochs: int = 50):
    """
    Train a digit-wise model for a specific relation.

    Args:
        relation_name: Name of the recurrence relation
        epochs: Number of training epochs
    """
    print(f"\n{'='*70}")
    print(f"Training Digit-Wise Model: {relation_name}")
    print(f"{'='*70}")

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_dir = Path(f'experiments_digit/{relation_name}')
    model_dir = exp_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading training data...")
    with open(exp_dir / 'data' / 'train_examples.json', 'r') as f:
        train_data = json.load(f)
    train_examples = [tuple(ex) for ex in train_data['examples']]

    with open(exp_dir / 'data' / 'train_metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"Loaded {len(train_examples)} training examples")
    print(f"Vocabulary size: {metadata['vocab_size']}")

    # Create tokenizer
    tokenizer = DigitTokenizer()

    # Create dataset
    train_dataset = DigitFibonacciDataset(train_examples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_digit_fn
    )

    # Create model (SAME architecture as original!)
    print("\nInitializing model...")
    model = FibonacciTransformer(
        vocab_size=tokenizer.vocab_size,  # Only 12 tokens!
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=100  # Longer sequences due to digit encoding
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    train_losses = []
    train_accs = []
    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        train_loss, train_acc = train_epoch_digit(
            model, train_loader, optimizer, criterion, device, tokenizer
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
                'vocab_size': tokenizer.vocab_size,
                'tokenization': 'digit-wise'
            }
            torch.save(checkpoint, model_dir / 'best_model.pt')
            print(f"✓ Saved best model (loss: {best_loss:.4f})")

    # Save final model
    checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_losses[-1],
        'accuracy': train_accs[-1],
        'vocab_size': tokenizer.vocab_size,
        'tokenization': 'digit-wise'
    }
    torch.save(checkpoint, model_dir / 'final_model.pt')

    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accs,
        'relation': relation_name,
        'epochs': epochs,
        'best_loss': best_loss,
        'tokenization': 'digit-wise'
    }
    with open(exp_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n=== Training Complete ===")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final accuracy: {train_accs[-1]:.2f}%")

    return history


def main():
    """Train digit-wise models for all relations."""
    relations = ['fibonacci']  # Start with just Fibonacci for testing

    for relation_name in relations:
        train_digit_model(relation_name, epochs=100)


if __name__ == "__main__":
    main()
