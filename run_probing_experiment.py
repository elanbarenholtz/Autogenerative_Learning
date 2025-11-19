"""
Mechanistic Probing Experiment for Fibonacci Model

Tests whether the model learns:
- Addition Rule: F(n) = F(n-1) + F(n-2) [needs both terms]
- Ratio Approximation: F(n) ≈ φ * F(n-1) [only needs F(n-1)]

Method: Train linear probes to decode F(n-1) and F(n-2) from hidden states.
- High F(n-2) accuracy → Model uses addition
- Low F(n-2) accuracy → Model uses ratio approximation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import existing modules (adapted to current codebase)
from model import FibonacciTransformer, NumberTokenizer
from data_generation import generate_fibonacci_sequence

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
PROBE_LR = 0.001
PROBE_EPOCHS = 50
LAYERS_TO_PROBE = [0, 1, 2]  # Probe all 3 layers

class LinearProbe(nn.Module):
    """A simple linear classifier to decode tokens from hidden states."""
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        return self.linear(x)

def get_activations(model, inputs, layer_indices):
    """
    Runs the model and captures activations from specified layers.
    Returns a dictionary: {layer_idx: [batch_size, seq_len, hidden_dim]}

    Adapted for FibonacciTransformer architecture.
    """
    activations = defaultdict(list)
    handles = []

    def get_hook(layer_idx):
        def hook(module, input, output):
            # output is (batch, seq, hidden)
            activations[layer_idx] = output.detach()
        return hook

    # Access transformer encoder layers
    # In FibonacciTransformer, structure is: model.transformer.layers
    transformer_layers = model.transformer.layers

    for i in layer_indices:
        if i < len(transformer_layers):
            handles.append(transformer_layers[i].register_forward_hook(get_hook(i)))

    # Forward pass
    with torch.no_grad():
        model(inputs)

    # Remove hooks
    for h in handles:
        h.remove()

    return activations

def train_probes(model, train_loader, test_loader, vocab_size, hidden_dim):
    """
    Trains two probes per layer:
    1. Target = F(n-1) (The term needed for Ratio approximation)
    2. Target = F(n-2) (The term ONLY needed for the Addition Rule)
    """

    # Store results: {layer: {'n-1': acc, 'n-2': acc}}
    results = defaultdict(dict)

    print(f"\n--- Training Probes on {DEVICE} ---")

    for layer in LAYERS_TO_PROBE:
        print(f"\nAnalyzing Layer {layer}...")

        # Initialize probes for this layer
        probe_n1 = LinearProbe(hidden_dim, vocab_size).to(DEVICE)  # Probes F(n-1)
        probe_n2 = LinearProbe(hidden_dim, vocab_size).to(DEVICE)  # Probes F(n-2)

        optimizer = optim.Adam(
            list(probe_n1.parameters()) + list(probe_n2.parameters()),
            lr=PROBE_LR
        )
        criterion = nn.CrossEntropyLoss()

        # Training Loop
        for epoch in range(PROBE_EPOCHS):
            probe_n1.train()
            probe_n2.train()

            epoch_loss = 0
            for inputs, targets_n1, targets_n2 in train_loader:
                # Get model activations for this batch
                acts_dict = get_activations(model, inputs, [layer])

                # We probe the LAST token's activation
                # acts shape: [batch, seq_len, hidden] -> [batch, hidden]
                layer_acts = acts_dict[layer][:, -1, :]

                # Forward pass probes
                pred_n1 = probe_n1(layer_acts)
                pred_n2 = probe_n2(layer_acts)

                # Loss
                loss = criterion(pred_n1, targets_n1) + criterion(pred_n2, targets_n2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{PROBE_EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

        # Evaluation Loop
        probe_n1.eval()
        probe_n2.eval()
        correct_n1 = 0
        correct_n2 = 0
        total = 0

        with torch.no_grad():
            for inputs, targets_n1, targets_n2 in test_loader:
                acts_dict = get_activations(model, inputs, [layer])
                layer_acts = acts_dict[layer][:, -1, :]

                # Check F(n-1) Accuracy
                pred_n1 = probe_n1(layer_acts).argmax(dim=1)
                correct_n1 += (pred_n1 == targets_n1).sum().item()

                # Check F(n-2) Accuracy
                pred_n2 = probe_n2(layer_acts).argmax(dim=1)
                correct_n2 += (pred_n2 == targets_n2).sum().item()

                total += inputs.size(0)

        acc_n1 = correct_n1 / total
        acc_n2 = correct_n2 / total

        results[layer]['n-1'] = acc_n1
        results[layer]['n-2'] = acc_n2

        print(f"  Layer {layer} Results:")
        print(f"    Probe F(n-1) [Ratio Term]: {acc_n1*100:.2f}%")
        print(f"    Probe F(n-2) [Rule Term]:  {acc_n2*100:.2f}%")

    return results

def prepare_probing_data(sequences, tokenizer, context_len=10):
    """
    Converts sequences into (Input, Target_N-1, Target_N-2) tuples.

    For each position t in sequence:
    - Input: tokens up to position t
    - Target N-1: token at position t (F(n-1) for predicting next)
    - Target N-2: token at position t-1 (F(n-2) for predicting next)
    """
    inputs = []
    targets_n1 = []  # F(n-1)
    targets_n2 = []  # F(n-2)

    for seq in sequences:
        # Create sliding windows
        if len(seq) < context_len + 2:
            continue

        for i in range(2, len(seq) - 1):
            # Input window ending at index i
            start_idx = max(0, i + 1 - context_len)
            curr_input = seq[start_idx : i + 1]

            # Pad if short
            if len(curr_input) < context_len:
                curr_input = [0] * (context_len - len(curr_input)) + curr_input

            # Encode to token IDs
            encoded_input = tokenizer.encode(curr_input)
            inputs.append(encoded_input)

            # Targets (already as token values, need to encode)
            targets_n1.append(tokenizer.encode([seq[i]])[0])      # F(n-1)
            targets_n2.append(tokenizer.encode([seq[i-1]])[0])    # F(n-2)

    return (
        torch.tensor(inputs, dtype=torch.long).to(DEVICE),
        torch.tensor(targets_n1, dtype=torch.long).to(DEVICE),
        torch.tensor(targets_n2, dtype=torch.long).to(DEVICE)
    )

def main():
    print("="*70)
    print("MECHANISTIC PROBING EXPERIMENT")
    print("="*70)

    print("\n1. Loading Trained Model...")
    checkpoint_path = 'models/best_model.pt'

    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please run 'python train.py' first to train the model.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Reconstruct tokenizer from saved vocabulary
    vocabulary = checkpoint['vocabulary']
    tokenizer = NumberTokenizer(vocabulary)

    # Reconstruct model
    model = FibonacciTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.0  # No dropout for evaluation
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    print("\n2. Generating Probing Data...")
    # Use subset of TRAINING seeds that are within vocabulary bounds
    # This ensures we have enough data and tests what model actually learned
    test_seeds = [(0, 1), (1, 1), (2, 3), (7, 11), (2, 5), (4, 7), (6, 10), (1, 3)]

    sequences = []
    for seed in test_seeds:
        seq = generate_fibonacci_sequence(seed, length=20)
        # Filter to vocabulary
        valid_seq = [x for x in seq if x in vocabulary]
        if len(valid_seq) >= 15:  # Only use if we have enough terms
            sequences.append(valid_seq)

    print(f"✓ Generated {len(sequences)} sequences for probing")

    inputs, t_n1, t_n2 = prepare_probing_data(sequences, tokenizer, context_len=10)

    print(f"✓ Created {len(inputs)} probe examples")

    # Split Train/Test for the PROBES (not the model)
    split_idx = int(len(inputs) * 0.8)
    train_dataset = TensorDataset(inputs[:split_idx], t_n1[:split_idx], t_n2[:split_idx])
    test_dataset = TensorDataset(inputs[split_idx:], t_n1[split_idx:], t_n2[split_idx:])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print("\n3. Running Probe Training...")
    print("This tests what the model has learned internally:")
    print("  - High F(n-2) accuracy → Model uses ADDITION rule")
    print("  - Low F(n-2) accuracy → Model uses RATIO approximation")

    results = train_probes(
        model, train_loader, test_loader,
        tokenizer.vocab_size,
        model.d_model
    )

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    layers = sorted(results.keys())
    n1_accs = [results[l]['n-1'] for l in layers]
    n2_accs = [results[l]['n-2'] for l in layers]

    print("\nLayer-by-layer Analysis:")
    for layer in layers:
        print(f"  Layer {layer}:")
        print(f"    F(n-1) accuracy: {results[layer]['n-1']*100:.2f}%")
        print(f"    F(n-2) accuracy: {results[layer]['n-2']*100:.2f}%")

        # Interpretation
        if results[layer]['n-2'] > 0.7:
            print(f"    → Strong evidence for ADDITION rule")
        elif results[layer]['n-2'] < 0.3:
            print(f"    → Strong evidence for RATIO approximation")
        else:
            print(f"    → Mixed/uncertain representation")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(layers, n1_accs, marker='o', label='F(n-1) [Ratio Term]',
             linewidth=3, markersize=10)
    plt.plot(layers, n2_accs, marker='x', label='F(n-2) [Addition Term]',
             linewidth=3, markersize=10, linestyle='--')
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random baseline')
    plt.xlabel('Transformer Layer', fontsize=12)
    plt.ylabel('Probe Accuracy', fontsize=12)
    plt.title('Mechanistic Probing: What Does the Model Represent?',
              fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(layers)

    os.makedirs('visualizations', exist_ok=True)
    save_path = 'visualizations/probing_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {save_path}")

    # Save results
    import json
    results_dict = {
        'layers': layers,
        'f_n1_accuracy': n1_accs,
        'f_n2_accuracy': n2_accs,
        'interpretation': {
            layer: {
                'uses_addition': results[layer]['n-2'] > 0.7,
                'uses_ratio': results[layer]['n-2'] < 0.3,
                'uncertain': 0.3 <= results[layer]['n-2'] <= 0.7
            }
            for layer in layers
        }
    }

    results_path = 'results/probing_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"✓ Results saved to {results_path}")

    print("\n" + "="*70)
    print("INTERPRETATION SUMMARY")
    print("="*70)

    avg_n2_acc = np.mean(n2_accs)
    if avg_n2_acc > 0.7:
        print("✓ Model appears to use ADDITION RULE: F(n) = F(n-1) + F(n-2)")
        print("  Evidence: Strong representation of F(n-2) across layers")
    elif avg_n2_acc < 0.3:
        print("✓ Model appears to use RATIO APPROXIMATION: F(n) ≈ φ * F(n-1)")
        print("  Evidence: Weak representation of F(n-2), only F(n-1) is tracked")
    else:
        print("~ Mixed results - model may use both strategies")
        print(f"  Average F(n-2) accuracy: {avg_n2_acc*100:.1f}%")

if __name__ == "__main__":
    main()
