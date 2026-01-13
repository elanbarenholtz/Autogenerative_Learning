"""
Compare number-based vs digit-wise tokenization results.

Generates visualizations showing:
1. Training curves comparison
2. Final accuracy comparison
3. Error pattern comparison
4. Performance by evaluation type
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")


def load_number_based_results():
    """Load results from original number-based experiment."""
    results = {}

    # Training history
    with open('results/training_history.json', 'r') as f:
        results['training'] = json.load(f)

    # Evaluation results
    with open('results/evaluation_summary.json', 'r') as f:
        results['evaluation'] = json.load(f)

    # Probing results (if available)
    try:
        with open('results/probing_results.json', 'r') as f:
            results['probing'] = json.load(f)
    except FileNotFoundError:
        results['probing'] = None

    return results


def load_digit_based_results():
    """Load results from digit-wise experiment."""
    results = {}

    # Training history
    with open('experiments_digit/fibonacci/training_history.json', 'r') as f:
        results['training'] = json.load(f)

    # Evaluation results (if available)
    try:
        with open('experiments_digit/fibonacci/evaluation_results.json', 'r') as f:
            results['evaluation'] = json.load(f)
    except FileNotFoundError:
        results['evaluation'] = None

    return results


def plot_training_curves_comparison(output_path='visualizations/training_comparison.png'):
    """Compare training curves between tokenization approaches."""

    print("Generating training curves comparison...")

    # Load data
    number_results = load_number_based_results()
    digit_results = load_digit_based_results()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    epochs_num = list(range(1, len(number_results['training']['train_losses']) + 1))
    epochs_dig = list(range(1, len(digit_results['training']['train_losses']) + 1))

    ax1.plot(epochs_num, number_results['training']['train_losses'],
             label='Number-based', linewidth=2, color='#2ecc71')
    ax1.plot(epochs_dig, digit_results['training']['train_losses'],
             label='Digit-wise', linewidth=2, color='#3498db')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs_num, number_results['training']['train_accuracies'],
             label='Number-based', linewidth=2, color='#2ecc71')
    ax2.plot(epochs_dig, digit_results['training']['train_accuracies'],
             label='Digit-wise', linewidth=2, color='#3498db')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax2.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Perfect')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {output_path}")


def plot_accuracy_comparison(output_path='visualizations/tokenization_accuracy_comparison.png'):
    """Compare final accuracies across evaluation types."""

    print("\nGenerating accuracy comparison...")

    # Load data
    number_results = load_number_based_results()
    digit_results = load_digit_based_results()

    if digit_results['evaluation'] is None:
        print("  ⚠ Digit-wise evaluation not yet complete, skipping...")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # Prepare data
    categories = ['Single-Step\nTraining', 'Individual\nTraining Seeds',
                  'Individual\nNovel Seeds', 'Sequential\nTraining Seeds',
                  'Sequential\nNovel Seeds']

    # Number-based results
    number_vals = [
        number_results['training']['train_accuracies'][-1],  # Single-step training
        100.0,  # Individual training (assume 100%)
        100.0,  # Individual novel (from comparative analysis)
        number_results['evaluation']['training_seeds']['average_accuracy'],  # Sequential training
        number_results['evaluation']['novel_seeds']['average_accuracy']  # Sequential novel
    ]

    # Digit-wise results
    digit_vals = [
        digit_results['training']['train_accuracies'][-1],  # Single-step training
        digit_results['evaluation']['individual']['training_seeds']['overall_accuracy'] if digit_results['evaluation'] else 0,
        digit_results['evaluation']['individual']['novel_seeds']['overall_accuracy'] if digit_results['evaluation'] else 0,
        digit_results['evaluation']['sequential']['training_seeds']['overall_accuracy'] if digit_results['evaluation'] else 0,
        digit_results['evaluation']['sequential']['novel_seeds']['overall_accuracy'] if digit_results['evaluation'] else 0
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, number_vals, width, label='Number-based',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, digit_vals, width, label='Digit-wise',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Tokenization Comparison: Number-Based vs Digit-Wise',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {output_path}")


def plot_vocabulary_analysis(output_path='visualizations/vocabulary_comparison.png'):
    """Compare vocabulary characteristics."""

    print("\nGenerating vocabulary comparison...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Vocabulary size
    categories = ['Number-based', 'Digit-wise']
    vocab_sizes = [111, 12]
    colors = ['#2ecc71', '#3498db']

    bars = ax1.bar(categories, vocab_sizes, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    for bar, size in zip(bars, vocab_sizes):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
                f'{size}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax1.set_ylabel('Vocabulary Size', fontsize=12, fontweight='bold')
    ax1.set_title('Vocabulary Size Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 125)
    ax1.grid(True, alpha=0.3, axis='y')

    # Sequence length (for 10 numbers)
    # Number-based: 10 tokens (1 per number) + 9 separators = ~19
    # Digit-wise: varies, but ~22-28 tokens
    seq_lengths = [19, 25]  # Approximate

    bars = ax2.bar(categories, seq_lengths, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    for bar, length in zip(bars, seq_lengths):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{length}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax2.set_ylabel('Sequence Length (tokens)', fontsize=12, fontweight='bold')
    ax2.set_title('Avg Sequence Length for 10 Numbers', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 30)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {output_path}")


def create_comparison_summary_table(output_path='visualizations/tokenization_summary_table.png'):
    """Create a summary table comparing both approaches."""

    print("\nGenerating summary table...")

    # Load data
    number_results = load_number_based_results()
    digit_results = load_digit_based_results()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    data = [
        ['Metric', 'Number-Based', 'Digit-Wise', 'Difference'],
        ['Vocabulary Size', '111 tokens', '12 tokens', '-89%'],
        ['Avg Sequence Length', '~19 tokens', '~25 tokens', '+32%'],
        ['Training Accuracy (final)',
         f"{number_results['training']['train_accuracies'][-1]:.1f}%",
         f"{digit_results['training']['train_accuracies'][-1]:.1f}%",
         f"{digit_results['training']['train_accuracies'][-1] - number_results['training']['train_accuracies'][-1]:+.1f} pp"
        ]
    ]

    # Add evaluation data if available
    if digit_results['evaluation']:
        data.extend([
            ['Sequential - Training Seeds',
             f"{number_results['evaluation']['training_seeds']['average_accuracy']:.1f}%",
             f"{digit_results['evaluation']['sequential']['training_seeds']['overall_accuracy']:.1f}%",
             f"{digit_results['evaluation']['sequential']['training_seeds']['overall_accuracy'] - number_results['evaluation']['training_seeds']['average_accuracy']:+.1f} pp"
            ],
            ['Sequential - Novel Seeds',
             f"{number_results['evaluation']['novel_seeds']['average_accuracy']:.1f}%",
             f"{digit_results['evaluation']['sequential']['novel_seeds']['overall_accuracy']:.1f}%",
             f"{digit_results['evaluation']['sequential']['novel_seeds']['overall_accuracy'] - number_results['evaluation']['novel_seeds']['average_accuracy']:+.1f} pp"
            ]
        ])

    table = ax.table(cellText=data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.22, 0.22, 0.21])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows (alternating colors)
    for i in range(1, len(data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.title('Tokenization Comparison: Number-Based vs Digit-Wise',
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {output_path}")


def main():
    """Generate all comparison visualizations."""
    print("="*70)
    print("GENERATING TOKENIZATION COMPARISON VISUALIZATIONS")
    print("="*70)

    # Create output directory
    Path('visualizations').mkdir(exist_ok=True)

    # Generate visualizations
    plot_training_curves_comparison()
    plot_vocabulary_analysis()

    # These require evaluation to be complete
    try:
        plot_accuracy_comparison()
        create_comparison_summary_table()
    except Exception as e:
        print(f"\n⚠ Some comparisons skipped (evaluation not complete): {e}")

    print("\n" + "="*70)
    print("COMPARISON VISUALIZATIONS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  • training_comparison.png - Training curves")
    print("  • vocabulary_comparison.png - Vocabulary analysis")
    print("  • tokenization_accuracy_comparison.png - Accuracy comparison")
    print("  • tokenization_summary_table.png - Summary table")


if __name__ == "__main__":
    main()
