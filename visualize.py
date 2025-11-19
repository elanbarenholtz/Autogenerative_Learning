"""
Visualization of experimental results.
"""
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_accuracy_comparison(results_summary, output_path='results/accuracy_comparison.png'):
    """Compare accuracy between training and novel seeds."""

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Training Seeds\n(Seen)', 'Novel Seeds\n(Unseen)']
    accuracies = [
        results_summary['training_seeds']['average_accuracy'],
        results_summary['novel_seeds']['average_accuracy']
    ]

    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Exact Match Accuracy (%)', fontsize=12)
    ax.set_title('Model Performance: Training vs Novel Seeds', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random baseline')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Accuracy comparison saved to {output_path}")


def plot_error_progression(detailed_results, output_path='results/error_progression.png'):
    """Plot how errors evolve over sequence positions."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Training seeds
    for result in detailed_results['training_seeds'][:3]:  # Plot first 3
        seed = result['seed']
        errors = result['errors']
        positions = list(range(len(errors)))
        ax1.plot(positions, errors, marker='o', alpha=0.7, label=f'Seed {seed}')

    ax1.set_xlabel('Position in Sequence', fontsize=11)
    ax1.set_ylabel('Absolute Error', fontsize=11)
    ax1.set_title('Error Progression: Training Seeds', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Novel seeds
    for result in detailed_results['novel_seeds'][:3]:  # Plot first 3
        seed = result['seed']
        errors = result['errors']
        positions = list(range(len(errors)))
        ax2.plot(positions, errors, marker='o', alpha=0.7, label=f'Seed {seed}')

    ax2.set_xlabel('Position in Sequence', fontsize=11)
    ax2.set_ylabel('Absolute Error', fontsize=11)
    ax2.set_title('Error Progression: Novel Seeds', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error progression saved to {output_path}")


def plot_prediction_examples(detailed_results, output_path='results/prediction_examples.png'):
    """Show specific prediction examples."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Plot 2 training seeds and 2 novel seeds
    seeds_to_plot = [
        ('Training', detailed_results['training_seeds'][0]),
        ('Training', detailed_results['training_seeds'][1]),
        ('Novel', detailed_results['novel_seeds'][0]),
        ('Novel', detailed_results['novel_seeds'][1])
    ]

    for idx, (seed_type, result) in enumerate(seeds_to_plot):
        ax = axes[idx]

        seed = result['seed']
        predictions = result['predictions'][:15]  # First 15 predictions
        targets = result['targets'][:15]
        positions = list(range(len(predictions)))

        ax.plot(positions, targets, marker='o', label='True', linewidth=2, markersize=8)
        ax.plot(positions, predictions, marker='x', label='Predicted', linewidth=2, markersize=8)

        ax.set_xlabel('Position', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'{seed_type} Seed {seed} - Accuracy: {result["exact_match_accuracy"]:.1f}%',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prediction examples saved to {output_path}")


def plot_mae_distribution(detailed_results, output_path='results/mae_distribution.png'):
    """Plot distribution of mean absolute errors."""

    fig, ax = plt.subplots(figsize=(10, 6))

    training_maes = [r['mean_absolute_error'] for r in detailed_results['training_seeds']]
    novel_maes = [r['mean_absolute_error'] for r in detailed_results['novel_seeds']]

    positions = [1, 2]
    box_data = [training_maes, novel_maes]

    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True)

    # Color the boxes
    colors = ['#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(['Training Seeds', 'Novel Seeds'])
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"MAE distribution saved to {output_path}")


def plot_individual_seed_performance(results_summary, output_path='results/individual_seeds.png'):
    """Plot accuracy for each individual seed."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Training seeds
    training_data = results_summary['training_seeds']['individual_results']
    training_seeds = [str(r['seed']) for r in training_data]
    training_accs = [r['exact_match_accuracy'] for r in training_data]

    ax1.barh(training_seeds, training_accs, color='#2ecc71', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Exact Match Accuracy (%)', fontsize=11)
    ax1.set_title('Training Seeds Performance', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.grid(True, alpha=0.3, axis='x')

    # Novel seeds
    novel_data = results_summary['novel_seeds']['individual_results']
    novel_seeds = [str(r['seed']) for r in novel_data]
    novel_accs = [r['exact_match_accuracy'] for r in novel_data]

    ax2.barh(novel_seeds, novel_accs, color='#e74c3c', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Exact Match Accuracy (%)', fontsize=11)
    ax2.set_title('Novel Seeds Performance', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Individual seed performance saved to {output_path}")


def create_summary_table(results_summary, output_path='results/summary_table.png'):
    """Create a summary table as an image."""

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    data = [
        ['Metric', 'Training Seeds', 'Novel Seeds', 'Degradation'],
        [
            'Exact Match Accuracy (%)',
            f"{results_summary['training_seeds']['average_accuracy']:.2f}",
            f"{results_summary['novel_seeds']['average_accuracy']:.2f}",
            f"{results_summary['training_seeds']['average_accuracy'] - results_summary['novel_seeds']['average_accuracy']:.2f}"
        ],
        [
            'Mean Absolute Error',
            f"{results_summary['training_seeds']['average_mae']:.2f}",
            f"{results_summary['novel_seeds']['average_mae']:.2f}",
            f"{results_summary['novel_seeds']['average_mae'] - results_summary['training_seeds']['average_mae']:.2f}"
        ],
        [
            'Number of Seeds',
            f"{len(results_summary['training_seeds']['individual_results'])}",
            f"{len(results_summary['novel_seeds']['individual_results'])}",
            '-'
        ]
    ]

    table = ax.table(cellText=data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.2, 0.2, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, 4):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.title('Experiment Results Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary table saved to {output_path}")


def main():
    print("=== Generating Visualizations ===\n")

    # Load results
    print("Loading evaluation results...")
    with open('results/evaluation_summary.json', 'r') as f:
        results_summary = json.load(f)

    detailed_results = torch.load('results/detailed_results.pt', weights_only=False)

    # Create output directory
    Path('visualizations').mkdir(exist_ok=True)

    # Generate all plots
    print("\nGenerating plots...")

    plot_accuracy_comparison(results_summary, 'visualizations/accuracy_comparison.png')
    plot_error_progression(detailed_results, 'visualizations/error_progression.png')
    plot_prediction_examples(detailed_results, 'visualizations/prediction_examples.png')
    plot_mae_distribution(detailed_results, 'visualizations/mae_distribution.png')
    plot_individual_seed_performance(results_summary, 'visualizations/individual_seeds.png')
    create_summary_table(results_summary, 'visualizations/summary_table.png')

    print("\n=== All Visualizations Complete ===")
    print("Visualizations saved to visualizations/ directory")


if __name__ == "__main__":
    main()
