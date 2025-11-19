"""
Run experiments on all recurrence relations and compare results.
"""
import subprocess
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")


def run_all_experiments():
    """Run experiments for all relations."""
    relations = ['fibonacci', 'linear', 'tribonacci', 'geometric', 'fibonacci_plus_constant']

    all_results = []

    for relation in relations:
        print(f"\n{'='*70}")
        print(f"RUNNING: {relation}")
        print(f"{'='*70}\n")

        # Run experiment
        result = subprocess.run(
            ['python', 'run_relation_experiment.py', relation],
            capture_output=False
        )

        if result.returncode != 0:
            print(f"Warning: {relation} experiment failed")
            continue

        # Load results
        results_file = Path(f'experiments/{relation}/results.json')
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                all_results.append(data)

    return all_results


def create_comparison_plots(results):
    """Create comparative visualizations."""

    # Create dataframe
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    x = range(len(df))
    width = 0.35

    ax1.bar([i - width/2 for i in x], df['train_accuracy'], width,
            label='Training Seeds', color='#2ecc71', alpha=0.8)
    ax1.bar([i + width/2 for i in x], df['test_accuracy'], width,
            label='Novel Seeds', color='#e74c3c', alpha=0.8)

    ax1.set_xlabel('Recurrence Relation', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy: Training vs Novel Seeds', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([r.split(':')[0] for r in df['relation_name']], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Degradation
    ax2 = axes[0, 1]
    colors = ['#e74c3c' if d > 20 else '#f39c12' if d > 10 else '#2ecc71'
              for d in df['degradation']]
    ax2.barh(range(len(df)), df['degradation'], color=colors, alpha=0.8)
    ax2.set_yticks(range(len(df)))
    ax2.set_yticklabels([r.split(':')[0] for r in df['relation_name']])
    ax2.set_xlabel('Accuracy Degradation (pp)', fontsize=12)
    ax2.set_title('Performance Degradation on Novel Seeds', fontsize=14, fontweight='bold')
    ax2.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. MAE comparison
    ax3 = axes[1, 0]
    x = range(len(df))
    ax3.bar([i - width/2 for i in x], df['train_mae'], width,
            label='Training Seeds', color='#2ecc71', alpha=0.8)
    ax3.bar([i + width/2 for i in x], df['test_mae'], width,
            label='Novel Seeds', color='#e74c3c', alpha=0.8)

    ax3.set_xlabel('Recurrence Relation', fontsize=12)
    ax3.set_ylabel('Mean Absolute Error', fontsize=12)
    ax3.set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([r.split(':')[0] for r in df['relation_name']], rotation=45, ha='right')
    ax3.legend()
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    table_data = []
    table_data.append(['Relation', 'Train Acc', 'Test Acc', 'Degrade'])

    for _, row in df.iterrows():
        relation_short = row['relation_name'].split(':')[0][:15]
        table_data.append([
            relation_short,
            f"{row['train_accuracy']:.1f}%",
            f"{row['test_accuracy']:.1f}%",
            f"{row['degradation']:.1f} pp"
        ])

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.tight_layout()
    plt.savefig('experiments/comparison_all_relations.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to experiments/comparison_all_relations.png")

    return df


def print_summary_table(results):
    """Print a formatted summary table."""
    print("\n" + "="*90)
    print("COMPARATIVE RESULTS: ALL RECURRENCE RELATIONS")
    print("="*90)
    print(f"{'Relation':<30} {'Train Acc':<12} {'Test Acc':<12} {'Degradation':<15} {'Test MAE':<15}")
    print("-"*90)

    for r in results:
        relation_name = r['relation_name'].split(':')[0]
        print(f"{relation_name:<30} {r['train_accuracy']:>10.2f}% {r['test_accuracy']:>10.2f}% "
              f"{r['degradation']:>12.2f} pp {r['test_mae']:>14.2f}")

    print("="*90)

    # Analysis
    avg_degradation = sum(r['degradation'] for r in results) / len(results)
    all_failed = all(r['test_accuracy'] < 30 for r in results)

    print(f"\nAverage Degradation: {avg_degradation:.2f} percentage points")

    if all_failed:
        print("\n*** ALL RELATIONS SHOW SYSTEMATIC FAILURE ON NOVEL SEEDS ***")
        print("This confirms the hypothesis applies broadly, not just to Fibonacci!")
    else:
        print("\nSome relations show better generalization than others.")
        best = max(results, key=lambda x: x['test_accuracy'])
        worst = min(results, key=lambda x: x['test_accuracy'])
        print(f"Best generalization: {best['relation_name'].split(':')[0]} ({best['test_accuracy']:.1f}%)")
        print(f"Worst generalization: {worst['relation_name'].split(':')[0]} ({worst['test_accuracy']:.1f}%)")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║        COMPARATIVE RECURRENCE RELATION LEARNING EXPERIMENT       ║
    ║                                                                  ║
    ║  Testing distributional learning limits across multiple          ║
    ║  recurrence relations.                                          ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Create experiments directory
    Path('experiments').mkdir(exist_ok=True)

    # Run all experiments
    print("Running experiments for all recurrence relations...")
    print("This may take several minutes...\n")

    results = run_all_experiments()

    if not results:
        print("No results to compare!")
        return

    # Save all results
    with open('experiments/all_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Create comparison visualizations
    df = create_comparison_plots(results)

    # Print summary
    print_summary_table(results)

    print("\n" + "="*90)
    print("EXPERIMENTS COMPLETE!")
    print("="*90)
    print("Results saved to experiments/all_results.json")
    print("Visualization saved to experiments/comparison_all_relations.png")


if __name__ == "__main__":
    main()
