"""
Master script to run the complete Fibonacci experiment pipeline.
"""
import subprocess
import sys
import os
from pathlib import Path
import time


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*70)
    print(f"{description}")
    print("="*70)

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error: {e}")
        return False


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║           FIBONACCI RECURRENCE LEARNING EXPERIMENT               ║
    ║                                                                  ║
    ║  Testing whether transformers can learn F(n) = F(n-1) + F(n-2)  ║
    ║  from distributional patterns alone.                            ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Check if we're in the right directory
    if not Path('data_generation.py').exists():
        print("Error: Please run this script from the fibonacci-experiment directory")
        sys.exit(1)

    # Create necessary directories
    for dir_name in ['data', 'models', 'results', 'visualizations']:
        Path(dir_name).mkdir(exist_ok=True)

    overall_start = time.time()

    # Step 1: Generate data
    if not run_command(
        f"{sys.executable} data_generation.py",
        "STEP 1: Generating Fibonacci sequences and training data"
    ):
        print("\nExperiment failed at data generation step")
        sys.exit(1)

    # Step 2: Train model
    if not run_command(
        f"{sys.executable} train.py",
        "STEP 2: Training transformer model"
    ):
        print("\nExperiment failed at training step")
        sys.exit(1)

    # Step 3: Evaluate model
    if not run_command(
        f"{sys.executable} evaluate.py",
        "STEP 3: Evaluating on novel seeds and extended sequences"
    ):
        print("\nExperiment failed at evaluation step")
        sys.exit(1)

    # Step 4: Generate visualizations
    if not run_command(
        f"{sys.executable} visualize.py",
        "STEP 4: Generating visualizations"
    ):
        print("\nExperiment failed at visualization step")
        sys.exit(1)

    overall_time = time.time() - overall_start

    # Print final summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nTotal time: {overall_time/60:.1f} minutes")
    print("\nResults available in:")
    print("  • results/evaluation_summary.json - Detailed metrics")
    print("  • visualizations/*.png - All plots")
    print("  • models/best_model.pt - Trained model")
    print("\nKey files to review:")
    print("  • visualizations/accuracy_comparison.png")
    print("  • visualizations/prediction_examples.png")
    print("  • visualizations/summary_table.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
