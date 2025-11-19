# Files to Share with AI for Paper Writing

## Essential Files (Share These)

### 1. Visualizations (7 figures)
- `visualizations/accuracy_comparison.png` - Main result: 44% → 18%
- `visualizations/error_progression.png` - Error compounding over time
- `visualizations/prediction_examples.png` - Visual examples of predictions
- `visualizations/mae_distribution.png` - Error distribution comparison
- `visualizations/individual_seeds.png` - Per-seed breakdown
- `visualizations/summary_table.png` - Clean summary table
- `visualizations/probing_results.png` - Mechanistic interpretability

### 2. Documentation
- `FIGURES_GUIDE.md` - **START HERE** - Explains each figure and how they relate
- `COMPLETE_EXPERIMENTAL_REPORT.md` - Full technical analysis
- `README.md` - Project overview and quick start

### 3. Raw Data
- `results/evaluation_summary.json` - Numerical results for all evaluations
- `results/probing_results.json` - Mechanistic probing data
- `results/training_history.json` - Shows 100% training accuracy

## Quick Summary for AI

### Experiment Design
- **Goal**: Test if transformers can learn Fibonacci recurrence relation F(n) = F(n-1) + F(n-2) from next-token prediction alone
- **Hypothesis**: They cannot, because Fibonacci is "not autogenerative" (rule exists independently of distribution)
- **Architecture**: Small transformer (630K params, 3 layers, trained from scratch)
- **Training Data**: 20 diverse Fibonacci seeds, 138 training examples

### Key Results

#### 1. Training (Single-Step Prediction)
- **100% accuracy** from epoch 5 onwards
- Model can perfectly predict next token given correct context

#### 2. Sequential Evaluation (Main Finding)
- **Training seeds**: 44% exact match accuracy (positions 10-30)
- **Novel seeds**: 18% exact match accuracy (positions 10-30)
- **26 percentage point degradation**
- Errors compound exponentially

#### 3. Mechanistic Probing (Why It Fails)
- F(n-1) probe: 62.5% accuracy
- F(n-2) probe: 58.3% accuracy
- **Conclusion**: Model learned neither addition rule nor ratio approximation
- Weak, mixed representations explain poor performance

### Critical Insight
**Perfect memorization ≠ Rule learning**

The model can memorize local patterns (100% single-step) but cannot generate coherent sequences (44% → 18% sequential) because it never learned the underlying F(n) = F(n-1) + F(n-2) computation.

### Paper Narrative Arc

1. **Motivation**: Can LLMs learn recurrence relations from distributional patterns?
2. **Experiment**: Train transformer on Fibonacci sequences
3. **Result 1**: Perfect single-step accuracy (100%) - model appears to learn
4. **Result 2**: Poor sequential accuracy (44% → 18%) - but fails to generalize
5. **Result 3**: Weak internal representations (58%) - didn't learn the rule
6. **Conclusion**: Supports "non-autogenerative" hypothesis - some patterns require rule-based computation, not just distributional learning

### Figures Tell the Story

- **Figure 1** (accuracy_comparison.png): THE KEY FIGURE - Shows complete story: 100% → 44% → 18%
  - Perfect single-step (memorization works)
  - 56pp drop to sequential (no rule learning)
  - 26pp drop on novel seeds (no generalization)
- **Figures 2-6**: Different perspectives on sequential evaluation failure
- **Figure 7** (probing_results.png): The mechanistic explanation for WHY

### Additional Context Available

The complete codebase is at: https://github.com/elanbarenholtz/Autogenerative_Learning

All code, data generation, training, and evaluation scripts are included for reproducibility.
