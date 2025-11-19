# Fibonacci Recurrence Learning Experiment

## Overview

This experiment tests whether a pure distributional learner (transformer doing next-token prediction) can learn the Fibonacci recurrence relation `F(n) = F(n-1) + F(n-2)` from examples alone.

**Hypothesis**: The model will fail to generalize to novel seeds because Fibonacci is not autogenerative - the rule exists independently of distributional patterns.

## Experimental Design

### Training Data
- **20 diverse seeds**: `[(0,1), (1,1), (2,3), (1,2), (5,8), ...]`
- **Sequence length**: 30 numbers per seed
- **Training examples**: Sliding window approach with context size of 10
- **Total examples**: ~400 training examples

### Model Architecture
- Small GPT-style decoder-only transformer
- **Layers**: 3
- **Embedding dimension**: 128
- **Attention heads**: 4
- **Feedforward dimension**: 512
- **Training objective**: Next-token prediction

### Test Conditions

1. **Novel Seeds (Primary Test)**
   - Seeds never seen during training: `[(17,29), (23,37), (13,21), (31,50), (19,31)]`
   - Same sequence length as training (30 numbers)
   - Tests whether model learned the recurrence rule vs. memorized patterns

2. **Training Seeds (Baseline)**
   - Evaluate on seeds seen during training
   - Establishes upper bound performance

3. **Extended Sequences**
   - Training seeds extended to 50 numbers
   - Tests length generalization

4. **Autoregressive Generation**
   - Model uses its own predictions as context
   - Tests robustness of learned patterns

## Installation

```bash
cd fibonacci-experiment
pip install -r requirements.txt
```

## Usage

### Run Complete Experiment

```bash
python run_experiment.py
```

This will execute all steps:
1. Generate training/test data
2. Train the model
3. Evaluate on all test conditions
4. Generate visualizations

### Run Individual Steps

```bash
# Generate data only
python data_generation.py

# Train model only
python train.py

# Evaluate trained model
python evaluate.py

# Generate visualizations
python visualize.py
```

## Project Structure

```
fibonacci-experiment/
├── data_generation.py      # Generate Fibonacci sequences
├── model.py                # Transformer architecture
├── train.py                # Training loop
├── evaluate.py             # Evaluation suite
├── visualize.py            # Result visualization
├── run_experiment.py       # Master pipeline script
├── requirements.txt        # Dependencies
├── data/                   # Generated datasets
├── models/                 # Trained model checkpoints
├── results/                # Evaluation metrics (JSON)
└── visualizations/         # Generated plots
```

## Evaluation Metrics

### Primary Metrics
- **Exact Match Accuracy**: Percentage of predictions that exactly match the true Fibonacci number
- **Mean Absolute Error (MAE)**: Average deviation from correct values
- **Degradation**: Performance drop from training seeds to novel seeds

### Expected Results
- **Training Seeds**: High accuracy (~95%+)
- **Novel Seeds**: Systematic failures (accuracy drop)
- **Conclusion**: Model cannot generalize the recurrence rule

## Results Files

After running the experiment:

- `results/evaluation_summary.json` - Aggregate metrics
- `results/detailed_results.pt` - Full predictions and targets
- `results/training_history.json` - Training curves
- `visualizations/accuracy_comparison.png` - Main result
- `visualizations/prediction_examples.png` - Specific examples
- `visualizations/error_progression.png` - Error over sequence positions
- `visualizations/mae_distribution.png` - Error distributions
- `visualizations/summary_table.png` - Results table

## Key Visualizations

1. **Accuracy Comparison**: Bar chart comparing training vs novel seed performance
2. **Prediction Examples**: Side-by-side plots of predictions vs ground truth
3. **Error Progression**: How errors evolve across sequence positions
4. **MAE Distribution**: Box plots showing error distributions

## Interpretation

### If Hypothesis Confirmed
- Training accuracy high, novel seed accuracy low
- Model has memorized patterns, not learned the rule
- Demonstrates limits of pure distributional learning

### If Hypothesis Rejected
- Model generalizes well to novel seeds
- Suggests transformers can learn recurrence relations
- Would require deeper analysis of what was learned

## Optional Enhancements

To test generality beyond Fibonacci:

```python
# In data_generation.py, add alternative recurrence:
# F(n) = 2*F(n-1) + F(n-2)
```

## Technical Notes

- **Vocabulary Management**: Capped at 10,000 to handle large Fibonacci numbers
- **Tokenization**: Each number is treated as a discrete token
- **Context Window**: Fixed at 10 numbers (sufficient for F(n-1) + F(n-2))
- **No data augmentation**: Pure distributional test

## Citation

If you use this code or approach:

```
Fibonacci Recurrence Learning Experiment
Testing distributional learning limits on arithmetic recurrence relations
https://github.com/[your-repo]
```

## License

MIT License - feel free to use and modify for research purposes.
