# Fibonacci Experiment: Project Context

## Project Overview

**Research Question**: Can transformer models learn recurrence relations (like Fibonacci sequences) from examples? What representations do they learn internally?

**Key Finding**: Models achieve perfect (100%) single-step prediction but struggle with sequential generation (44% training seeds, 18% novel seeds). Digit-wise tokenization catastrophically fails (0% evaluation accuracy).

**Current Status**: Completed baseline experiments (Experiment 1) and digit-wise tokenization comparison (Experiment 2A). Ready for probing experiments to understand internal representations.

---

## Quick Start

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Check everything is installed
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Running Key Scripts
```bash
# Train on Fibonacci (number-based tokenization)
python train.py

# Evaluate trained model
python evaluate.py

# Run probing analysis
python probing.py

# Generate comparative visualizations
python visualize_results.py

# Compare number-based vs digit-wise
python compare_tokenizations.py
```

---

## Project Structure

```
fibonacci-experiment/
├── Core Model & Training
│   ├── model.py                    # Transformer architecture (3 layers, 4 heads, 128 dim)
│   ├── tokenizer.py                # Number tokenizer (0-10000 vocab)
│   ├── train.py                    # Training script (AdamW, 100 epochs)
│   └── evaluate.py                 # Evaluation (individual + sequential)
│
├── Data Generation
│   ├── recurrence_relations.py    # All 5 recurrence relation definitions
│   ├── data_generation.py         # Generate training/test data
│   └── data/                       # Generated sequences and examples
│
├── Experiment 2A: Digit-Wise Tokenization
│   ├── digit_tokenizer.py         # Digit-level tokenizer (12 tokens)
│   ├── digit_train.py             # Training for digit-wise encoding
│   ├── digit_evaluate.py          # Evaluation for digit-wise
│   ├── digit_data_generation.py   # Data generation for digit-wise
│   ├── compare_tokenizations.py   # Comparative analysis script
│   └── experiments_digit/         # Results for digit-wise experiments
│       ├── fibonacci/             # Fibonacci digit-wise results
│       └── linear/                # Linear digit-wise results
│
├── Analysis & Visualization
│   ├── probing.py                 # Probing classifiers for internal representations
│   ├── visualize_results.py       # Generate plots and charts
│   └── compare_all_relations.py   # Cross-relation comparisons
│
├── Results & Documentation
│   ├── results/                   # Training histories, evaluation results
│   ├── visualizations/            # Generated plots (PNG files)
│   ├── EXPERIMENT_2A_SUMMARY.md   # Digit-wise tokenization findings
│   └── PROJECT_CONTEXT.md         # This file
│
└── Environment
    ├── venv/                      # Python virtual environment
    └── requirements.txt           # Python dependencies
```

---

## Key Experiments & Results

### Experiment 1: Baseline (Number-Based Tokenization)

**Purpose**: Establish baseline performance with standard number tokenization

**Architecture**:
- 3-layer transformer decoder
- 4 attention heads
- d_model = 128, FFN = 512
- ~630K parameters
- Vocabulary: 0-10000 (one token per number)

**Training**:
- 20 training seeds (different Fibonacci starting points)
- 400 training examples (20 seeds × 20 examples each)
- Sliding window: predict next number from 10 previous
- AdamW optimizer (lr=0.0005, weight_decay=0.01)
- 100 epochs

**Results**:
```
Single-Step Prediction (given correct context):
  - Training seeds: 100.0%
  - Novel seeds:    100.0%

Sequential Generation (autoregressive):
  - Training seeds: 44.0%
  - Novel seeds:    18.0%
```

**Key Insight**: Model learns perfect next-step prediction but struggles with multi-step rollout, suggesting error accumulation or distribution shift during autoregressive generation.

**Files**:
- Training history: `results/training_history.json`
- Evaluation results: `results/evaluation_summary.json`
- Model checkpoint: `fibonacci_model.pt`

### Experiment 2A: Digit-Wise Tokenization (NEGATIVE RESULT)

**Purpose**: Test if breaking numbers into digits helps learning

**Approach**:
- Tokenize each number as separate digit tokens
- Example: 144 → [DIGIT_1][DIGIT_4][DIGIT_4][SEP]
- Vocabulary reduced from 111 → 12 tokens (89% reduction)
- Sequence length increased from ~19 → ~25 tokens

**Tested Relations**:
1. **Fibonacci** (F_{n+2} = F_{n+1} + F_n) - Medium complexity
2. **Linear** (a_{n+1} = a_n + d) - Trivial complexity

**Results**:

| Relation | Training Acc | Individual (train) | Individual (novel) | Sequential (train) | Sequential (novel) |
|----------|--------------|-------------------|-------------------|-------------------|-------------------|
| **Fibonacci** | 75.5% | 0% | 0% | 0% | 0% |
| **Linear** | **18.75%** | 0% | 0% | 0% | 0% |
| Number-based (Fib) | 100% | 100% | 100% | 44% | 18% |

**Critical Finding**: **INVERSE COMPLEXITY RELATIONSHIP**
- Model performs 4× worse on trivial Linear task (18.75%) than medium Fibonacci (75.5%)
- 0% evaluation accuracy across ALL tests for BOTH relations
- Definitive proof digit-wise tokenization fails fundamentally

**Why It Failed**:
1. Never learned that digit sequences represent numbers
2. Never learned to predict SEP token (termination)
3. Generates 10-digit repeating patterns (e.g., 1111111111, 9797979797)
4. Multi-token prediction too complex for standard transformers

**Conclusion**: **ABANDON digit-wise tokenization entirely.** Number-based is the only viable approach.

**Files**:
- Summary: `EXPERIMENT_2A_SUMMARY.md`
- Fibonacci results: `experiments_digit/fibonacci/`
- Linear results: `experiments_digit/linear/`
- Visualizations: `visualizations/tokenization_*.png`

### Probing Experiments (Completed)

**Purpose**: Understand what internal representations the model learns

**Method**: Train linear probes on hidden states to predict:
- Current number value
- Position in sequence
- Recurrence pattern

**Results**: (See `results/probing_results.json`)

**Key Findings**:
- Layer 0: Primarily encodes token identity
- Layer 1: Mixed representations (position + value)
- Layer 2: Task-relevant abstractions

---

## Recurrence Relations Implemented

All relations defined in `recurrence_relations.py`:

1. **Fibonacci**: F(n) = F(n-1) + F(n-2)
   - Seeds: 20 training, 5 novel test
   - Example: (0,1) → [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, ...]

2. **Linear**: F(n) = 2·F(n-1) + F(n-2)
   - Seeds: 20 training, 5 novel test
   - Example: (1,1) → [1, 1, 3, 7, 17, 41, 99, ...]

3. **Tribonacci**: F(n) = F(n-1) + F(n-2) + F(n-3)
   - Seeds: 20 training, 5 novel test
   - Example: (0,0,1) → [0, 0, 1, 1, 2, 4, 7, 13, 24, ...]

4. **Geometric**: F(n) = 2·F(n-1)
   - Seeds: 20 training, 5 novel test
   - Example: (1) → [1, 2, 4, 8, 16, 32, 64, ...]

5. **Fibonacci+1**: F(n) = F(n-1) + F(n-2) + 1
   - Seeds: 20 training, 5 novel test
   - Example: (0,1) → [0, 1, 2, 4, 7, 12, 20, 33, ...]

---

## Key Files Reference

### Training & Models
- `model.py:77-162` - FibonacciTransformer class definition
- `train.py:115-180` - Main training loop
- `tokenizer.py:1-93` - FibonacciTokenizer class

### Evaluation
- `evaluate.py:163-250` - Individual prediction evaluation
- `evaluate.py:253-350` - Sequential generation evaluation

### Data Generation
- `recurrence_relations.py:28-56` - Fibonacci class
- `data_generation.py:55-112` - Training data generation
- Training seeds: `recurrence_relations.py:46-52`

### Digit-Wise (Experiment 2A)
- `digit_tokenizer.py:1-177` - DigitTokenizer with 12-token vocab
- `digit_train.py:115-270` - Training with multi-token targets
- `digit_evaluate.py:163-315` - Evaluation with autoregressive number generation

### Analysis
- `probing.py:1-250` - Probing classifier experiments
- `visualize_results.py:1-300` - Plotting and visualization

---

## Training Configuration

### Number-Based (Standard)
```python
# Model
vocab_size = 111  # Covers 0-100 range in training
d_model = 128
nhead = 4
num_layers = 3
dim_feedforward = 512
dropout = 0.1
max_seq_len = 100

# Training
batch_size = 32
learning_rate = 0.0005
weight_decay = 0.01
epochs = 100
optimizer = AdamW
```

### Digit-Wise (Experiment 2A)
```python
# Model (same architecture, different tokenization)
vocab_size = 12  # [PAD], [SEP], [DIGIT_0]-[DIGIT_9]
d_model = 128
nhead = 4
num_layers = 3
dim_feedforward = 512
dropout = 0.1
max_seq_len = 200  # Longer due to digit sequences

# Training (same hyperparameters)
batch_size = 32
learning_rate = 0.0005
weight_decay = 0.01
epochs = 100
optimizer = AdamW
```

---

## Important Findings Summary

### ✅ What Works
1. **Number-based tokenization** (one token per number)
2. **Single-step prediction** (100% accuracy)
3. **Small transformer architectures** (~630K params sufficient)
4. **Limited training data** (400 examples enough for single-step)

### ❌ What Doesn't Work
1. **Digit-wise tokenization** (0% evaluation accuracy)
2. **Sequential generation** (only 18-44% accuracy)
3. **Compositional number representations** (never emerge)
4. **Smaller vocabularies** (89% reduction → catastrophic failure)

### 🔬 Key Research Questions (Open)
1. Why does sequential generation fail (44% → 18%)?
   - Error accumulation?
   - Distribution shift?
   - Planning/lookahead limitations?

2. What do models actually learn?
   - Lookup tables?
   - Algorithmic patterns?
   - Position-specific rules?

3. Can architectural modifications help?
   - Recurrent connections?
   - Memory modules?
   - Specialized number representations?

---

## Next Steps & Future Work

### Immediate Priorities
1. **Probing analysis** - Understand internal representations
2. **Error analysis** - Where/why does sequential generation fail?
3. **Ablation studies** - Which components are critical?

### Experimental Ideas
1. **Curriculum learning** - Start with shorter sequences
2. **Teacher forcing schedules** - Gradual exposure to errors
3. **Attention analysis** - Visualize what model attends to
4. **Longer training** - Does more data help sequential?
5. **Architectural variants** - Recurrence, memory, graph networks

### Analysis & Writing
1. Document error patterns in sequential generation
2. Create visualizations of model predictions vs. targets
3. Write up Experiment 2A as cautionary tale for papers
4. Compare with related work on numerical reasoning

---

## Common Commands

### Training
```bash
# Train Fibonacci (number-based)
python train.py

# Train Linear (number-based)
python train.py --relation linear

# Train Fibonacci (digit-wise) - NOT RECOMMENDED
python digit_train.py --relation fibonacci --epochs 100
```

### Evaluation
```bash
# Evaluate Fibonacci
python evaluate.py

# Evaluate Linear
python evaluate.py --relation linear

# Evaluate digit-wise
python digit_evaluate.py --relation fibonacci
```

### Analysis
```bash
# Run probing experiments
python probing.py

# Generate visualizations
python visualize_results.py

# Compare number-based vs digit-wise
python compare_tokenizations.py
```

### Data Generation
```bash
# Generate data for all relations
python data_generation.py

# Generate digit-wise data
python digit_data_generation.py
```

---

## Troubleshooting

### Model not training well?
- Check learning rate (0.0005 is good baseline)
- Verify data loading (should have 400 examples)
- Check loss curve (should drop quickly to ~0.01)

### Out of memory?
- Reduce batch size (32 → 16)
- Reduce max_seq_len (100 → 50)
- Use CPU: `device = 'cpu'`

### Digit-wise failing?
- This is expected (0% evaluation accuracy)
- Don't waste compute - use number-based instead
- See EXPERIMENT_2A_SUMMARY.md for details

### Can't find results?
- Training: `results/training_history.json`
- Evaluation: `results/evaluation_summary.json`
- Model: `fibonacci_model.pt` or `results/models/`
- Digit-wise: `experiments_digit/{relation}/`

---

## File Formats

### Training History JSON
```json
{
  "train_losses": [2.5, 1.2, 0.5, ...],
  "train_accuracies": [20.0, 60.0, 95.0, ...],
  "relation": "fibonacci",
  "epochs": 100
}
```

### Evaluation Results JSON
```json
{
  "individual": {
    "training_seeds": {"accuracy": 100.0, "predictions": [...]},
    "novel_seeds": {"accuracy": 100.0, "predictions": [...]}
  },
  "sequential": {
    "training_seeds": {"overall_accuracy": 44.0, ...},
    "novel_seeds": {"overall_accuracy": 18.0, ...}
  }
}
```

---

## Git Repository

**Repository**: https://github.com/elanbarenholtz/Autogenerative_Learning.git

**Recent Commits**:
- `6213d62` - Add Experiment 2A: Digit-wise tokenization (catastrophic failure)
- `b53c8dd` - (previous work)

**Branch**: main

---

## Citation & References

If using this code or findings, please cite:

```
[Paper in preparation]
Title: Learning Recurrence Relations: What Transformers Can and Cannot Do
Authors: [Your Name]
Institution: [Your Institution]
Year: 2025
```

**Related Work**:
- Transformers struggle with arithmetic (Nogueira et al., 2021)
- Numerical embeddings matter (Spithourakis & Riedel, 2018)
- Length generalization failures (Anil et al., 2022)

---

## Contact & Contributing

**Maintainer**: Elan Barenholtz
**Issues**: Report via GitHub Issues
**Pull Requests**: Welcome!

**Code Style**: Black formatter, PEP 8
**Documentation**: Docstrings required
**Tests**: Add for new features

---

## Changelog

### 2025-01-13
- Completed Experiment 2A (digit-wise tokenization)
- Tested Fibonacci and Linear relations
- Found catastrophic failure (0% evaluation accuracy)
- Updated summary document with Linear results
- Committed and pushed to GitHub

### [Earlier dates]
- Implemented baseline experiments
- Created probing analysis
- Generated comparative visualizations

---

## Quick Reference: Key Metrics

| Metric | Number-Based | Digit-Wise (Fib) | Digit-Wise (Linear) |
|--------|--------------|------------------|---------------------|
| **Vocabulary Size** | 111 | 12 | 12 |
| **Sequence Length** | ~19 tokens | ~25 tokens | ~28 tokens |
| **Training Acc** | 100% | 75.5% | 18.75% |
| **Individual (train)** | 100% | 0% | 0% |
| **Individual (novel)** | 100% | 0% | 0% |
| **Sequential (train)** | 44% | 0% | 0% |
| **Sequential (novel)** | 18% | 0% | 0% |

---

## Final Notes

- **Number-based tokenization is the only viable approach**
- **Digit-wise fails catastrophically - don't waste compute**
- **Sequential generation is the hard problem** (44% → 18%)
- **Focus on understanding WHY sequential fails**
- **This is a valuable negative result** for publication

---

**Last Updated**: 2025-01-13
**Project Status**: Active - Ready for probing experiments and sequential generation analysis
