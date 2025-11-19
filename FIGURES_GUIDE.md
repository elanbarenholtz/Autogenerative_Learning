# Experimental Results - Figure Guide

## Overview of Two Evaluation Methods

### Method 1: Single-Step Training Accuracy (100%)
- **What it measures**: Can the model predict the next token given a perfect context window?
- **Context**: Uses TRUE Fibonacci values from training data
- **Task**: Given correct sequence [a, b, c, d, e, f, g, h, i, j], predict k
- **Result**: 100% accuracy from epoch 5 onwards
- **Why it works**: Model memorized local patterns in the training distribution

### Method 2: Sequential/Autoregressive Evaluation (44% training, 18% novel)
- **What it measures**: Can the model generate long coherent sequences?
- **Context**: Uses model's OWN predictions fed back iteratively
- **Task**: Starting at position 10, generate positions 11-30 sequentially
- **Results**:
  - Training seeds: 44% exact match accuracy
  - Novel seeds: 18% exact match accuracy
- **Why it fails**: Errors compound, distribution drift, no true rule understanding

---

## Figure Descriptions

### Figure 1: Accuracy Comparison (accuracy_comparison.png)
**Evaluation Methods**: Single-Step (Method 1) + Sequential/Autoregressive (Method 2)

**What it shows**:
- **Single-Step Training**: 100.0% exact match accuracy
- **Sequential Training Seeds**: 44.0% exact match accuracy
- **Sequential Novel Seeds**: 18.0% exact match accuracy
- **56 percentage point drop** from single-step to sequential (on training data)
- **26 percentage point drop** from training to novel seeds (sequential)

**Key Takeaway**: This is THE KEY FIGURE showing the complete story. The model achieves perfect single-step accuracy (memorization works) but fails dramatically at sequential generation (56pp drop), proving it didn't learn the underlying rule. The additional 26pp drop on novel seeds shows the memorization doesn't generalize. This is the PRIMARY evidence that the model didn't learn the F(n) = F(n-1) + F(n-2) rule.

**File**: `visualizations/accuracy_comparison.png`

---

### Figure 2: Error Progression (error_progression.png)
**Evaluation Method**: Sequential/Autoregressive (Method 2)

**What it shows**:
- Left panel: Error growth over positions 0-20 for training seeds
- Right panel: Error growth over positions 0-20 for novel seeds
- Both show exponential error growth

**Key Takeaway**: Even on training seeds, errors compound rapidly. Novel seeds show even faster divergence. This demonstrates that the model lacks a stable internal computation for the Fibonacci rule.

**File**: `visualizations/error_progression.png`

---

### Figure 3: Prediction Examples (prediction_examples.png)
**Evaluation Method**: Sequential/Autoregressive (Method 2)

**What it shows**:
- 4 subplots showing true sequence (pink circles) vs predicted sequence (yellow X's)
- Top 2: Training seeds with 55% and 50% accuracy
- Bottom 2: Novel seeds with 20% and 15% accuracy

**Key Takeaway**: Visual demonstration of how predictions diverge from true values. Even "successful" cases show the model tracking the exponential growth but missing exact values.

**File**: `visualizations/prediction_examples.png`

---

### Figure 4: MAE Distribution (mae_distribution.png)
**Evaluation Method**: Sequential/Autoregressive (Method 2)

**What it shows**:
- Box plots comparing Mean Absolute Error between training and novel seeds
- Training seeds: Median ~250K, tight distribution
- Novel seeds: Median ~2.8M, wide distribution with outliers

**Key Takeaway**: Novel seeds have 10x higher errors with much greater variance. This quantifies the generalization failure.

**File**: `visualizations/mae_distribution.png`

---

### Figure 5: Individual Seeds Performance (individual_seeds.png)
**Evaluation Method**: Sequential/Autoregressive (Method 2)

**What it shows**:
- Left panel: Each training seed's accuracy (range: 30-60%)
- Right panel: Each novel seed's accuracy (range: 15-30%)

**Key Takeaway**: Shows variation across seeds. Even best training seed only achieves 60%. Novel seeds consistently underperform.

**File**: `visualizations/individual_seeds.png`

---

### Figure 6: Summary Table (summary_table.png)
**Evaluation Method**: Sequential/Autoregressive (Method 2)

**What it shows**:
- Exact Match Accuracy: 44.00% (training) → 18.00% (novel)
- Mean Absolute Error: 273,740 (training) → 3,112,269 (novel)
- Degradation: 26.00 percentage points

**Key Takeaway**: Clean summary of the main quantitative results for the paper.

**File**: `visualizations/summary_table.png`

---

### Figure 7: Probing Results (probing_results.png)
**Evaluation Method**: Mechanistic Interpretability (Linear Probes)

**What it shows**:
- Probe accuracy for decoding F(n-1) and F(n-2) from hidden states
- All layers show ~62% F(n-1) accuracy, ~58% F(n-2) accuracy
- Both barely above 50% random baseline
- Flat across all 3 transformer layers

**Key Takeaway**:
- Model represents NEITHER pure addition rule (would need high F(n-2)) nor pure ratio approximation (would need low F(n-2))
- Weak, mixed representations explain poor sequential performance
- No hierarchical processing across layers
- **This is the mechanistic evidence for WHY the model fails**

**Interpretation Thresholds**:
- F(n-2) > 70% → Uses addition rule F(n) = F(n-1) + F(n-2)
- F(n-2) < 30% → Uses ratio approximation F(n) ≈ φ × F(n-1)
- 30% ≤ F(n-2) ≤ 70% → Mixed/uncertain (our result: 58%)

**File**: `visualizations/probing_results.png`

---

## How Figures Relate to Each Other

### Primary Evidence (Figures 1-6): Model Fails at Sequential Generation
These all use **Method 2** (sequential evaluation) and show:
1. **Figure 1**: Overall failure (44% → 18%)
2. **Figures 2-5**: Different views of the same phenomenon (error growth, visual examples, variance)
3. **Figure 6**: Summary table

### Mechanistic Explanation (Figure 7): WHY the Model Fails
- **Figure 7** uses probes to peek inside the model's representations
- Shows the model learned weak pattern matching, not the true rule
- Explains why sequential generation fails: no robust F(n-1) + F(n-2) computation

### The Complete Story
1. **Training**: Model achieves 100% single-step accuracy (not shown in figures, but documented in `training_history.json`)
2. **Sequential Evaluation (Figs 1-6)**: Model fails dramatically at generating sequences (44% → 18%)
3. **Mechanistic Analysis (Fig 7)**: Probing reveals model lacks strong internal representations of the addition rule

---

## Key Numbers to Report in Paper

### Training Performance
- Single-step accuracy: **100%** (epochs 5-100)
- Final training loss: **0.0018**

### Sequential Evaluation Performance
- Training seeds: **44.0%** exact match accuracy
- Novel seeds: **18.0%** exact match accuracy
- Generalization gap: **26 percentage points**

### Mechanistic Probing
- F(n-1) probe accuracy: **62.5%** (all layers)
- F(n-2) probe accuracy: **58.3%** (all layers)
- Interpretation: **Mixed/uncertain** - neither addition nor ratio

### Error Metrics
- Training seeds MAE: **273,740**
- Novel seeds MAE: **3,112,269**
- MAE increase: **~11x worse** on novel seeds

---

## Files to Share with Other AI

### Essential Files:
1. All 7 PNG files in `visualizations/`
2. This guide (`FIGURES_GUIDE.md`)
3. `COMPLETE_EXPERIMENTAL_REPORT.md` - Full technical writeup
4. `results/evaluation_summary.json` - Raw numerical data
5. `results/probing_results.json` - Probing data

### Optional Supporting Files:
6. `README.md` - Project overview
7. `training_history.json` - Shows 100% training accuracy
8. `INDEX.md` - Navigation guide

---

## Critical Distinction for the Paper

⚠️ **IMPORTANT**: The paper should clearly distinguish:

1. **Training Accuracy (100%)**: Model can do perfect next-token prediction when given correct context
   - This is NOT shown in the main figures
   - This is the baseline capability

2. **Sequential/Generative Accuracy (44% → 18%)**: Model CANNOT maintain coherence over long sequences
   - This IS what Figures 1-6 show
   - This is the main experimental finding

3. **Mechanistic Explanation (58% probe accuracy)**: Model didn't learn the true rule
   - This IS what Figure 7 shows
   - This explains WHY sequential generation fails

The key insight: **Perfect memorization ≠ Rule learning**. The model memorized patterns but didn't abstract the underlying F(n) = F(n-1) + F(n-2) computation.
