# Step 2 Report — Generator Verification + Reproduction Validity Gate

Recoverability / statistical-complexity study. Per research brief Section 9, steps 2.
All runs from scratch, seeds fixed (DATA_SEED=0, model seeds [42,123,7]), MPS (Apple GPU).
Code: `recoverability_study/` (verify_generators.py, repro_*.py); raw outputs in
`recoverability_study/repro_results/` and `experiments_v2/`.

## 1. Generator + oracle verification (all PASS)
- Rule 30 / 90 / 110 oracle = 100% (next row deterministic from previous); no OOV tokens.
- Parity CA (radius-5): A/B partition disjoint & complete; train rows A-only (b_fraction=0);
  OOD rows contain B; symbolic oracle = 100% on both A and B cells.
- Dyck-1/2: all generated strings valid; no OOV; wrong-type corruption detected 198/198.
- Fibonacci/modular: UNK = 0 in train and test (no truncation/OOV artifact).

## 2. Reproductions vs prior numbers

| Target | Prior | This run (clean protocol) | Verdict |
|---|---|---|---|
| Fibonacci vs n-gram | 89.33% = 89.33% | 89.33% = 89.33% (UNK=0) | reproduced exactly |
| Modular vs n-gram | 41.3% ≈ 41.3% | 41.33% = 41.33% | reproduced exactly |
| Parity CA, held-out B-nbhd | B≈90.4%, n-gram≈49% | **B=97.6%**, n-gram=49.3% | reproduced (exceeds) |
| Rule 30 OOD | 100% | 100% (id/longer/density) | reproduced exactly |
| Rule 90 OOD | 99.4% | ~50% at reduced scale | scale-limited → Colab |

Transformer matches n-gram on Fibonacci/modular (no rule induction). Parity-CA transformer
trained only on A-neighborhoods generalizes to unseen B-neighborhoods at ~97.6% while n-gram
is at chance — confirming case for the recoverability framing.

## 3. Dyck — first-ever numbers (no prior existed; baseline established)
Teacher-forced, pilot scale (2000 train / 500 test/split), 3 seeds. illegal = wrong_type + underflow.

| Lang | split | token_acc | illegal | wrong_type | free-run validity (greedy/sampled) |
|---|---|---|---|---|---|
| D1 | id_test | 67.7% | 0.83% | 0.00% | 66.7% / 80.7% |
| D1 | longer_ood | 59.0% | 1.70% | 0.00% | |
| D1 | deeper_ood | 63.4% | 0.47% | 0.00% | |
| D2 | id_test | 55.9% | 0.41% | **0.38%** | 100% / 88.2% |
| D2 | deeper_ood | 52.1% | 2.94% | 2.93% | |
| D2 | longer_ood | 42.2% | 15.91% | **15.13%** | |
| D2 | combined_ood | 42.2% | 16.40% | **15.63%** | |

Confirms the predicted signature: Dyck-2 wrong-type-closure rate is low in-distribution and
rises sharply under longer/combined OOD. Dyck-1 cannot make wrong-type errors (one bracket type);
its failures are underflow and stay mild. n-gram(5) id_test: D1 61.7%, D2 45.9% (transformer beats it).

## 4. Key findings beyond reproduction
1. **Parity-class learning threshold.** Radius-5 parity: chance (~55%) at 1500 rows, ~98% at 5000.
   Rule 90 (= left XOR right, a parity) collapsed to chance at reduced scale (all 3 seeds identical
   trivial predictor); Rule 30 (non-parity) learns perfectly even at small scale. Recoverability is
   necessary but learnability also depends on function class / sample complexity (parity is SGD-hard).
2. **Compute.** No CUDA locally; MPS gives ~25–30 min per heavy run. Full-scale Rule 90 (2000 seeds)
   is ~hours on MPS → moved to Colab/GPU. Colab battery package built (`recoverability_study/colab/`).

## 5. Outstanding for Step 2 closure
- Full-scale Rule 90 on Colab (expected to reproduce ~99.4%; mechanism already confirmed via parity).
- Optional: full-scale Dyck (10k/2k) on Colab to tighten variance.

## 6. Independence note (for the Section 1 guardrail, ahead of Step 3)
No complexity measurement (CSSR / Cμ / excess entropy) has been computed or used anywhere yet.
Complexity module is Step 3 and will be computed from generator definitions + raw sequences only,
frozen to disk before any further training reads it.
