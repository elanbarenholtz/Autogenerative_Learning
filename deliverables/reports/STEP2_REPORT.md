# Step 2 Report — Generator Verification + Reproduction Validity Gate

Recoverability / statistical-complexity study. Per research brief Section 9, step 2.
All runs from scratch, seeds fixed (DATA_SEED=0, model-init seeds [42,123,7]).
Canonical results: full-scale 3-seed battery run on Colab GPU (CUDA), stored in `runs/`
(`ca/`, `parity_ca/`, `dyck/`, `baselines/`). Earlier local (MPS) and prior-Feb runs are
archived in `runs/legacy_gen1/superseded_pre_colab/`. Code in `src/`.

## 1. Generator + oracle verification (all PASS, local + GPU)
- Rule 30 / 90 / 110 oracle = 100% (next row deterministic from previous); no OOV tokens.
- Parity CA (radius-5): A/B partition disjoint & complete; train rows A-only (b_fraction=0);
  OOD rows contain B; symbolic oracle = 100% on both A and B cells.
- Dyck-1/2: all generated strings valid; no OOV; wrong-type corruption detected 198/198.
- Fibonacci/modular: UNK = 0 in train and test (no truncation/OOV artifact).

## 2. Reproductions vs prior numbers (full scale, 3 seeds)

| Target | Prior | This run (clean protocol, full scale) | Verdict |
|---|---|---|---|
| Fibonacci vs n-gram | 89.33% = 89.33% | 89.33% = 89.33% (UNK=0) | reproduced exactly |
| Modular vs n-gram | 41.3% ≈ 41.3% | 41.33% = 41.33% (UNK=0) | reproduced exactly |
| Parity CA, held-out B-nbhd | B≈90.4%, n-gram≈49% | **B=97.79%** (3-seed), n-gram≈49% | reproduced (exceeds) |
| Rule 30 OOD | 100% | 100% (all splits) + free-run 100% | reproduced exactly |
| Rule 90 OOD | 99.4% | **99.57%** (3-seed) + free-run 100% (longer) | reproduced exactly |

Details:
- **Rule 30:** TF cell = 100% on id/longer/density(0.2,0.8), all 3 seeds; free-run rollout
  100% (zero errors over full 29/61-step rollouts).
- **Rule 90:** TF cell id=99.57%, longer=99.79%, density=99.57% (per-seed id 100/100/98.7);
  free-run id=99.83%, longer=100%. The local ~50% (chance) at reduced scale (200 seeds) was a
  sample-threshold artifact, not a pipeline fault — see Section 4.
- **Parity CA:** trained only on A-neighborhoods; held-out B-inclusive overall=98.07%, A=98.25%,
  B=97.79%, A–B gap=0.46 (per-seed overall 97.3/97.7/99.2); max-B split B=97.93%. n-gram at
  chance (≈49%) on B. Generalizes to unseen neighborhoods → learned the rule, not the surface.

Transformer matches n-gram exactly on Fibonacci/modular (no rule induction). Parity/CA learners
recover the generative function and generalize OOD — the confirming low-complexity cases.

## 3. Dyck — first-ever numbers (full scale, 10k train / 2k test/split, 3 seeds)
illegal = wrong_type + underflow. Token accuracy is bounded by legal branching (entropy floor),
so the diagnostic signal is the illegal / wrong-type rates, not raw token accuracy.

| Lang | split | token_acc | illegal | wrong_type | free-run validity (greedy/sampled) |
|---|---|---|---|---|---|
| D1 | id_test | 68.9% | 0.00% | 0.00% | 100% / 99.8% |
| D1 | longer_ood | 58.2% | 1.17% | 0.00% | |
| D1 | deeper_ood | 63.8% | 0.00% | 0.00% | |
| D1 | combined_ood | 58.1% | 1.14% | 0.00% | |
| D2 | id_test | 56.2% | 0.04% | **0.04%** | 100% / 97.3% |
| D2 | deeper_ood | 52.5% | 3.63% | 3.63% | |
| D2 | longer_ood | 38.8% | 14.24% | **13.13%** | |
| D2 | combined_ood | 38.7% | 14.37% | **13.29%** | |

Confirms (and tightens vs the local pilot) the predicted signature: Dyck-2 wrong-type-closure
rate is ~0% in-distribution and rises sharply under longer/combined OOD — the hidden-state
(unbounded typed stack) failure localized at the recoverability boundary. Dyck-1 cannot make
wrong-type errors (one bracket type); its failures are mild underflow under length OOD only.

## 4. Key findings beyond reproduction
1. **Parity-class learning threshold (confirmed full scale).** Radius-5 parity: chance (~55%) at
   1500 rows → ~98% at 5000. Rule 90 (= left XOR right, a parity): chance at 200 seeds → 99.6% at
   2000 seeds. Rule 30 (non-parity) learns perfectly even at small scale. So recoverability is
   *necessary but not sufficient*: learnability also depends on the function class / sample
   complexity (parity is SGD-hard). Implication for the master scatter — recoverable systems can
   sit below threshold until enough data; the *shape* of the recoverability-at-width curve
   (gradual vs flat-then-cliff), not just endpoint Cμ, is the candidate predictor.
2. **Learnability decomposes into ≥3 gates**, only the first of which is the thesis variable:
   (a) recoverability — is the info in the window? (b) coverage/induction — does the conditional
   generalize across the alphabet? (Fibonacci/modular fail here → n-gram-matching). (c) trainability
   — can SGD extract it in the data budget? (parity threshold). Step 3 must isolate (a).

## 5. Step 2 closure status: CLOSED
All targets met at full scale, 3 seeds: Fibonacci/modular = n-gram; parity-CA generalizes to
held-out neighborhoods (B=97.8%); Rule 30 = 100%; Rule 90 = 99.6%; Dyck-2 wrong-type 0.04%→13%
under OOD. Pipeline validated. Canonical outputs in `runs/`. Proceed to Step 3.

## 6. Independence note (Section 1 guardrail, ahead of Step 3)
No complexity measurement (CSSR / Cμ / excess entropy) has been computed or used anywhere yet.
Step 3 computes them from generator definitions + raw sequences only, frozen to disk before any
further training reads them. The master scatter (learnability vs measured complexity) does not
exist yet — everything above is ordered by known structure, which Step 3 replaces with measurement.
