# Pre-registration — Recoverability predicts learnability

**Frozen: 2026-06-28, before running the unified learner battery.** Committed to git; the
commit hash is the timestamp of record. The complexity table (`runs/complexity/complexity_table.json`)
was frozen first and independently (model-free). This document commits, in advance, to the
predictions and the falsifiers. We will report against it verbatim, including disconfirmations,
and will not tune the pipeline to produce the predicted pattern.

## 0. Honesty ledger: what is already observed vs genuinely predicted
Some learnability data already exists from the Step 2 validity gate and is **locked, not
predicted** (we state it as observed):
- CA (rule 30/90/110, parity-CA): learned to ~100%, generalizes across content/length/density
  and to held-out neighborhoods; CA free-run rollouts stay ~100% valid.
- Dyck-2: in-distribution competent; wrong-type closures rise sharply under longer/deeper OOD.
- Fibonacci / modular: transformer == n-gram (no rule induction).
- Width-generalization: CA learners are width-locked (absolute and RoPE both).

The genuinely **a-priori** content of this pre-registration is: (i) the *unified* metrics
(gap-to-oracle and the productivity/coverage assay) applied across the whole battery under one
clean protocol; (ii) the not-yet-built generators — counters (B2), position tasks (B4); and
(iii) the load-bearing **Group C (rendered physics)** and **Group D (language)**, for which we
have no data at all. The headline test of the thesis lives in (iii).

## 1. Independent variables (from the frozen complexity table)
Scalarized per generator, all model-free:
- **R_width** = smallest context width w at which recoverability r(w)=H(next|last w) falls
  within 0.05 bits of the entropy-rate floor hμ. Small R_width = recoverable at short range.
  (Markov-k → k; CA → ~one row; fib-mod → 2; Dyck → not reached within the window = censored/∞.)
- **Cμ** (statistical complexity, CSSR peak) and whether it is **bounded/convergent** vs
  **growing** with history length.
- **E** (excess entropy) = stored predictive information ≈ memory/coverage burden.

## 2. Dependent variables (learnability, to be measured by the battery)
- **gap-to-oracle**: model cross-entropy minus the generator's oracle entropy floor, reported
  in-distribution and OOD. (Raw accuracy is not comparable across generators with different hμ.)
- **OOD generalization**: accuracy on each generator's held-out OOD splits.
- **productivity**: novelty-validity rate (fraction of sampled generations that are novel AND
  satisfy the generator's validator) and coverage of the valid set.
- All seed-averaged (≥3 seeds), with matched n-gram baselines, capacity, and token budget.

## 3. Primary hypothesis (quantitative, layered)
Learnability is **not** a single-axis function of recoverability. We pre-register the layered
model: a generator is learned-and-productive to the extent that it is (1) recoverable within a
bounded window, (2) coverable (state space seeable in the data budget), (3) trainable (function
not SGD-hostile), and (4) within the learner's representational invariances.

**H1 (recoverability ↔ learnability).** Across generators, OOD gap-to-oracle is *monotonically
decreasing* in recoverability (small R_width, bounded Cμ) and productivity is *increasing* in it.
Quantitatively: Spearman ρ(R_width, OOD-gap) ≥ +0.6 and ρ(Cμ-or-E, OOD-gap) ≥ +0.6, with the
sign such that more memory/longer reach → larger gap.

**H2 (necessity, not sufficiency — named expected deviations).** Two generators will deviate
from a naive recoverability-only fit, in *predicted* directions, and these deviations are part
of the hypothesis, not against it:
- **Parity-class (Rule 90 / radius-5 parity):** recoverable and bounded, but learnable *only*
  above a data threshold (trainability gate). Predict: fails at low data, succeeds at high data.
- **Fibonacci/modular:** small R_width (=2) but large E / state space → fails via *coverage*,
  matching n-gram. Predict: high-E predicts this failure where R_width alone would not.

**H3 (the target contrast).**
- **Group C (physics-trace):** low in-distribution gap-to-oracle (good continuation) BUT a
  **productivity gap** — low novelty-validity/coverage; cannot compose arbitrary valid novel
  worlds; generations collapse toward plausible continuations. Tied to a hidden upstream
  generator → high effective Cμ / recoverability not achievable from the surface in a bounded
  window.
- **Group D (language):** patterns **with Group A** — low gap-to-oracle and **no productivity
  gap** (high novelty-validity, broad coverage, fluent displaced/negated/abstract structure).
  Tied to language's next-token law being largely recoverable from bounded context (modest
  R_width, bounded effective Cμ). The destroyed-structure control (phrase-shuffled / n-gram-
  resampled) should shift recoverability and learnability *together*.

## 4. Per-group predictions
| Group | gap-to-oracle (ID / OOD) | OOD generalization | productivity / coverage | complexity tie |
|---|---|---|---|---|
| A: iid/Markov, CA | ~0 / ~0 | high (≥95%) | high; broad coverage | low/bounded Cμ, small R_width |
| B1: Dyck | small / **grows w/ depth,len** | degrades OOD | valid short; wrong-type ↑ OOD | R_width censored, E high (D2>D1) |
| B2: counters aⁿbⁿ(cⁿ) | small / fails beyond trained n | fails OOD length | breaks at the count boundary | unbounded counter memory |
| B3: recurrence/modular | large (≈n-gram) | poor on novel seeds | low (can't gen valid novel) | small R_width but high E → coverage |
| B4: position tasks | mixed | position-dependent; PE-sensitive | partial | representational-invariance gate |
| C: physics-trace | **low / low** | continues well | **GAP: low novelty-validity** | hidden generator → high Cμ |
| D: language | low / low | generalizes | **no gap: high productivity** | recoverable; modest R_width |

## 5. Falsification criteria (any of these counts against the thesis)
- A high-Cμ / censored-R_width generator that is **fully learnable with full productivity**.
- **Language patterns with the high-Cμ group**: large OOD gap-to-oracle, or a productivity gap,
  or its learnability *not* tracking recoverability (e.g. phrase-shuffled control learns as well
  as intact language).
- **Physics-trace shows no productivity gap** (composes arbitrary valid novel worlds as fluently
  as it continues).
- Fibonacci/modular turn out **fully productive** (generate valid novel sequences from unseen
  seeds), i.e. coverage was not the binding constraint.
- The monotonic relationships (H1) come out near zero or wrong-signed, *and* the H2 named
  deviations don't account for it.

## 6. Reporting commitments
- Report every generator and every metric, including nulls and disconfirmations, prominently.
- n-gram baseline of matched order throughout; a transformer merely matching n-gram is reported
  as such.
- Match capacity / token budget within each comparison set; seed-average (≥3) with variance.
- Soft metrics (language validity, displacement) labeled soft; no over-claiming.
- The master scatter (learnability vs measured complexity) is the headline figure regardless of
  outcome.
- This file is frozen; if anything here must change, the change is logged with rationale and
  dated, not silently edited.
