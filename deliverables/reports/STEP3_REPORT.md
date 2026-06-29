# Step 3 Report — Complexity Module (frozen)

The independent variable, measured. Everything here is computed **from raw serialized
sequences / process definitions only — no trained model is read** (Section 1 guardrail).
Frozen output: `runs/complexity/complexity_table.json` (+ figures). Code: `src/complexity/`.

## 1. Apparatus validated against analytic ε-machines (exact)
Before measuring real generators, the estimators were checked on processes with known
closed-form answers:

| machine | Cμ (CSSR / analytic) | hμ (est / an) | E (est / an) |
|---|---|---|---|
| iid | 0 / 0 | 1.00 / 1.00 | 0 / 0 |
| period-2 | 1.00 / 1.00 | 0 / 0 | 1.00 / 1.00 |
| golden-mean | 0.918 / 0.918 | 0.667 / 0.667 | 0.252 / 0.252 |
| order-2 Markov | 1.895 / 1.896 | 0.891 / 0.891 | 0.113 / 0.113 |

Block-entropy estimators (hμ, E, recoverability-at-width) and CSSR (Cμ) reproduce exact
values. This is the brief's CSSR-vs-analytic validity gate.

## 2. Frozen measurements (learner-stream serialization)

| generator | hμ | E (excess entropy) | Cμ (CSSR peak) | recoverability reaches floor at w ≈ |
|---|---|---|---|---|
| markov-1 | 0.93 | 0.06 | 1.00 (2 states) | 1 |
| markov-2 | 0.80 | 0.18 | 1.78 (4) | 2 |
| markov-3 | 0.84 | 0.26 | 2.78 (8) | 3 |
| CA rule30 | ~0 | 6.29 | 5.16 (47) | ~one row |
| CA rule90 | ~0 | 3.14 | 2.33 (17) | ~one row |
| CA rule110 | ~0 | 6.45 | 5.15 (46) | ~one row |
| CA rule150 | ~0 | 5.63 | 2.53 (8) | ~one row |
| Dyck-1 | 0.83 | 0.97 | 3.57 (13) | does not fully reach 0 |
| Dyck-2 | ~0.2 | 5.19 | 3.10 (13) | does not fully reach 0 |
| fib mod 31 | 0 | 9.91 | 4.95 (31)* | 2 |

(*CSSR-next-symbol under-merges deterministic recurrences — see caveats. E is the robust
complexity indicator there: E≈9.9 ≈ 2·log2(31) = exactly the two-residue state.)

## 3. What the numbers say (preview — not yet plotted against learnability)
- **Controls (Markov-k):** Cμ ≈ k bits, small E, recoverability bottoms out exactly at w=k.
  Clean low-complexity anchors; the axis behaves.
- **CA (rules 30/90/110/150):** entropy rate ≈ 0 (deterministic given enough context) with
  **high excess entropy** (E ≈ 3–6.5 bits) — i.e. you must store ~one row of predictive
  information. Cμ is moderate and **bounded**: flattening a 2-D local rule to 1-D costs
  ~one-row memory, but it does not grow with sequence length. Recoverability is achieved at
  a finite width (~one row).
- **Dyck:** recoverability **does not reach 0** within the window (residual hμ), and Dyck-2
  carries high E (5.19) vs Dyck-1 (0.97) — the typed stack stores more, unbounded, predictive
  information. This is the hidden-state / open-span signature.
- **Fibonacci-mod-31:** the instructive outlier. Recoverability hits **0 at w=2** (fully
  determined by the last two symbols — tiny span), yet E and the state space are **large**
  (≈9.9 bits ≈ 2 residues; ~p distinct one-step states). So it is recoverable-at-small-width
  but high-memory: the difficulty is **coverage** (you must see/store a huge state space),
  not span. This is exactly why it failed to learn (collapsed to the n-gram) despite being
  "fixed-span." High E/Cμ predicts that failure where span alone would not.

This matches the layered account: recoverability-at-width captures *how far back*; E/Cμ
capture *how much memory / how many states*; the two come apart (fib-mod: small span, big
memory), and both matter for learnability.

## 4. Caveats (honest labeling, per brief)
- **CSSR variant uses next-symbol equivalence.** Validated exact for Markov-class. For
  deterministic multi-step recurrences it under-merges (under-estimates Cμ); E (block-entropy,
  robust) is used as the primary complexity indicator there. Reported Cμ is the curve **peak**
  before the large-L undersampling collapse (state counts rise then fall as long histories go
  sparse); full per-L curves are in the table for transparency.
- **CA measured at width 8** for tractability (plug-in block entropy can't reach 64-token
  blocks). The one-row-memory property is the width-invariant of interest; Cμ/E scale with
  width but stay bounded. Radius-5 parity-CA from the experiments is represented here by the
  radius-1 parity family (rules 90, 150).
- Counters (B2), position tasks (B4), physics (C), language (D) generators not built yet, so
  not in the table.

## 5. Independence / guardrail
`complexity_table.json` carries a `_meta` block: `model_free: true`, computed from raw
sequences only, timestamped at creation. It is written by `src/complexity/` and never by any
training code; the learner battery will *read* it and never write to it. No transformer output
entered any quantity here.

## 6. Status: Step 3 complete; STOP gate (Section 9.4)
Frozen complexity table + recoverability-at-width figure + (Cμ, E) scatter delivered;
estimator/CSSR validity gate passed. Next per the brief: write + freeze `preregistration.md`
(predictions tying these complexity numbers to learnability), THEN run the learner battery and
build the master scatter. Awaiting go-ahead before pre-registration/training.
