# Recoverability / Statistical-Complexity Study

Testing whether a sequence's **recoverability** — how much of the next-step law is
extractable from the observable stream within a bounded window (low statistical
complexity Cμ / high finite-context predictability) — predicts how **learnable** it is by
next-token prediction, and whether the learner reaches full generative **productivity**.

This reframes the earlier "fibonacci-experiment" project. The old framing
("externally rule-generated sequences are unlearnable") is **superseded**: cellular
automata are externally rule-generated yet learnable, because the rule is local and the
next state is recoverable from observable context. Gen-1 material is kept under
`runs/legacy_gen1/` and `reports/` for the record.

## Repository layout

```
src/                 All Python: shared modules + experiment scripts
  model.py, experiment_framework.py, recurrence_relations.py, symbol_tokenizer.py
  exp_ca.py, exp_parity_ca.py, exp_dyck.py, exp5_baselines.py   (generators + clean eval)
  verify_generators.py, repro_ca.py, repro_parity.py, repro_dyck.py, bench.py
runs/                All experiment outputs (one folder per experiment)
  ca/  parity_ca/  dyck/  baselines/  recurrence/  repro_gate/  legacy_gen1/
reports/             Markdown reports (STEP2_REPORT.md is current; rest are Gen-1)
notebooks/           Colab/exploratory notebooks
  colab/             GPU battery runner + uploadable code bundle
requirements.txt
```

## How to run

Run scripts from the **repo root** so imports resolve (script dir = `src/`) and outputs
land under `runs/`:

```bash
source venv/bin/activate            # local venv (Python 3.13, torch + MPS on Apple)
python src/verify_generators.py     # validity gate: oracles, no-OOV, disjoint splits
python src/exp5_baselines.py        # Fibonacci/modular vs n-gram
python src/repro_parity.py          # parity CA held-out-neighborhood reproduction
python src/repro_ca.py              # Rule 30/90 (scaled)
python src/repro_dyck.py            # Dyck-1/2 (MPS)
```

GPU (full scale): upload `colab/recoverability_code.zip` to the Colab notebook
`colab/run_battery_colab.ipynb` (set runtime to GPU) and run top to bottom.

## Status

Step 2 (validity gate + reproductions) complete except full-scale Rule 90, which is
compute-blocked locally and queued for Colab. See `reports/STEP2_REPORT.md` for results.
Next: Step 3, the complexity module (Cμ via CSSR, excess entropy, recoverability-at-width),
computed independently of any trained model and frozen before further training.

## Reproducibility

Seeds fixed: `DATA_SEED=0`, model-init seeds `[42, 123, 7]`. Model checkpoints (`*.pt`)
are gitignored to keep the repo lean; result JSON/logs/figures are committed.
