# Deliverables (consolidated, refreshed)

One findable place for the things you actually use. Canonical sources stay in
`src/`, `runs/`, `reports/`, `colab/`; this folder is kept in sync from them.

**To update:** `bash deliverables/refresh.sh`

## What's here
- `run_on_colab/` — open the `.ipynb` in Colab (GPU), upload the matching `.zip` when prompted.
  - `run_learner_battery.ipynb` + `battery_bundle.zip` — the main learner battery (current step).
  - `run_battery_colab.ipynb` + `recoverability_code.zip` — the Step-2 reproduction battery.
- `reports/` — STEP2 / STEP3 reports, frozen `preregistration.md`, project README.
- `figures/` — recoverability-at-width curves and the (Cμ, E) plane.
- `results/` — frozen `complexity_table_full.json` (the independent variable for all generators);
  `battery_results.json` appears here after you run the learner battery.
