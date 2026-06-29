#!/bin/bash
# Refresh the consolidated deliverables folder from the canonical repo locations.
# Run from anywhere:  bash deliverables/refresh.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/deliverables"
mkdir -p "$D/run_on_colab" "$D/reports" "$D/figures" "$D/results"

# --- things you upload to Colab ---
cp -f "$ROOT/colab/run_learner_battery.ipynb"   "$D/run_on_colab/" 2>/dev/null || true
cp -f "$ROOT/colab/battery_bundle.zip"          "$D/run_on_colab/" 2>/dev/null || true
cp -f "$ROOT/colab/run_battery_colab.ipynb"     "$D/run_on_colab/" 2>/dev/null || true
cp -f "$ROOT/colab/recoverability_code.zip"     "$D/run_on_colab/" 2>/dev/null || true

# --- reports ---
cp -f "$ROOT/reports/STEP2_REPORT.md" "$D/reports/" 2>/dev/null || true
cp -f "$ROOT/reports/STEP3_REPORT.md" "$D/reports/" 2>/dev/null || true
cp -f "$ROOT/preregistration.md"      "$D/reports/" 2>/dev/null || true
cp -f "$ROOT/README.md"               "$D/reports/" 2>/dev/null || true

# --- figures ---
cp -f "$ROOT/runs/complexity/"*.png   "$D/figures/" 2>/dev/null || true

# --- results / frozen tables ---
cp -f "$ROOT/runs/complexity/complexity_table_full.json" "$D/results/" 2>/dev/null || true
cp -f "$ROOT/runs/battery/results.json" "$D/results/battery_results.json" 2>/dev/null || true

echo "deliverables refreshed under $D"
ls -R "$D" | sed 's/^/  /'
