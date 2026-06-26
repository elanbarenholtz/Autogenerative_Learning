"""Run the Dyck-1/Dyck-2 experiment on MPS (stock script defaults to CPU).
Pilot scale (2000 train / 500 test), 3 seeds. Writes to experiments_v2/exp_dyck/.
First-ever Dyck numbers: in-distribution wrong-type-closure rate + OOD (longer/deeper/combined).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import exp_dyck

DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
os.makedirs(exp_dyck.OUTPUT_DIR, exist_ok=True)
t0 = time.time()
all_results = {}
for bt in (1, 2):
    all_results[f'D{bt}'] = exp_dyck.run_dyck_experiment(
        bracket_types=bt, random_seeds=exp_dyck.RANDOM_SEEDS, pilot=True, device=DEV)
summary = {'elapsed_seconds': time.time() - t0, 'pilot': True, 'device': DEV,
           'random_seeds': exp_dyck.RANDOM_SEEDS, 'results': all_results}
with open(os.path.join(exp_dyck.OUTPUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nDONE in {summary['elapsed_seconds']:.0f}s -> {exp_dyck.OUTPUT_DIR}/summary.json")
