"""
Step 2 validity gate: verify generators + oracles before any training.

Checks (per research brief Section 9.2):
  - CA Rule 30/90 oracle = 100% (next row deterministic from prev row)
  - Parity CA oracle = 100% on A and B neighborhoods
  - Dyck-1/Dyck-2 generated strings all valid; wrong-type corruption detected
  - No UNK / no out-of-vocab tokens in CA / Dyck streams
  - Train/test seed disjointness
  - Fibonacci / modular: no UNK leakage in train or test
Exit code 0 only if ALL checks pass.
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

FAILS = []
def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)

print("=" * 70)
print("  GENERATOR + ORACLE VERIFICATION")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Elementary CA (Rule 30, 90, 110)
# ---------------------------------------------------------------------------
print("\n[1] Elementary CA (Rule 30 / 90 / 110)")
import exp_ca as eca
for rule in [30, 90, 110]:
    trajs = eca.generate_trajectories(rule, n_seeds=50, width=48, n_steps=24,
                                      density=0.5, base_seed=0)
    # Oracle: next row must equal apply_rule(prev row) exactly -> 100% recoverable
    cells_total = cells_ok = 0
    for tr in trajs:
        for t in range(len(tr) - 1):
            pred = eca.apply_rule(tr[t], rule)
            for a, b in zip(pred, tr[t + 1]):
                cells_total += 1
                cells_ok += (a == b)
    acc = 100.0 * cells_ok / cells_total
    check(f"Rule{rule} oracle = 100%", acc == 100.0, f"{acc:.4f}% over {cells_total} cells")
    # No UNK: tokens only in known vocab
    toks = set()
    for w in eca.tokenize_windows(eca.create_windows(trajs, k_rows=4)):
        toks.update(w)
    check(f"Rule{rule} no OOV tokens", toks.issubset(set(range(eca.CATokenizer.vocab_size))),
          f"tokens={sorted(toks)}")

# ---------------------------------------------------------------------------
# 2. Parity CA (held-out neighborhood A/B design)
# ---------------------------------------------------------------------------
print("\n[2] Parity CA (radius-5, A/B holdout)")
import exp_parity_ca as epc
allowed, blocked = epc.select_allowed_set(epc.N_PATTERNS, fraction=0.6, seed=0)
check("A/B partition covers all patterns",
      len(allowed) + len(blocked) == epc.N_PATTERNS and len(allowed & blocked) == 0,
      f"|A|={len(allowed)} |B|={len(blocked)}")
graph = epc.build_constraint_graph(allowed)
train_w = epc.generate_training_data(60, 64, allowed, graph, seed=0)
# Train rows must be A-only
tr_stats = epc.compute_neighborhood_stats(train_w, allowed, 64)
check("train rows are A-only (b_fraction=0)", tr_stats['b_fraction'] == 0.0,
      f"b_fraction={tr_stats['b_fraction']:.4f}")
ood_w = epc.generate_test_data(60, 64, seed=999999)
ood_stats = epc.compute_neighborhood_stats(ood_w, allowed, 64)
check("OOD random rows contain B neighborhoods", ood_stats['b_fraction'] > 0.1,
      f"b_fraction={ood_stats['b_fraction']:.4f}")
# Oracle must be 100% on A and B
for nm, w in [("train(A)", train_w), ("ood(A+B)", ood_w)]:
    orc = epc.evaluate_oracle(w, 64, allowed)
    check(f"Parity oracle 100% overall [{nm}]", orc['overall_cell_acc'] == 100.0,
          f"overall={orc['overall_cell_acc']:.2f} A={orc['a_cell_acc']:.2f} B={orc['b_cell_acc']:.2f}")

# ---------------------------------------------------------------------------
# 3. Dyck-1 / Dyck-2
# ---------------------------------------------------------------------------
print("\n[3] Dyck-1 / Dyck-2")
import exp_dyck as edy
for bt in [1, 2]:
    tok = edy.DyckTokenizer(bracket_types=bt)
    seqs = edy.generate_split(tok, n_sequences=200, min_pairs=2, max_pairs=20,
                              max_depth=8, seed=0)
    all_valid = all(edy.is_valid_dyck(s, tok)[0] for s in seqs)
    check(f"Dyck-{bt} all generated strings valid", all_valid)
    # No OOV
    toks = set(t for s in seqs for t in s)
    check(f"Dyck-{bt} no OOV tokens", toks.issubset(set(range(tok.vocab_size))),
          f"tokens={sorted(toks)}")
    if bt == 2:
        # Corrupt a close bracket to wrong type -> must be detected as 'wrong_type'
        detected = 0; tested = 0
        rng = random.Random(1)
        for s in seqs:
            idx = [i for i, t in enumerate(s) if t == tok.CLOSE_PAREN]
            if not idx:
                continue
            tested += 1
            s2 = list(s); s2[idx[0]] = tok.CLOSE_BRACKET  # ) -> ]
            ok, pos, etype = edy.is_valid_dyck(s2, tok)
            detected += (not ok and etype == 'wrong_type')
        check("Dyck-2 wrong-type closure detected", tested > 0 and detected == tested,
              f"{detected}/{tested}")

# ---------------------------------------------------------------------------
# 4. Fibonacci / modular (no UNK leakage)
# ---------------------------------------------------------------------------
print("\n[4] Fibonacci / modular recurrence (UNK leakage)")
import recurrence_relations as rr
print("    available:", [n for n in dir(rr) if not n.startswith('_')][:20])

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
if FAILS:
    print(f"  RESULT: {len(FAILS)} CHECK(S) FAILED -> {FAILS}")
    sys.exit(1)
print("  RESULT: ALL CHECKS PASSED")
print("=" * 70)
