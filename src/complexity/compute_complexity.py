"""
Compute and FREEZE complexity_table.json for all available generators.
Model-free: every quantity comes from raw serialized sequences / process definitions.
No trained model is read. Run BEFORE training reads the table.

Per generator: hmu, excess entropy E, recoverability-at-width r(w), and CSSR Cmu
(with L-sweep so the bounded-vs-growing memory signature is visible). Plus analytic
controls (iid, golden-mean, order-2 Markov) with exact values.
"""
import sys, os, json, time, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from complexity import estimators as EST
from complexity import cssr as CSSR
from complexity import machines as MAC
from complexity import generators_stream as GS

OUT = 'runs/complexity'
os.makedirs(OUT, exist_ok=True)

# name, builder(), group, est_Lmax, cssr_Lmax, note
GENERATORS = [
    ('markov1',   lambda: GS.markov_stream(1),                 'A_control', 8, 6, 'order-1 Markov'),
    ('markov2',   lambda: GS.markov_stream(2),                 'A_control', 9, 6, 'order-2 Markov'),
    ('markov3',   lambda: GS.markov_stream(3),                 'A_control', 9, 6, 'order-3 Markov'),
    ('CA_rule30', lambda: GS.ca_stream(30,  width=8),          'A_ca', 12, 9, 'elementary CA, chaotic, W=8'),
    ('CA_rule90', lambda: GS.ca_stream(90,  width=8),          'A_ca', 12, 9, 'elementary CA, XOR/parity radius-1, W=8'),
    ('CA_rule110',lambda: GS.ca_stream(110, width=8),          'A_ca', 12, 9, 'elementary CA, class-4, W=8'),
    ('CA_rule150',lambda: GS.ca_stream(150, width=8),          'A_ca', 12, 9, 'additive CA, XOR-3, W=8'),
    ('dyck1',     lambda: GS.dyck_stream(1),                   'B_dyck', 12, 8, 'Dyck-1 (one bracket type)'),
    ('dyck2',     lambda: GS.dyck_stream(2),                   'B_dyck', 12, 8, 'Dyck-2 (typed stack)'),
    ('fib_mod31', lambda: GS.fib_mod_stream(31),               'B_recur', 4, 3, 'Fibonacci mod 31 (order-2 recurrence)'),
]


def measure(seqs, A, est_Lmax, cssr_Lmax, label):
    summ = EST.summarize(seqs, est_Lmax, A, label)
    cs = CSSR.cssr_cmu(seqs, A, Lmax=cssr_Lmax)
    state_curve = {L: cs['grid'][f'L{L}_a{cs["alpha_used"]:.0e}']['n_states']
                   for L in range(1, cssr_Lmax + 1)}
    cmu_curve = {L: cs['grid'][f'L{L}_a{cs["alpha_used"]:.0e}']['Cmu']
                 for L in range(1, cssr_Lmax + 1)}
    # Robust Cmu = peak of the curve before the undersampling collapse (state count
    # rises then falls as long histories get sparse; the peak is the reliable estimate).
    Lpeak = max(cmu_curve, key=lambda L: cmu_curve[L])
    cmu_peak = cmu_curve[Lpeak]
    n_states_peak = state_curve[Lpeak]
    # did states still increase up to the peak (memory not yet saturated within range)?
    growing = state_curve[Lpeak] == max(state_curve.values()) and Lpeak >= cssr_Lmax - 1
    return {
        'alphabet_size': A,
        'n_sequences': len(seqs),
        'n_tokens': int(sum(len(s) for s in seqs)),
        'entropy_rate_hmu': summ['entropy_rate_hmu'],
        'excess_entropy_E': summ['excess_entropy_E'],
        'recoverability_at_width': summ['recoverability_at_width'],
        'Cmu_cssr_peak': round(cmu_peak, 4),
        'Cmu_peak_L': Lpeak,
        'n_states_peak': n_states_peak,
        'Cmu_still_growing_at_range_end': bool(growing),
        'Cmu_state_curve_by_L': state_curve,
        'Cmu_curve_by_L': cmu_curve,
        'undersampled_at_Lmax': summ['undersampled_at_Lmax'],
    }


def main():
    t0 = time.time()
    table = {'_meta': {
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'model_free': True,
        'computed_from': 'raw serialized sequences / process definitions only; no trained model read',
        'estimators': 'block-entropy (Miller-Madow) -> hmu, E, recoverability-at-width; CSSR (chi2 homogeneity) -> Cmu',
        'validated_against': 'analytic epsilon-machines (iid, period2, golden-mean, order-2): exact agreement',
        'serialization': 'each generator measured on the stream the learner sees (CA flattened rows+sep; Dyck brackets; residues)',
        'note_ca': 'flattening a 2-D local rule to 1-D requires ~one-row memory, so CA Cmu is moderate but BOUNDED/convergent; discriminator vs Dyck is bounded-vs-growing Cmu, not magnitude',
    }}

    # analytic controls (exact)
    for name in ['iid_fair', 'golden_mean', 'order2_markov']:
        m = MAC.KNOWN[name]
        seqs = [MAC.sample(m, 20000, seed=s) for s in range(20)]
        r = EST.recoverability_at_width(EST.block_entropy_curve(seqs, 8, 2)[0])
        table[name] = {
            'group': 'analytic_control',
            'Cmu_analytic': round(MAC.Cmu_analytic(m), 4),
            'entropy_rate_hmu': round(MAC.hmu_analytic(m), 4),
            'excess_entropy_E': round(MAC.excess_entropy_analytic(m), 4),
            'recoverability_at_width': [round(float(x), 4) for x in r],
            'method': 'exact analytic + empirical (validated equal)',
        }
        print(f"  {name:<14} Cmu={table[name]['Cmu_analytic']:.3f} (analytic)")

    for name, build, group, eL, cL, note in GENERATORS:
        ts = time.time()
        seqs, A = build()
        ent = measure(seqs, A, eL, cL, name)
        ent['group'] = group
        ent['note'] = note
        table[name] = ent
        print(f"  {name:<12} Cmu_peak={ent['Cmu_cssr_peak']:.3f} (L={ent['Cmu_peak_L']}, "
              f"states={ent['n_states_peak']}, growing={ent['Cmu_still_growing_at_range_end']}) "
              f"hmu={ent['entropy_rate_hmu']:.3f} E={ent['excess_entropy_E']:.3f} [{time.time()-ts:.0f}s]")

    table['_meta']['elapsed_sec'] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, 'complexity_table.json'), 'w') as f:
        json.dump(table, f, indent=2)
    print(f"\nFROZEN -> {OUT}/complexity_table.json  ({table['_meta']['elapsed_sec']}s)")
    make_figures(table)


def make_figures(table):
    # recoverability-at-width curves
    plt.figure(figsize=(8, 5))
    for name, e in table.items():
        if name == '_meta' or 'recoverability_at_width' not in e:
            continue
        r = e['recoverability_at_width']
        plt.plot(range(len(r)), r, marker='.', label=name)
    plt.xlabel('context width w (symbols)')
    plt.ylabel('H(next | last w)  [bits]  — lower = more recoverable')
    plt.title('Recoverability-at-width by generator')
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'recoverability_at_width.png'), dpi=130)
    plt.close()

    # Cmu vs E scatter
    plt.figure(figsize=(7, 6))
    for name, e in table.items():
        if name == '_meta':
            continue
        cmu = e.get('Cmu_cssr_peak', e.get('Cmu_analytic'))
        E = e.get('excess_entropy_E')
        if cmu is None or E is None:
            continue
        plt.scatter(cmu, E)
        plt.annotate(name, (cmu, E), fontsize=7, xytext=(3, 3), textcoords='offset points')
    plt.xlabel('Cμ  (statistical complexity, bits)')
    plt.ylabel('E  (excess entropy, bits)')
    plt.title('Generators on the complexity plane (Cμ, E)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'cmu_vs_E.png'), dpi=130)
    plt.close()
    print(f"figures -> {OUT}/recoverability_at_width.png, cmu_vs_E.png")


if __name__ == '__main__':
    main()
