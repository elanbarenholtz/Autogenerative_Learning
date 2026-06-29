"""
Run the unified learner battery: train matched transformers across all adapters,
measure gap-to-oracle + OOD + productivity, join the frozen complexity table, and
write runs/battery/results.json (the input to the master scatter).
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from battery import core
from battery import adapters as AD

OUT = 'runs/battery'
FROZEN = 'runs/complexity/complexity_table_full.json'
NAME2CX = {'physics': 'physics_v1', 'language_ngram': 'language_ngram5'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--small', action='store_true')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 7])
    ap.add_argument('--device', default=None)
    ap.add_argument('--pos', default='absolute')
    ap.add_argument('--n_prod', type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cx = json.load(open(FROZEN)) if os.path.exists(FROZEN) else {}

    adapters = AD.build_all(small=args.small)
    results = {'_meta': {'epochs': args.epochs, 'seeds': args.seeds, 'pos': args.pos,
                         'small': args.small, 'created': time.strftime('%Y-%m-%d %H:%M')}}
    for ad in adapters:
        t0 = time.time()
        print(f"\n=== {ad.name} ({ad.group}) ===")
        per_seed = []
        prod = None
        for sd in args.seeds:
            model, dev = core.train_lm(ad, seed=sd, epochs=args.epochs, pos=args.pos,
                                       device=args.device, verbose=args.small)
            res = {}
            for split in ad.splits:
                if split == 'train':
                    continue
                res[split] = core.teacher_forced(model, ad, split, dev)
            per_seed.append(res)
            if sd == args.seeds[0]:
                prod = core.free_run_productivity(model, ad, dev, n_samples=args.n_prod)
                ng = core.ngram_ce(ad, 'id_test', n=5)
            print(f"  seed{sd}: " + " | ".join(
                f"{s} gap={res[s]['gap_to_oracle_bits']:.3f} acc={res[s]['accuracy']:.2f}" for s in res))
        # aggregate gap per split across seeds
        agg = {}
        for split in per_seed[0]:
            gaps = [ps[split]['gap_to_oracle_bits'] for ps in per_seed]
            accs = [ps[split]['accuracy'] for ps in per_seed]
            agg[split] = {'gap_mean': float(np.mean(gaps)), 'gap_std': float(np.std(gaps)),
                          'acc_mean': float(np.mean(accs)),
                          'ce_mean': float(np.mean([ps[split]['ce_bits'] for ps in per_seed])),
                          'floor': per_seed[0][split]['oracle_floor_bits']}
        ckey = NAME2CX.get(ad.name, ad.name)
        cxe = cx.get(ckey, {})
        results[ad.name] = {
            'group': ad.group, 'note': ad.note,
            'learnability': agg, 'ngram_baseline': ng, 'productivity': prod,
            'complexity': {k: cxe.get(k) for k in
                           ['entropy_rate_hmu', 'excess_entropy_E', 'Cmu_cssr_peak',
                            'Cmu_still_growing_at_range_end']},
            'elapsed_sec': round(time.time() - t0, 1),
        }
        p = results[ad.name]['productivity']
        pstr = f"prod nov-val={p['novelty_validity_rate']:.2f}" if p and p.get('available') else "prod n/a"
        print(f"  -> id gap={agg.get('id_test',{}).get('gap_mean',float('nan')):.3f} | {pstr} "
              f"| Cmu={cxe.get('Cmu_cssr_peak')} E={cxe.get('excess_entropy_E')} [{results[ad.name]['elapsed_sec']}s]")

    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}/results.json")


if __name__ == '__main__':
    main()
