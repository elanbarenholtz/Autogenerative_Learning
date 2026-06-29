"""
Step 3, second pass: complexity for the generators built after the v1 freeze
(counters, position tasks, physics, language). Model-free; before training those.
Writes runs/complexity/complexity_table_addendum.json and merges with v1 into
complexity_table_full.json; refreshes combined figures.
"""
import sys, os, json, time, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from complexity import estimators as EST
from complexity import cssr as CSSR
import gen_counter as GC, gen_position as GP, gen_physics as GPHYS, gen_language as GL

OUT = 'runs/complexity'


def physics_coord_stream(n_traj=3000, G=12, n_frames=40, speed=0.07, seed0=0):
    """Ball cell (cx,cy) per frame, interleaved -> info-equivalent to the rendered grid."""
    seqs = []
    for i in range(n_traj):
        cells = GPHYS.simulate(G, n_frames, speed, seed=seed0 + i)
        s = []
        for (cx, cy) in cells:
            s.append(cx); s.append(cy)
        seqs.append(s)
    return seqs, G


def lang_ids(control=None):
    text = GL.load_corpus('runs/lang/corpus.txt')
    tok = GL.CharTokenizer(text)
    train = GL.split_text(text)['train']
    if control == 'word_shuffle':
        train = GL.word_block_shuffle(train)
    ids = tok.encode(train)
    if control == 'ngram':
        ids = GL.char_ngram_resample(ids, n=5)
    return [ids], tok.vocab_size


GEN = [
    ('counter_anbn',   lambda: GC.stream(4000, 1, 20, 'anbn'),     'B_counter', 14, 8, 'a^n b^n'),
    ('counter_anbncn', lambda: GC.stream(4000, 1, 15, 'anbncn'),   'B_counter', 14, 8, 'a^n b^n c^n'),
    ('pos_kth',        lambda: GP.stream('kth', 4000, 4, 24),      'B_position', 10, 6, 'query-answer; stationary measures less natural'),
    ('pos_middle',     lambda: GP.stream('middle', 4000, 4, 24),   'B_position', 10, 6, 'query-answer'),
    ('pos_last2',      lambda: GP.stream('last2', 4000, 4, 24),    'B_position', 10, 6, 'query-answer'),
    ('physics_v1',     lambda: physics_coord_stream(3000, 12, 40, 0.07), 'C_physics', 10, 6, 'ball-cell coords; hidden velocity'),
    ('language',       lambda: lang_ids(None),               'D_language', 8, 3, 'char-level English'),
    ('language_word_shuffle', lambda: lang_ids('word_shuffle'), 'D_language', 8, 3, 'D2 control'),
    ('language_ngram5', lambda: lang_ids('ngram'),           'D_language', 8, 3, 'D2 control'),
]


def measure(seqs, A, eL, cL):
    summ = EST.summarize(seqs, eL, A, "")
    cs = CSSR.cssr_cmu(seqs, A, Lmax=cL)
    cmu_curve = {L: cs['grid'][f'L{L}_a{cs["alpha_used"]:.0e}']['Cmu'] for L in range(1, cL + 1)}
    state_curve = {L: cs['grid'][f'L{L}_a{cs["alpha_used"]:.0e}']['n_states'] for L in range(1, cL + 1)}
    Lpeak = max(cmu_curve, key=lambda L: cmu_curve[L])
    growing = state_curve[Lpeak] == max(state_curve.values()) and Lpeak >= cL - 1
    return {
        'alphabet_size': A, 'n_sequences': len(seqs),
        'n_tokens': int(sum(len(s) for s in seqs)),
        'entropy_rate_hmu': summ['entropy_rate_hmu'],
        'excess_entropy_E': summ['excess_entropy_E'],
        'recoverability_at_width': summ['recoverability_at_width'],
        'Cmu_cssr_peak': round(cmu_curve[Lpeak], 4), 'Cmu_peak_L': Lpeak,
        'n_states_peak': state_curve[Lpeak],
        'Cmu_still_growing_at_range_end': bool(growing),
        'Cmu_state_curve_by_L': state_curve, 'Cmu_curve_by_L': cmu_curve,
        'undersampled_at_Lmax': summ['undersampled_at_Lmax'],
    }


def main():
    t0 = time.time()
    add = {'_meta': {
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'model_free': True,
        'computed_from': 'raw serialized sequences only; no trained model read; before training these generators',
        'note_physics': 'measured on ball-cell coordinate stream (info-equivalent to rendered grid; frame scaffolding dropped for tractability)',
        'note_position': 'query-answer tasks; stationary-process measures are reported but less natural than for streams',
        'note_language': 'char-level; CSSR Cmu unreliable at this alphabet/data -> use E and recoverability curve',
    }}
    for name, build, group, eL, cL, note in GEN:
        ts = time.time()
        seqs, A = build()
        e = measure(seqs, A, eL, cL)
        e['group'] = group; e['note'] = note
        add[name] = e
        print(f"  {name:<22} A={A:<3} hmu={e['entropy_rate_hmu']:.3f} E={e['excess_entropy_E']:.3f} "
              f"Cmu_peak={e['Cmu_cssr_peak']:.3f} (grow={e['Cmu_still_growing_at_range_end']}) [{time.time()-ts:.0f}s]")
    add['_meta']['elapsed_sec'] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, 'complexity_table_addendum.json'), 'w') as f:
        json.dump(add, f, indent=2)

    # merge with v1
    v1 = json.load(open(os.path.join(OUT, 'complexity_table.json')))
    full = dict(v1)
    full['_meta'] = {'note': 'combined v1 + addendum; both model-free, pre-training',
                     'v1_meta': v1.get('_meta'), 'addendum_meta': add['_meta']}
    for k, v in add.items():
        if k != '_meta':
            full[k] = v
    with open(os.path.join(OUT, 'complexity_table_full.json'), 'w') as f:
        json.dump(full, f, indent=2)
    print(f"\nfull table -> {OUT}/complexity_table_full.json ({add['_meta']['elapsed_sec']}s)")
    make_fig(full)


def make_fig(table):
    plt.figure(figsize=(9, 6))
    for name, e in table.items():
        if name == '_meta' or 'recoverability_at_width' not in e:
            continue
        r = e['recoverability_at_width']
        plt.plot(range(len(r)), r, marker='.', label=name, alpha=0.8)
    plt.xlabel('context width w (symbols)'); plt.ylabel('H(next | last w) [bits]')
    plt.title('Recoverability-at-width — all generators')
    plt.legend(fontsize=6, ncol=3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'recoverability_all.png'), dpi=130); plt.close()
    print(f"figure -> {OUT}/recoverability_all.png")


if __name__ == '__main__':
    main()
