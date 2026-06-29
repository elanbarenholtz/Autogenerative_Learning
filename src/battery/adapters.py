"""
Per-generator Adapters for the unified battery. Each spans one group so the master
scatter has a representative across the complexity axis. Oracle floor = 0 for
deterministic targets, else the entropy-rate hmu from the frozen complexity table
(the provably-optimal per-symbol CE), loaded model-free.
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp_ca as eca
import exp_dyck as edy
import gen_counter as gc
import gen_physics as gph
import gen_language as gl
from battery.core import Adapter

_FROZEN = os.path.join('runs', 'complexity', 'complexity_table_full.json')


def _hmu(name, default=0.0):
    try:
        t = json.load(open(_FROZEN))
        return float(t.get(name, {}).get('entropy_rate_hmu', default))
    except Exception:
        return default


def scored_all(s):                      # predict every next token
    return list(range(len(s) - 1))


# ---------------- CA (immanent) ----------------
def ca_adapter(rule=90, width=24, k=4, n_train=600, n_test=200, small=False):
    if small:
        width, n_train, n_test = 12, 80, 40
    steps, steps_ood = 24, 40
    tr = eca.generate_trajectories(rule, n_train, width, steps, 0.5, eca.DATA_SEED)
    base = eca.DATA_SEED + n_train
    idt = eca.generate_trajectories(rule, n_test, width, steps, 0.5, base)
    lng = eca.generate_trajectories(rule, n_test, width, steps_ood, 0.5, base)
    tok = lambda T: eca.tokenize_windows(eca.create_windows(T, k))
    splits = {'train': tok(tr), 'id_test': tok(idt), 'longer_ood': tok(lng)}
    tgt0 = 1 + k * (width + 1)                       # first target-row cell index
    def scored(s):                                   # score target-row cells only (deterministic)
        return [tgt0 + j - 1 for j in range(width)]
    return Adapter(f'CA_rule{rule}', 'A_ca', eca.CATokenizer.vocab_size, eca.CATokenizer.PAD,
                   splits, scored, oracle_entropy=lambda s: [0.0] * width,
                   note='target-row cells, deterministic (floor 0)')


# ---------------- Dyck (hidden-state) ----------------
def dyck_adapter(bt=2, n_train=4000, n_test=800, small=False):
    if small:
        n_train, n_test = 200, 80
    t = edy.DyckTokenizer(bt)
    g = lambda n, a, b, d, sd: edy.generate_split(t, n, a, b, d, sd)
    splits = {
        'train': g(n_train, 1, 20, 6, edy.DATA_SEED),
        'id_test': g(n_test, 1, 20, 6, edy.DATA_SEED + 1),
        'longer_ood': g(n_test, 30, 40, 14, edy.DATA_SEED + 2),
        'deeper_ood': g(n_test, 1, 20, 14, edy.DATA_SEED + 3),
        'combined_ood': g(n_test, 30, 40, 14, edy.DATA_SEED + 4),
    }
    floor = _hmu('dyck2' if bt == 2 else 'dyck1', 0.5)
    def validator(s):
        return edy.is_valid_dyck(s, t)[0]
    def prompt():
        return [t.BOS]
    return Adapter(f'dyck{bt}', 'B_dyck', t.vocab_size, t.PAD, splits, scored_all,
                   oracle_entropy=lambda s: [floor] * (len(s) - 1),
                   validator=validator, free_run_prompt=prompt, free_run_len=60,
                   note=f'oracle floor = entropy rate {floor:.2f} bits')


# ---------------- Counter ----------------
def counter_adapter(lang='anbn', n_train=4000, n_test=800, small=False):
    if small:
        n_train, n_test = 200, 80
    splits = {
        'train': gc.generate_split(n_train, 1, 20, lang, 0),
        'id_test': gc.generate_split(n_test, 1, 20, lang, 7),
        'longer_ood': gc.generate_split(n_test, 30, 50, lang, 9),
    }
    def scored(s):                                   # only forced positions (floor 0): count competence
        out = []
        for i in range(len(s) - 1):
            if len(gc.oracle_next_legal(s[:i + 1], lang)) == 1:
                out.append(i)
        return out
    return Adapter(f'counter_{lang}', 'B_counter', gc.vocab_size(lang), gc.PAD, splits, scored,
                   oracle_entropy=lambda s: [0.0] * len(scored(s)),
                   validator=lambda s: gc.is_valid(s, lang)[0],
                   free_run_prompt=lambda: [gc.BOS], free_run_len=60,
                   note='forced positions, deterministic (floor 0)')


# ---------------- Recurrence (coverage) ----------------
def fibmod_adapter(p=31, length=25, n_train=300, n_test=100, small=False):
    PAD = p                                          # extra id as pad
    def gen(n, seed0):
        seqs = []
        for i in range(n):
            r = random.Random(seed0 + i)
            a, b = r.randrange(p), r.randrange(p)
            s = [a, b]
            for _ in range(length - 2):
                a, b = b, (a + b) % p; s.append(b)
            seqs.append(s)
        return seqs
    splits = {'train': gen(n_train, 0), 'id_test': gen(n_test, 10**6),
              'novel_seeds_ood': gen(n_test, 2 * 10**6)}
    def scored(s):
        return list(range(1, len(s) - 1))            # predict from position>=2 (deterministic)
    def validator(s):
        return all((s[i] == (s[i-1] + s[i-2]) % p) for i in range(2, len(s)))
    return Adapter(f'fib_mod{p}', 'B_recur', p + 1, PAD, splits, scored,
                   oracle_entropy=lambda s: [0.0] * len(scored(s)),
                   validator=validator,
                   free_run_prompt=lambda: [random.randrange(p), random.randrange(p)],
                   free_run_len=20, note='deterministic given last 2 (floor 0); failure = coverage')


# ---------------- Physics (hidden generator) ----------------
def physics_adapter(G=8, n_frames=8, speed=0.07, n_train=2000, n_test=400, small=False):
    if small:
        n_frames, n_train, n_test = 5, 120, 40
    BOS, EOS = gph.VOCAB, gph.VOCAB + 1
    V = gph.VOCAB + 2
    def render(seed0, n):
        out = []
        for i in range(n):
            cells = gph.simulate(G, n_frames, speed, seed=seed0 + i)
            out.append([BOS] + gph.render_tokens(cells, G) + [EOS])
        return out
    splits = {'train': render(0, n_train), 'id_test': render(10**6, n_test),
              'ood_regime': render(2 * 10**6, n_test)}   # ood handled by speed if desired
    floor = _hmu('physics_v1', 0.55)
    return Adapter('physics', 'C_physics', V, EOS + 1 if False else V, splits, scored_all,
                   oracle_entropy=lambda s: [floor] * (len(s) - 1),
                   validator=lambda s: gph.is_valid_physics(s, G)[0],
                   free_run_prompt=lambda: [BOS], free_run_len=2 * (G * (G + 1) + 1),
                   note=f'rendered grid G={G}; hidden velocity; floor={floor:.2f}')


# ---------------- Language (target) ----------------
def language_adapter(control=None, seq_len=128, small=False):
    path = os.path.join('runs', 'lang', 'corpus.txt')
    b = gl.build(path, seq_len=seq_len, control=control)
    tok = b['tokenizer']
    name = 'language' + ('_' + control if control else '')
    floor = _hmu({'word_shuffle': 'language_word_shuffle', 'ngram': 'language_ngram5'}.get(control, 'language'), 0.77)
    splits = {'train': b['splits']['train']['windows'],
              'id_test': b['splits']['val']['windows'],
              'held_out': b['splits']['test']['windows']}
    if small:
        for k in splits:
            splits[k] = splits[k][:80]
    # acceptability proxy: fraction of generated words that are REAL words (corpus vocab).
    import re
    from collections import Counter
    text = gl.load_corpus(path)
    wc = Counter(re.findall(r"[a-z]+", text.lower()))
    word_set = {w for w, c in wc.items() if c >= 2 and len(w) >= 2}
    def word_rate(s):
        txt = tok.decode(s[10:]).lower()
        ws = [w for w in re.findall(r"[a-z]+", txt) if len(w) >= 2]
        if not ws:
            return 0.0
        return sum(1 for w in ws if w in word_set) / len(ws)
    return Adapter(name, 'D_language', b['vocab_size'], 0, splits, scored_all,
                   oracle_entropy=lambda s: [floor] * (len(s) - 1),
                   validator=lambda s: word_rate(s) > 0.6, soft_score=word_rate,
                   free_run_prompt=lambda: splits['id_test'][0][:10],
                   free_run_len=120,
                   note=f'char-level; floor(hmu)={floor:.2f}; productivity = real-word-rate (soft but grounded)')


def build_all(small=False):
    A = []
    A.append(ca_adapter(90, small=small))
    A.append(dyck_adapter(2, small=small))
    A.append(counter_adapter('anbn', small=small))
    A.append(fibmod_adapter(small=small))
    A.append(physics_adapter(small=small))
    A.append(language_adapter(None, small=small))
    A.append(language_adapter('ngram', small=small))
    return A
