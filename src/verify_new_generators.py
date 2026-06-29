"""Verification gate for the new generators (B2 counters, B4 position, C physics, D language)."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_counter as GC
import gen_position as GP
import gen_physics as GPHYS
import gen_language as GL

FAILS = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)

print("=" * 64)
print("  NEW GENERATOR VERIFICATION")
print("=" * 64)

# ---- B2 counters ----
print("\n[B2] Counter languages")
for lang in ['anbn', 'anbncn']:
    seqs = GC.generate_split(300, 1, 20, lang, seed=0)
    check(f"{lang}: all generated valid", all(GC.is_valid(s, lang)[0] for s in seqs))
    # oracle consistency: every actual next token is in the legal set
    ok = True
    for s in seqs[:50]:
        for i in range(1, len(s)):
            if s[i] not in GC.oracle_next_legal(s[:i], lang):
                ok = False; break
    check(f"{lang}: oracle permits every true continuation", ok)
    # corruption rejected: drop one trailing b -> unequal counts
    bad = list(seqs[10]); bad.remove(GC.B)
    check(f"{lang}: count-corruption rejected", not GC.is_valid(bad, lang)[0])
    toks = set(t for s in seqs for t in s)
    check(f"{lang}: no OOV", toks.issubset(set(range(GC.vocab_size(lang)))))
# hand check: 'BOS a a b' must require exactly one more b
check("anbn oracle forces remaining b", GC.oracle_next_legal([GC.BOS, GC.A, GC.A, GC.B], 'anbn') == {GC.B})

# ---- B4 position ----
print("\n[B4] Position / indexing tasks")
for task in ['kth', 'middle', 'last2']:
    exs = GP.generate_split(task, 400, 4, 24, seed=1)
    ok = all([e['tokens'][p] for p in e['answer_positions']] == e['answer'] for e in exs)
    check(f"{task}: answer sits at its marked positions", ok)
    check(f"{task}: oracle == embedded answer", all(GP.oracle_answer(e) == e['answer'] for e in exs))
    toks = set(t for e in exs for t in e['tokens'])
    check(f"{task}: no OOV", toks.issubset(set(range(GP.vocab_size()))))

# ---- C physics ----
print("\n[C] Physics-trace")
G = 12
cells = GPHYS.simulate(G, 40, speed=0.07, seed=3)
toks = GPHYS.render_tokens(cells, G)
dec = GPHYS.decode_frames(toks, G)
check("render->decode round-trips ball cells", dec == cells, f"{len(cells)} frames")
check("no OOV", set(toks).issubset(set(range(GPHYS.VOCAB))))
ok_real, d_real = GPHYS.is_valid_physics(toks, G)
check("validator ACCEPTS real trace", ok_real, d_real)
# corrupt: random ball positions -> should be rejected
rng = random.Random(0)
rand_cells = [(rng.randrange(G), rng.randrange(G)) for _ in range(40)]
ok_rand, d_rand = GPHYS.is_valid_physics(GPHYS.render_tokens(rand_cells, G), G)
check("validator REJECTS random trace", not ok_rand, d_rand)

# ---- D language ----
print("\n[D] Language pipeline")
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs/lang/placeholder.txt')
b = GL.build(path, seq_len=64)
check("tokenizer vocab built", b['vocab_size'] > 10, f"vocab={b['vocab_size']}")
tok = b['tokenizer']
check("encode/decode round-trip", tok.decode(tok.encode("the river")) == "the river")
check("train windows non-empty", len(b['splits']['train']['windows']) > 0,
      f"{len(b['splits']['train']['windows'])} windows")
bs = GL.build(path, seq_len=64, control='word_shuffle')
check("word-shuffle control builds, same vocab", bs['vocab_size'] == b['vocab_size'])
bn = GL.build(path, seq_len=64, control='ngram', n_ngram=4)
check("ngram-resample control builds", len(bn['splits']['train']['windows']) > 0)

print("\n" + "=" * 64)
print(f"  RESULT: {'ALL PASSED' if not FAILS else str(len(FAILS)) + ' FAILED -> ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
