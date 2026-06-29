"""
Unified learner-battery core: one matched protocol for every generator.

An Adapter exposes a generator as: vocab, splits (train/id_test/OOD...), which
positions to score, an oracle entropy floor (bits) at those positions, and an
optional validator for the productivity assay. The trainer/metrics here are
generator-agnostic so all numbers are comparable.

Metrics:
  - gap-to-oracle = model cross-entropy (bits) - oracle entropy floor (bits), on
    scored positions. Deterministic oracles -> floor 0.
  - n-gram CE baseline (matched order) on the same positions.
  - productivity = novelty-validity rate + coverage from free-run samples.
"""
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class Adapter:
    name: str
    group: str
    vocab_size: int
    pad_id: int
    splits: Dict[str, List[List[int]]]            # 'train','id_test', + OOD splits
    scored_positions: Callable[[List[int]], List[int]]   # seq -> positions i scored for predicting seq[i+1]
    oracle_entropy: Optional[Callable[[List[int]], List[float]]] = None  # bits per scored position
    validator: Optional[Callable[[List[int]], bool]] = None
    free_run_prompt: Optional[Callable[[], List[int]]] = None            # returns a BOS-ish prompt
    free_run_len: int = 64
    soft_score: Optional[Callable[[List[int]], float]] = None            # continuous [0,1] quality (e.g. word-rate)
    note: str = ""


# ---------------------------------------------------------------------------
class SeqDataset(Dataset):
    def __init__(self, seqs): self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return torch.tensor(self.seqs[i], dtype=torch.long)


def collate(batch, pad_id):
    L = max(t.size(0) for t in batch)
    X, Y, M = [], [], []
    for s in batch:
        inp, tgt = s[:-1], s[1:]
        p = L - s.size(0)
        if p > 0:
            inp = torch.cat([inp, torch.full((p,), pad_id, dtype=torch.long)])
            tgt = torch.cat([tgt, torch.full((p,), pad_id, dtype=torch.long)])
        X.append(inp); Y.append(tgt); M.append(inp == pad_id)
    return torch.stack(X), torch.stack(Y), torch.stack(M)


def make_model(vocab, d_model, nhead, layers, ff, dropout, max_len, device, pos='absolute'):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if pos == 'rope':
        from model_rope import RoPETransformer
        return RoPETransformer(vocab, d_model, nhead, layers, ff, dropout).to(device)
    from model import FibonacciTransformer
    return FibonacciTransformer(vocab, d_model, nhead, layers, ff, dropout, max_len).to(device)


def train_lm(adapter, seed=42, epochs=30, batch_size=32, lr=1e-3, d_model=128,
             nhead=4, layers=3, ff=512, dropout=0.1, pos='absolute', device=None, verbose=False):
    device = device or ('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(seed); np.random.seed(seed)
    seqs = adapter.splits['train']
    # size positional table for the LONGEST sequence across all splits (OOD is longer)
    max_len = max(len(s) for sp in adapter.splits.values() for s in sp) + 5
    loader = DataLoader(SeqDataset(seqs), batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, adapter.pad_id))
    model = make_model(adapter.vocab_size, d_model, nhead, layers, ff, dropout, max_len, device, pos)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for X, Y, M in loader:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            opt.zero_grad()
            logits = model(X, src_key_padding_mask=M)
            loss = F.cross_entropy(logits.reshape(-1, adapter.vocab_size), Y.reshape(-1),
                                   ignore_index=adapter.pad_id)
            loss.backward(); opt.step()
        if verbose and (ep + 1) % 10 == 0:
            print(f"      ep{ep+1} loss={loss.item():.4f}")
    model.eval()
    return model, device


@torch.no_grad()
def teacher_forced(model, adapter, split, device, max_eval=400):
    """Mean model CE (bits) and gap-to-oracle on scored positions, plus accuracy."""
    seqs = adapter.splits[split][:max_eval]
    tot_ce = tot_floor = tot_pos = correct = 0.0
    for s in seqs:
        pos = adapter.scored_positions(s)
        if not pos:
            continue
        x = torch.tensor([s], dtype=torch.long, device=device)
        logits = model(x)[0]                                   # (L, V)
        logp = F.log_softmax(logits, dim=-1) / math.log(2)     # bits
        floors = adapter.oracle_entropy(s) if adapter.oracle_entropy else [0.0] * len(pos)
        for i, fl in zip(pos, floors):
            tgt = s[i + 1]
            tot_ce += -logp[i, tgt].item()
            tot_floor += fl
            correct += (logp[i].argmax().item() == tgt)
            tot_pos += 1
    n = max(1, tot_pos)
    ce, floor = tot_ce / n, tot_floor / n
    return {'ce_bits': ce, 'oracle_floor_bits': floor, 'gap_to_oracle_bits': ce - floor,
            'accuracy': correct / n, 'n_positions': int(tot_pos)}


def ngram_ce(adapter, split, n=5, max_eval=400):
    """Matched n-gram CE (bits) with Laplace smoothing on scored positions."""
    from collections import defaultdict
    ctx = defaultdict(lambda: defaultdict(int))
    V = adapter.vocab_size
    for s in adapter.splits['train']:
        for i in adapter.scored_positions(s):
            ctx[tuple(s[max(0, i - n + 1):i + 1])][s[i + 1]] += 1
    tot, npos = 0.0, 0
    for s in adapter.splits[split][:max_eval]:
        for i in adapter.scored_positions(s):
            key = tuple(s[max(0, i - n + 1):i + 1])
            d = ctx.get(key, {})
            tot_c = sum(d.values())
            p = (d.get(s[i + 1], 0) + 1) / (tot_c + V)         # Laplace
            tot += -math.log2(p); npos += 1
    return {'ngram_ce_bits': tot / max(1, npos), 'n': n}


@torch.no_grad()
def free_run_productivity(model, adapter, device, n_samples=200, temperature=1.0, max_eval_train=4000):
    """Sample generations; measure novelty (vs train) + validity (adapter.validator) + coverage."""
    if adapter.validator is None or adapter.free_run_prompt is None:
        return {'available': False}
    train_set = set(tuple(s) for s in adapter.splits['train'][:max_eval_train])
    valid = novel = novel_valid = 0
    soft_total = 0.0
    seen = set()
    for k in range(n_samples):
        toks = list(adapter.free_run_prompt())
        for _ in range(adapter.free_run_len):
            x = torch.tensor([toks], dtype=torch.long, device=device)
            logits = model(x)[0, -1] / temperature
            p = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(p, 1).item()
            toks.append(nxt)
        is_valid = bool(adapter.validator(toks))
        is_novel = tuple(toks) not in train_set
        valid += is_valid; novel += is_novel; novel_valid += (is_valid and is_novel)
        if adapter.soft_score is not None:
            soft_total += float(adapter.soft_score(toks))
        if is_valid:
            seen.add(tuple(toks))
    out = {'available': True, 'validity_rate': valid / n_samples,
           'novelty_rate': novel / n_samples, 'novelty_validity_rate': novel_valid / n_samples,
           'coverage_distinct_valid': len(seen), 'n_samples': n_samples}
    if adapter.soft_score is not None:
        out['mean_soft_score'] = soft_total / n_samples
    return out
