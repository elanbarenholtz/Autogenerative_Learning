"""
Exp 1 digit-representation diagnostic.

Quick check: does Fibonacci generalization remain poor once we remove
the unbounded-integer/token-vocab confound by switching to digit tokens?

Runs Fibonacci and Fib+1 with DigitTokenizer, reports number-exact-match
and token accuracy on held-out seeds.
"""
import json
import os
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from recurrence_relations import get_relation
from digit_tokenizer import DigitTokenizer
from model import FibonacciTransformer
from experiment_framework import RANDOM_SEEDS, set_all_seeds

OUTPUT_DIR = 'runs/recurrence/exp1_digit_check'
SEQUENCE_LENGTH = 25
EPOCHS = 50


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DigitSeqDataset(Dataset):
    """Stores (context_token_ids, target_token_ids) pairs."""
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        ctx, tgt = self.examples[idx]
        return torch.tensor(ctx, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def _collate(batch):
    contexts, targets = zip(*batch)
    contexts_padded = pad_sequence(contexts, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return contexts_padded, targets_padded


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def make_examples(relation, seeds, seq_len, ctx_win, tokenizer):
    """
    Build (context_token_ids, target_token_ids) pairs.
    Context = last ctx_win numbers encoded as digits+SEP.
    Target  = next number encoded as digits (no trailing SEP needed for loss).
    """
    examples = []
    for seed in seeds:
        seq = relation.generate_sequence(seed, seq_len)
        for i in range(ctx_win, len(seq)):
            ctx_nums = seq[i - ctx_win:i]
            tgt_num = seq[i]
            ctx_ids = tokenizer.encode_sequence(ctx_nums)
            tgt_ids = tokenizer.encode_sequence([tgt_num])
            examples.append((ctx_ids, tgt_ids))
    return examples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(examples, tokenizer, random_seed, output_dir, verbose=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    dataset = DigitSeqDataset(examples)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=_collate)

    model = FibonacciTransformer(
        vocab_size=tokenizer.vocab_size,  # 12
        d_model=128, nhead=4, num_layers=3, dim_feedforward=512,
        dropout=0.1, max_seq_len=200,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    history = {'losses': [], 'num_accs': [], 'tok_accs': []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct_numbers = 0
        correct_tokens = 0
        total_numbers = 0
        total_tokens = 0

        for contexts, targets in loader:
            contexts, targets = contexts.to(device), targets.to(device)
            bs = contexts.size(0)
            tgt_len = targets.size(1)

            optimizer.zero_grad()
            full_seq = torch.cat([contexts, targets], dim=1)
            logits = model(full_seq)

            ctx_len = contexts.size(1)
            pred_logits = logits[:, ctx_len - 1:ctx_len + tgt_len - 1, :]

            pred_flat = pred_logits.reshape(-1, pred_logits.size(-1))
            tgt_flat = targets.reshape(-1)
            mask = tgt_flat != 0
            if mask.sum() > 0:
                loss = criterion(pred_flat[mask], tgt_flat[mask])
            else:
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                for b in range(bs):
                    pred_digits = []
                    tgt_digits = targets[b][targets[b] != 0].tolist()
                    for j in range(tgt_len):
                        if targets[b, j] == 0:
                            break
                        pid = pred_logits[b, j].argmax().item()
                        pred_digits.append(pid)
                        if pid == targets[b, j].item():
                            correct_tokens += 1
                        total_tokens += 1
                    if pred_digits == tgt_digits:
                        correct_numbers += 1
                    total_numbers += 1

        avg_loss = total_loss / max(len(loader), 1)
        num_acc = 100.0 * correct_numbers / max(total_numbers, 1)
        tok_acc = 100.0 * correct_tokens / max(total_tokens, 1)
        history['losses'].append(avg_loss)
        history['num_accs'].append(num_acc)
        history['tok_accs'].append(tok_acc)

        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': model.state_dict()},
                       os.path.join(output_dir, 'best_model.pt'))

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f}, "
                  f"num_acc={num_acc:.2f}%, tok_acc={tok_acc:.2f}%")

    # Reload best
    ckpt = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, relation, seeds, seq_len, ctx_win, tokenizer, device=None, label='test'):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    correct_numbers = 0
    correct_tokens = 0
    total_numbers = 0
    total_tokens = 0
    qualitative = []  # (seed, pred_nums, true_nums)

    with torch.no_grad():
        for seed in seeds:
            seq = relation.generate_sequence(seed, seq_len)
            pred_nums_for_seed = []
            true_nums_for_seed = []

            for i in range(ctx_win, len(seq)):
                ctx_nums = seq[i - ctx_win:i]
                tgt_num = seq[i]

                ctx_ids = tokenizer.encode_sequence(ctx_nums)
                tgt_ids = tokenizer.encode_sequence([tgt_num])

                full_ids = ctx_ids + tgt_ids
                src = torch.tensor([full_ids], dtype=torch.long, device=device)
                logits = model(src)

                ctx_len = len(ctx_ids)
                pred_logits = logits[0, ctx_len - 1:ctx_len + len(tgt_ids) - 1, :]

                # Token-level
                pred_tok_ids = []
                for j in range(len(tgt_ids)):
                    pid = pred_logits[j].argmax().item()
                    pred_tok_ids.append(pid)
                    if pid == tgt_ids[j]:
                        correct_tokens += 1
                    total_tokens += 1

                # Number-level
                if pred_tok_ids == tgt_ids:
                    correct_numbers += 1
                total_numbers += 1

                # Decode predicted number
                pred_num = tokenizer.decode_sequence(pred_tok_ids)
                pred_num = pred_num[0] if pred_num else -1
                pred_nums_for_seed.append(pred_num)
                true_nums_for_seed.append(tgt_num)

            qualitative.append((seed, pred_nums_for_seed, true_nums_for_seed))

    num_acc = 100.0 * correct_numbers / max(total_numbers, 1)
    tok_acc = 100.0 * correct_tokens / max(total_tokens, 1)

    # UNK check: DigitTokenizer has no UNK by construction (only digits + SEP + PAD)
    unk_count = 0  # always 0

    return {
        'number_exact_match': num_acc,
        'token_accuracy': tok_acc,
        'total_numbers': total_numbers,
        'total_tokens': total_tokens,
        'unk_count': unk_count,
        'qualitative': qualitative,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_check(relations=None, random_seeds=None):
    if relations is None:
        relations = ['fibonacci', 'fibonacci_plus_constant']
    if random_seeds is None:
        random_seeds = RANDOM_SEEDS

    tokenizer = DigitTokenizer()
    print(f"Vocab: {tokenizer.vocab} (size={tokenizer.vocab_size})")
    print(f"UNK token: NONE (by construction)")
    print()

    all_results = []

    for rel_name in relations:
        relation = get_relation(rel_name)
        ctx_win = relation.order + 8
        train_seeds = relation.get_training_seeds()
        test_seeds = relation.get_test_seeds()

        print(f"{'='*60}")
        print(f"  {rel_name}  (ctx_win={ctx_win}, seq_len={SEQUENCE_LENGTH})")
        print(f"  train_seeds={len(train_seeds)}, test_seeds={len(test_seeds)}")
        print(f"{'='*60}")

        train_examples = make_examples(relation, train_seeds, SEQUENCE_LENGTH, ctx_win, tokenizer)
        print(f"  Training examples: {len(train_examples)}")

        for rs in random_seeds:
            print(f"\n  --- RS={rs} ---")
            run_dir = os.path.join(OUTPUT_DIR, f'{rel_name}_RS{rs}')
            model, history = train_model(train_examples, tokenizer, rs, run_dir)

            # Evaluate on train seeds
            train_eval = evaluate(model, relation, train_seeds, SEQUENCE_LENGTH,
                                  ctx_win, tokenizer, label='train')
            # Evaluate on test seeds
            test_eval = evaluate(model, relation, test_seeds, SEQUENCE_LENGTH,
                                 ctx_win, tokenizer, label='test')

            print(f"\n  TRAIN: num_exact={train_eval['number_exact_match']:.2f}%, "
                  f"tok_acc={train_eval['token_accuracy']:.2f}%, "
                  f"UNK={train_eval['unk_count']}")
            print(f"  TEST:  num_exact={test_eval['number_exact_match']:.2f}%, "
                  f"tok_acc={test_eval['token_accuracy']:.2f}%, "
                  f"UNK={test_eval['unk_count']}")

            # Qualitative: first 3 test seeds, first 5 predictions
            print(f"\n  Qualitative (first 3 test seeds, first 5 predictions):")
            for seed, preds, trues in test_eval['qualitative'][:3]:
                print(f"    seed={seed}")
                for j in range(min(5, len(preds))):
                    match = 'OK' if preds[j] == trues[j] else 'XX'
                    print(f"      pos {ctx_win+j}: pred={preds[j]:>8}, true={trues[j]:>8}  [{match}]")

            result = {
                'relation': rel_name,
                'random_seed': rs,
                'train_number_exact': train_eval['number_exact_match'],
                'train_token_acc': train_eval['token_accuracy'],
                'test_number_exact': test_eval['number_exact_match'],
                'test_token_acc': test_eval['token_accuracy'],
                'unk_count_train': train_eval['unk_count'],
                'unk_count_test': test_eval['unk_count'],
                'final_train_loss': history['losses'][-1],
            }
            all_results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Relation':<25} {'RS':<6} {'Train Num%':<12} {'Test Num%':<12} "
          f"{'Train Tok%':<12} {'Test Tok%':<12}")
    print(f"{'-'*70}")
    for r in all_results:
        print(f"{r['relation']:<25} {r['random_seed']:<6} "
              f"{r['train_number_exact']:>8.2f}%   {r['test_number_exact']:>8.2f}%   "
              f"{r['train_token_acc']:>8.2f}%   {r['test_token_acc']:>8.2f}%")

    # Aggregated
    print(f"\n  Aggregated (mean ± std across RS):")
    for rel_name in relations:
        rel_results = [r for r in all_results if r['relation'] == rel_name]
        train_accs = [r['train_number_exact'] for r in rel_results]
        test_accs = [r['test_number_exact'] for r in rel_results]
        print(f"  {rel_name}: train={np.mean(train_accs):.2f}±{np.std(train_accs):.2f}%, "
              f"test={np.mean(test_accs):.2f}±{np.std(test_accs):.2f}%")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}/summary.json")
    return all_results


if __name__ == '__main__':
    run_check()
