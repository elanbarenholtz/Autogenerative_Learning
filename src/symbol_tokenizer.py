"""
Clean symbol tokenizer for Experiment 3C.

Maps 0..M-1 to token IDs 0..M-1, plus PAD (M) and UNK (M+1).
Interface-compatible with NumberTokenizer from model.py.
"""
from typing import List


class SymbolTokenizer:
    """
    One-to-one symbol tokenizer for bounded integer domains.

    Each value v in [0, modulus) gets its own token ID = v.
    PAD = modulus, UNK = modulus + 1.
    """

    def __init__(self, modulus: int):
        self.modulus = modulus
        self.vocabulary = list(range(modulus))

        # Token-to-ID: identity mapping for 0..M-1
        self.token_to_id = {v: v for v in range(modulus)}
        self.pad_token = -1
        self.unk_token = -2
        self.token_to_id[self.pad_token] = modulus
        self.token_to_id[self.unk_token] = modulus + 1

        # ID-to-token: reverse mapping
        self.id_to_token = {idx: num for num, idx in self.token_to_id.items()}

        self.vocab_size = modulus + 2
        self.pad_id = modulus
        self.unk_id = modulus + 1

    def encode(self, numbers: List[int]) -> List[int]:
        """Convert list of numbers to token IDs."""
        return [self.token_to_id.get(num, self.unk_id) for num in numbers]

    def decode(self, token_ids: List[int]) -> List[int]:
        """Convert token IDs back to numbers."""
        return [self.id_to_token.get(tid, self.unk_token) for tid in token_ids]
