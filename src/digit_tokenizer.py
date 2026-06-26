"""
Digit-wise tokenizer for Experiment 2A.

Instead of treating each number as a single token (e.g., 144 → TOKEN_144),
we break it into individual digits (e.g., 144 → [1, 4, 4]).

This changes the task from predicting a number token to predicting
a sequence of digit tokens.
"""
import torch
from typing import List

class DigitTokenizer:
    """Tokenizes numbers as sequences of digits with separators."""

    def __init__(self):
        # Special tokens
        self.pad_token = '[PAD]'
        self.sep_token = '[SEP]'

        # Digit tokens
        self.digit_tokens = [f'[DIGIT_{i}]' for i in range(10)]

        # Build vocabulary
        self.vocab = [self.pad_token, self.sep_token] + self.digit_tokens
        self.vocab_size = len(self.vocab)

        # Token to ID mapping
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        # Special IDs
        self.pad_id = self.token_to_id[self.pad_token]
        self.sep_id = self.token_to_id[self.sep_token]

        print(f"Digit tokenizer initialized:")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Tokens: {self.vocab}")

    def number_to_tokens(self, number: int) -> List[str]:
        """Convert a number to digit tokens."""
        digits = [int(d) for d in str(abs(number))]
        tokens = [f'[DIGIT_{d}]' for d in digits]
        return tokens

    def tokens_to_number(self, tokens: List[str]) -> int:
        """Convert digit tokens back to a number."""
        digits = []
        for token in tokens:
            if token.startswith('[DIGIT_'):
                digit = int(token.split('_')[1].rstrip(']'))
                digits.append(str(digit))
        if not digits:
            return 0
        return int(''.join(digits))

    def encode_sequence(self, numbers: List[int]) -> List[int]:
        """
        Encode a sequence of numbers into token IDs.

        Example: [1, 2, 13] → [DIGIT_1, SEP, DIGIT_2, SEP, DIGIT_1, DIGIT_3]
        """
        token_ids = []
        for i, number in enumerate(numbers):
            # Add digit tokens for this number
            digit_tokens = self.number_to_tokens(number)
            token_ids.extend([self.token_to_id[t] for t in digit_tokens])

            # Add separator (except after last number)
            if i < len(numbers) - 1:
                token_ids.append(self.sep_id)

        return token_ids

    def decode_sequence(self, token_ids: List[int]) -> List[int]:
        """
        Decode token IDs back into numbers.

        Split on SEP tokens and convert each group of digits to a number.
        """
        tokens = [self.id_to_token[tid] for tid in token_ids if tid != self.pad_id]

        numbers = []
        current_digits = []

        for token in tokens:
            if token == self.sep_token:
                if current_digits:
                    numbers.append(self.tokens_to_number(current_digits))
                    current_digits = []
            elif token.startswith('[DIGIT_'):
                current_digits.append(token)

        # Don't forget the last number
        if current_digits:
            numbers.append(self.tokens_to_number(current_digits))

        return numbers

    def encode_for_training(self, context: List[int], target: int) -> tuple:
        """
        Encode a (context, target) pair for training.

        Returns:
            context_ids: Token IDs for context
            target_ids: Token IDs for target number
        """
        context_ids = self.encode_sequence(context)
        target_ids = self.encode_sequence([target])  # Returns list of token IDs for this number

        return context_ids, target_ids

    def predict_number_from_logits(self, logits_sequence, max_digits=6):
        """
        Given a sequence of logits, predict the next number.

        Args:
            logits_sequence: [seq_len, vocab_size] logits from model
            max_digits: Maximum number of digits to generate

        Returns:
            predicted_number: Integer
            num_tokens_used: How many tokens were used
        """
        predicted_digits = []

        for i in range(max_digits):
            if i >= len(logits_sequence):
                break

            # Get prediction for this position
            logits = logits_sequence[i]
            predicted_id = torch.argmax(logits).item()
            predicted_token = self.id_to_token[predicted_id]

            # Stop if we hit SEP or PAD
            if predicted_token in [self.sep_token, self.pad_token]:
                break

            # If it's a digit, add it
            if predicted_token.startswith('[DIGIT_'):
                predicted_digits.append(predicted_token)
            else:
                break

        if not predicted_digits:
            return 0, 0

        predicted_number = self.tokens_to_number(predicted_digits)
        return predicted_number, len(predicted_digits)


def test_tokenizer():
    """Test the digit tokenizer."""
    print("="*60)
    print("Testing Digit Tokenizer")
    print("="*60)

    tokenizer = DigitTokenizer()

    # Test 1: Encode/decode single numbers
    print("\nTest 1: Single numbers")
    test_numbers = [0, 1, 5, 13, 144, 987]
    for num in test_numbers:
        tokens = tokenizer.number_to_tokens(num)
        decoded = tokenizer.tokens_to_number(tokens)
        print(f"  {num} → {tokens} → {decoded}")
        assert decoded == num, f"Mismatch: {num} != {decoded}"

    # Test 2: Encode/decode sequences
    print("\nTest 2: Sequences")
    test_seq = [1, 1, 2, 3, 5, 8, 13, 21]
    encoded = tokenizer.encode_sequence(test_seq)
    decoded = tokenizer.decode_sequence(encoded)
    print(f"  Input:   {test_seq}")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: {decoded}")
    assert decoded == test_seq, f"Mismatch: {test_seq} != {decoded}"

    # Test 3: Training pairs
    print("\nTest 3: Training pairs")
    context = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    target = 89
    context_ids, target_ids = tokenizer.encode_for_training(context, target)
    print(f"  Context: {context}")
    print(f"  Target: {target}")
    print(f"  Context IDs: {context_ids} (length={len(context_ids)})")
    print(f"  Target IDs: {target_ids} (length={len(target_ids)})")

    print("\n✓ All tests passed!")
    print(f"\nKey insight: Context of 10 numbers → ~{len(context_ids)} tokens")
    print(f"             Much longer than number-based encoding!")


if __name__ == "__main__":
    test_tokenizer()
