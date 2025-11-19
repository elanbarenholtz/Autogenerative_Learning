"""
Small GPT-style transformer for learning Fibonacci sequences.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional


class NumberTokenizer:
    """
    Simple tokenizer that treats each unique number as a token.
    Includes special tokens for padding.
    """

    def __init__(self, vocabulary: List[int]):
        """
        Args:
            vocabulary: List of all unique numbers in the dataset
        """
        self.vocabulary = sorted(vocabulary)
        self.pad_token = -1
        self.unk_token = -2

        # Create mappings
        self.token_to_id = {num: idx for idx, num in enumerate(self.vocabulary)}
        self.token_to_id[self.pad_token] = len(self.vocabulary)
        self.token_to_id[self.unk_token] = len(self.vocabulary) + 1

        self.id_to_token = {idx: num for num, idx in self.token_to_id.items()}

        self.vocab_size = len(self.token_to_id)
        self.pad_id = self.token_to_id[self.pad_token]
        self.unk_id = self.token_to_id[self.unk_token]

    def encode(self, numbers: List[int]) -> List[int]:
        """Convert list of numbers to token IDs."""
        return [self.token_to_id.get(num, self.unk_id) for num in numbers]

    def decode(self, token_ids: List[int]) -> List[int]:
        """Convert token IDs back to numbers."""
        return [self.id_to_token.get(tid, self.unk_token) for tid in token_ids]

    def encode_batch(self, batch: List[List[int]]) -> torch.Tensor:
        """Encode a batch of sequences with padding."""
        max_len = max(len(seq) for seq in batch)
        encoded = []
        for seq in batch:
            enc = self.encode(seq)
            # Pad to max length
            enc = enc + [self.pad_id] * (max_len - len(enc))
            encoded.append(enc)
        return torch.tensor(encoded, dtype=torch.long)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:x.size(1), :]


class FibonacciTransformer(nn.Module):
    """
    Small GPT-style decoder-only transformer for Fibonacci prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 50
    ):
        """
        Args:
            vocab_size: Size of the number vocabulary
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        """
        Args:
            src: Input tensor of shape (batch_size, seq_len) with token IDs
            src_mask: Optional attention mask

        Returns:
            Output logits of shape (batch_size, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.embedding(src) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # Create causal mask if not provided
        if src_mask is None:
            seq_len = src.size(1)
            src_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)

        # Apply transformer
        x = self.transformer(x, src_mask, is_causal=True)

        # Project to vocabulary
        logits = self.fc_out(x)

        return logits

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive prediction."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def predict_next(self, context: List[int], tokenizer: NumberTokenizer,
                     temperature: float = 1.0) -> int:
        """
        Predict the next number given a context.

        Args:
            context: List of previous numbers
            tokenizer: Tokenizer instance
            temperature: Sampling temperature (use 0 for greedy)

        Returns:
            Predicted next number
        """
        self.eval()
        with torch.no_grad():
            # Encode context
            src = torch.tensor([tokenizer.encode(context)], dtype=torch.long)

            # Get logits for last position
            logits = self.forward(src)
            logits = logits[0, -1, :]  # Last position

            if temperature == 0:
                # Greedy decoding
                predicted_id = logits.argmax().item()
            else:
                # Sample with temperature
                probs = F.softmax(logits / temperature, dim=-1)
                predicted_id = torch.multinomial(probs, 1).item()

            # Decode to number
            predicted_num = tokenizer.id_to_token[predicted_id]

            return predicted_num

    def predict_sequence(self, initial_context: List[int], length: int,
                        tokenizer: NumberTokenizer, temperature: float = 0.0) -> List[int]:
        """
        Generate a sequence by iteratively predicting next tokens.

        Args:
            initial_context: Initial context window
            length: Number of tokens to generate
            tokenizer: Tokenizer instance
            temperature: Sampling temperature

        Returns:
            Generated sequence (including initial context)
        """
        sequence = list(initial_context)

        for _ in range(length):
            # Use last context_window tokens
            context = sequence[-len(initial_context):]
            next_num = self.predict_next(context, tokenizer, temperature)
            sequence.append(next_num)

        return sequence


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    vocab = list(range(100))
    tokenizer = NumberTokenizer(vocab)

    model = FibonacciTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=3
    )

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Test forward pass
    batch_size = 4
    seq_len = 10
    test_input = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))

    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test prediction
    context = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    predicted = model.predict_next(context, tokenizer)
    print(f"\nContext: {context}")
    print(f"Predicted next: {predicted}")
