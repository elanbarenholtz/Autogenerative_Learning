"""
Small decoder-only transformer with RoPE (rotary) positional encoding.
Matched to FibonacciTransformer dims (d_model=128, nhead=4, num_layers=3, ff=512)
so the only meaningful difference vs the absolute-PE model is the position scheme.
RoPE is relative: 'one row back' is a relative offset, so this should generalize
across row widths where the absolute-PE model cannot.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):  # x: (B,H,T,Dh)  cos/sin: (T,Dh)
    return x * cos + rotate_half(x) * sin


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.nhead = nhead
        self.hd = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.nhead, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]               # (B,H,T,hd)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float('-inf'))
        att = self.drop(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, nhead, ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, nhead, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(),
                                 nn.Linear(ff, d_model), nn.Dropout(dropout))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class RoPETransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=512, dropout=0.1, rope_theta=10000.0, **kw):
        super().__init__()
        self.d_model = d_model
        self.hd = d_model // nhead
        self.emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _rope(self, T, device):
        t = torch.arange(T, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))   # (T, hd/2)
        emb = torch.cat((freqs, freqs), dim=-1)            # (T, hd)
        return emb.cos(), emb.sin()

    def forward(self, src, **kw):                          # src: (B,T)
        B, T = src.shape
        x = self.drop(self.emb(src))
        cos, sin = self._rope(T, src.device)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.ln_f(x))


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
