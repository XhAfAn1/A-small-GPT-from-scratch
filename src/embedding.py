"""
Raw Embedding Layer for GPT.
uses raw 2D weight matrices, direct array slicing, and manual dropout.
"""

import torch
import torch.nn as nn
from src.config import GPTConfig

class GPTEmbedding(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Raw Token Embedding Table (50,257 rows, 384 columns)
        # Random initial numbers with standard deviation 0.02 (GPT-2 standard)
        self.wte = nn.Parameter(torch.randn(config.vocab_size, config.d_model) * 0.02)

        # Raw Position Embedding Table (256 rows, 384 columns)
        self.wpe = nn.Parameter(torch.randn(config.block_size, config.d_model) * 0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = idx.shape

        # Positions are simply: 0, 1, 2, ..., seq_len - 1
        positions = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)

        # Slicing the matrix directly by rows:
        # self.wte[idx] literally pulls out the rows corresponding to each token ID
        tok_emb = self.wte[idx]           # Shape: [Batch, Seq_Len, 384]
        pos_emb = self.wpe[positions]     # Shape: [Seq_Len, 384]

        # Add word vector + position vector
        x = tok_emb + pos_emb             # Shape: [Batch, Seq_Len, 384]

        # Manual Dropout: randomly zero out 10% during training
        if self.training and self.config.dropout > 0.0:
            keep_prob = 1.0 - self.config.dropout
            mask = (torch.rand_like(x) < keep_prob).float()
            x = (x * mask) / keep_prob

        return x
