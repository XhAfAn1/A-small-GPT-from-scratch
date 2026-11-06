"""
Raw Causal Multi-Head Self-Attention for GPT.
Follows the fundamental attention equation:
Attention(Q, K, V) = Softmax( (Q @ K^T) / sqrt(d_k) + Mask ) @ V
"""

import math
import torch
import torch.nn as nn
from src.config import GPTConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # d_model must be divisible by n_head means equal size per head
        assert config.d_model % config.n_head == 0 

        self.config = config
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_model // config.n_head  

        # 1. Learnable projections to get Query, Key, and Value from embeddings
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # 2. Final output projection after combining heads
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # 3. Regularization dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch size, Sequence length, Embedding dimension (384)

        # Step 1: Calculate Query, Key, and Value vectors for all tokens
        q = self.q_proj(x)  # Shape: [B, T, 384]
        k = self.k_proj(x)  
        v = self.v_proj(x)  

        # Step 2: Split 384 into 6 heads 
        # Reshape to [B, n_head, T, d_head] so PyTorch computes all 6 heads in parallel
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # [B, 6, T, 64]
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)  
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)  

        # Step 3: Match Query with Key (Dot product) and scale by sqrt(d_head)
        # scores[i, j] represents how much token i wants to look at token j
        # this score tells us how much the query and key vectors want to align or similar they are
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, 6, T, T]

        # Step 4: Causal Mask (lower-triangular matrix)
        # We dont want the future tokens to have any information about the past tokens
        # Set future token scores to -infinity so they receive 0% attention
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 5: Softmax turns scores into percentages (each row sums to 1.0)
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Step 6: Blend Values (weighted sum of V based on attention percentages)
        out = weights @ v  # [B, 6, T, 64]

        # Step 7: Glue the 6 heads back together into one 384-dimensional vector
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, 384]

        # Step 8: Final linear projection + residual dropout
        out = self.resid_dropout(self.out_proj(out))
        return out
