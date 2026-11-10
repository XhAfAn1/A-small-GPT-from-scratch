"""
Single Transformer Block (The N=6 repeating unit in assets/selected architecture.png)

Flow inside 1 block:
1. Residual Split 1 -> LayerNorm -> Causal Self-Attention -> Residual Add (+)
2. Residual Split 2 -> LayerNorm -> Feed Forward MLP      -> Residual Add (+)
"""

import torch
import torch.nn as nn

from src.config import GPTConfig
from src.layernorm import LayerNorm
from src.attention import CausalSelfAttention
from src.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. LayerNorm -> Causal Attention + Linear Projection -> Residual (+)
        x = x + self.attn(self.ln_1(x))

        # 2. LayerNorm -> Feed Forward MLP -> Residual (+)
        x = x + self.mlp(self.ln_2(x))
        return x
