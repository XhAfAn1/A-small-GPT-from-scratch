"""
Raw Feed-Forward Network (MLP) for GPT.
Expands d_model (384) by 4x to 1,536, applies raw GELU non-linearity,
and compresses back to 384.
"""

import math
import torch
import torch.nn as nn
from src.config import GPTConfig

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.d_model * config.mlp_ratio  # 384 * 4 = 1536

        # 1. Expand: 384 -> 1536
        self.fc1 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)

        # 2. Compress back: 1536 -> 384
        self.fc2 = nn.Linear(hidden_dim, config.d_model, bias=config.bias)

        # 3. Regularization dropout
        self.dropout = nn.Dropout(config.dropout)

    def gelu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raw GPT-2 GELU equation:
        0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        """
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #Expand onto 4x wider workbench [B, T, 384] -> [B, T, 1536]
        x = self.fc1(x)

        #Apply GELU activation 
        x = self.gelu(x)

        #down to width [B, T, 1536] -> [B, T, 384]
        x = self.fc2(x)

        #Dropout
        return self.dropout(x)

