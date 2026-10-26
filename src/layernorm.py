"""
Raw Layer Normalization for GPT.
Direct 1-to-1 implementation of the formula:
y = gamma * (x - mean) / sqrt(var + eps) + beta
"""

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        '''
        initially we set gamma to 1 and beta to 0, so basically for the initial few epochs, 
        the layer norm does nothing more than just standardizing the input. while training 
        the model learns the optimal values for gamma and beta.
        '''
        # gamma (gain / scale), starts at 1.0
        self.gamma = nn.Parameter(torch.ones(d_model))
        # beta (bias / shift), starts at 0.0
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # The equation:
        return self.gamma * (x - mean) / torch.sqrt(var + 1e-5) + self.beta
