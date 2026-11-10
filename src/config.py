from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 256     # Maximum context window 
    vocab_size: int = 50257   # Vocabulary size 
    n_layer: int = 6          # Number of Transformer blocks 
    n_head: int = 6           # Number of parallel Attention heads
    d_model: int = 384        # Embedding dimension 
    mlp_ratio: int = 4        # Expansion factor for Feed-Forward network (384 -> 1536 -> 384)
    dropout: float = 0.1      # Dropout probability for regularization
    bias: bool = True         # Whether to include additive bias in Linear and LayerNorm
