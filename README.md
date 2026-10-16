# A small GPT from scratch

This repo documents my code and learning journey on how a generative AI works at its foundational level. here my goal is to create a small generative transformer model from scratch and train it on a dataset of small stories and check how good is it in english writing. 

So my plan is to create one component at a time after learning about it deeply, implementing it step by step, and write some short summary about the component and what it does in simple words.


---

## Targeted Architecture

The model is a **decoder-only autoregressive Transformer** similar to GPT-2, we will follow this architecture as baseline.

### Architecture Diagram

<p align="center">
  <img src="assets/selected%20architecture.png" alt="Targeted Architecture Diagram" width="600" />
</p>

---

## Target Hyperparameters

These hyperparameters are specified in [`src/config.py`]:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **`vocab_size`** | `50,257` | vocabulary size as we may try `tiktoken` |
| **`block_size`** | `256` | maximum context length |
| **`d_model`** | `384` | hidden representation dimension for each token |
| **`n_layer`** | `6` | number of transformer blocks |
| **`n_head`** | `6` | number of causal attention heads (64 per head (384/6)) |
| **`mlp_ratio`** | `4` | expansion factor for Feed-Forward network  |
| **`dropout`** | `0.1` | regularization dropout rate |
| **`bias`** | `True` | include learnable additive bias terms in Linear and LayerNorm layers |
| **`weight_tying`** | `True` | LM head projects using the transpose of the input embedding weights |

---

## Parameter Breakdown

With $d = 384$, $V = 50,257$, $T = 256$, and $L = 6$:

* **Input Embeddings:**
  * Token Embedding: $50,257 \times 384 = 19,298,688 \approx 19.30\text{M}$
  * Position Embedding: $256 \times 384 = 98,304 \approx 0.10\text{M}$
  * Subtotal: $\approx 19.40\text{M}$
* **Transformer Layers ($6\times$ Blocks):**
  * Multi-Head Attention: $6 \times \approx 0.59\text{M} \approx 3.55\text{M}$
  * Feed-Forward Network: $6 \times \approx 1.18\text{M} \approx 7.09\text{M}$
  * LayerNorms: $6 \times \approx 0.002\text{M} \approx 0.01\text{M}$
  * Subtotal: $\approx 10.65\text{M}$
* **Final LayerNorm:** $2 \times 384 = 768$
* **Output LM Head:** Tied with input embedding ($0$ additional weights)
* **Total Parameters:** **$\approx 30.04\text{M}$ parameters**
