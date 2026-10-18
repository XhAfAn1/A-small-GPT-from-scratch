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

With d_model = 384, vocab_size = 50,257, context = 256, and 6 layers:

* **Input Embeddings:**
  * Token Embedding: 50,257 * 384 = 19.30M
  * Position Embedding: 256 * 384 = 0.10M
  * Subtotal: ~19.40M
* **Transformer Layers (6 blocks):**
  * Multi-Head Attention: 6 * ~0.59M = ~3.55M
  * Feed-Forward Network: 6 * ~1.18M = ~7.09M
  * LayerNorms: 6 * ~0.002M = ~0.01M
  * Subtotal: ~10.65M
* **Final LayerNorm:** 2 * 384 = 768 parameters
* **Output LM Head:** Tied with input embedding (0 extra weights)
* **Total Parameters:** **~30.04M parameters**

---

## Components & Learning Notes

### 1. Byte Pair Encoding (BPE) Tokenizer

#### What it does:
Computers cannot understand words or letters directly. The tokenizer is the translator that turns human sentences into a list of integer IDs (for example, `"Once upon a time"` becomes `[7454, 2402, 257, 640]`). Though i have writen short note on the code file. so basically a tokenizer is just a function that map words or subwords to integer IDs and vice versa. so when we give it a string of text, it returns a list of integers and when we give it a list of integers, it returns a string of text. becuase giving raw text to the model or giving the numaric representations of words wont be efficient becuase model need to relate the letters and punctuation with each other and make some sense out of it for a language model to work better.  

#### How it works:
1. **Regex Word Splitting (`GPT2_SPLIT_PATTERN`):**
   Before tokenizing, it separates words from spaces, commas, question marks, numbers, and contractions (like `'s`, `'t`). This keeps words and punctuation clean so they don't glue together into messy tokens like `"dog."` or `"cat,"`.
2. **The BPE Engine (`_bpe`):**
   The main engine takes 1-by-1 characters of a word and checks if pairs are present in the vocabulary. While checking pairs, it always chooses the pair with the **lowest rank (priority number)**. It merges them and repeats the check until no more merges are possible, finally returning the token IDs. if we take word "cat" as an example, suppose it has never seen the word "cat" and the pairs it has seen are ["c", "a"], ["a", "t"], ["c", "a", "t"]. and let's assume that the pair ["c", "a"] has the lowest rank. so it will merge "c" and "a" and return ["ca", "t"]. and if the pair ["ca", "t"] has the lowest rank then it will merge "ca" and "t" and return ["cat"]. so in this way we can merge the pairs of tokens and make them in single token.   
3. **The Encoder (`encode`):**
   First splits text around the special `<|endoftext|>` token, splits normal text into words using the regex pattern, calls the `_bpe` function on each word, and returns the full list of token IDs.
4. **The Decoder (`decode`):**
   Takes the integer IDs, looks up their raw byte pieces in the reverse dictionary, glues them together, and reconstructs the original English text.
