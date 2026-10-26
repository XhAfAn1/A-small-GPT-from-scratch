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

---

### 2. Embedding Layer (Token & Position Embeddings)

#### What it does:
So embedding is basically simply creating a higher dimensional tensor and assigning them to each token. so it means we have 50k tokens in vocabulary and each token has now single value (an integer ID). but instead of that single value now we give so many values (in our case 384 values) for each token. because a single number cannot describe the meaning of a word to a computer and larger numbers don't mean "greater" words. initially it's just random values (384 values for each token) and when we train the model on collection of sentences the model itself will adjust and give meaningful values to each token based on how words are used together. and position embedding is needed because Transformers look at all words in a sentence at the exact same time in parallel, so they don't naturally know what word comes first or last. so when we give sentences for training it also checks what is the position of the token (position 0, position 1, position 2...) and according to that it learns the position embeddings. and finally we simply add the token meaning numbers and position numbers together so each token knows what it is and where it is in the sentence.
 
#### How it works:
1. **Raw Token Embedding Table (`self.wte`):**
   A raw 2D spreadsheet matrix of 50,257 rows and 384 columns. When token IDs come in (for example `[20037, 4124, 9246]`), it slices the matrix directly by rows (`self.wte[idx]`) to pull out the 384 numbers for each token.
2. **Raw Position Embedding Table (`self.wpe`):**
   A raw 2D matrix of 256 rows and 384 columns. It generates position numbers (`[0, 1, 2...]`) for the sentence and slices the matrix (`self.wpe[positions]`) to pull out the 384 numbers for each position slot. Because without this the model will not be able to know the difference between "Cat fired the missile" and "Missile fired the cat" because both sentences have same tokens but in different order and without position embedding the model will not be able to know the difference between them.  
3. **Combining them (`tok_emb + pos_emb`):**
   Simply adds the token meaning vector and position vector element-by-element so the output tensor has shape `[Batch_Size, Sequence_Length, 384]`.
4. **Manual Dropout:**
   During training, it randomly turns 10% of the numbers into zero so the model doesn't over-rely on any single number. this is used to prevent overfitting.

---

### 3. Layer Normalization (Pre-LN)

#### What it does:
In simple words, Layer Normalization acts like a volume equalizer or stabilizer for numbers. When tokens (each having 384 numbers) pass through multiple layers of matrix additions and multiplications, numbers can easily become too wild, either exploding into huge numbers (thousands) or shrinking down into tiny decimals near zero. If numbers explode, training crashes instantly with `NaN` (Not a Number) loss. If they shrink too much, gradients vanish and the model stops learning completely. Layer Normalization prevents this: for each individual token, it calculates the average (mean) and spread (variance), and rescales those 384 numbers so their average becomes 0 and their spread becomes 1. This keeps the numbers in a healthy, stable range across all 6 Transformer blocks without blowing up.

#### How it works:
1. **The Math Equation:**
   It directly implements the formula: `y = gamma * ((x - mean) / sqrt(var + eps)) + beta`
   - `mean`: The average of the 384 numbers for that token.
   - `var`: The variance (how widely the numbers are spread out from the average).
   - `eps` (`1e-5`): A tiny safety number added to the bottom so we never accidentally divide by zero.
2. **Standardization `(x - mean) / sqrt(...)`:**
   Subtracting the mean centers all numbers around 0. Dividing by the standard deviation squashes their spread to 1.
3. **Learnable Scale & Shift (`gamma` and `beta`):**
   Initially, `gamma` starts at 1.0 and `beta` starts at 0.0. In the beginning, LayerNorm does pure standardization. But as training goes on, the model learns the optimal values for `gamma` and `beta`, allowing it to stretch or shift values if it discovers that some features need to be stronger or quieter.

