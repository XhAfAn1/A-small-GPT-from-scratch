"""
Pure Python Byte Pair Encoding (BPE) Tokenizer for GPT-2.
Simple, raw, and easy to understand.
"""

import os
import base64
import regex

# Regex to split words, contractions, numbers, and punctuation
# so this part basically separate the words and other things like space coma question mark etc before tokenization for better token efficiency
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

DEFAULT_VOCAB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gpt2.tiktoken"
)

class GPT2Tokenizer:
    def __init__(self, vocab_path: str = DEFAULT_VOCAB_PATH):
        self.compiled_pattern = regex.compile(GPT2_SPLIT_PATTERN)
        self.mergeable_ranks = {}
        self.inverse_ranks = {}

        # 1. Read the 50,256 merge ranks from gpt2.tiktoken
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                b64_token, rank = line.strip().split()
                token_bytes = base64.b64decode(b64_token)
                rank = int(rank)
                self.mergeable_ranks[token_bytes] = rank
                self.inverse_ranks[rank] = token_bytes

        # 2. Add the single special token
        self.mergeable_ranks[b"<|endoftext|>"] = 50256
        self.inverse_ranks[50256] = b"<|endoftext|>"
        self.vocab_size = 50257
 

    """
    this is the main engine that basically tokenize the text using the vocab stored. basically it take 1 by 1 character of a word and then check if the 2 characters are present in the vocab. while checking pairs it always chooses the pair with the lowest rank (number). 
    if present then merge them and check again and again until no more merge is possible. and finally it returns the list of token ids.
    """

    def _bpe(self, word_bytes: bytes) -> list[int]:
        # Start with individual single bytes: [b'd', b'o', b'g']
        parts = [bytes([b]) for b in word_bytes]

        while len(parts) >= 2:
            # Look at all adjacent pairs and find the one with the lowest rank
            min_rank = float("inf")
            min_idx = -1

            for i in range(len(parts) - 1):
                pair = parts[i] + parts[i + 1]
                rank = self.mergeable_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_idx = i

            # If no pair can be merged, we are done
            if min_rank == float("inf"):
                break

            # Merge the winning pair
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]

        # Return the list of token IDs
        return [self.mergeable_ranks[p] for p in parts]

    """
    to encode the sentence it will first split chunks of text based on <|endoftext|> token
    then it will split the chunks into words based on the regex pattern
    then it will call _bpe function for each word
    then it will return the list of token ids
    """
    def encode(self, text: str) -> list[int]:
        tokens = []

        # If <|endoftext|> is in the text, split by it simply
        subtexts = text.split("<|endoftext|>")

        for i, subtext in enumerate(subtexts):
            # Split normal text into words using the regex
            words = regex.findall(self.compiled_pattern, subtext)
            for word in words:
                tokens.extend(self._bpe(word.encode("utf-8")))

            # If there was a <|endoftext|> between subtexts, add its ID (50256)
            if i < len(subtexts) - 1:
                tokens.append(50256)

        return tokens

    """
    just a decoder takes ids and find and merge the words
    """
    def decode(self, token_ids: list[int]) -> str:
        byte_list = [self.inverse_ranks[tid] for tid in token_ids]
        return b"".join(byte_list).decode("utf-8", errors="replace")

    def tokenize_with_details(self, text: str) -> list[tuple[int, str]]:
        token_ids = self.encode(text)
        return [(tid, self.inverse_ranks[tid].decode("utf-8", errors="replace")) for tid in token_ids]
