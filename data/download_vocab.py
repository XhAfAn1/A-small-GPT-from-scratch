"""
Downloads or exports the official OpenAI GPT-2 BPE vocabulary (gpt2.tiktoken)
into the data/ folder.
"""

import os
import base64
import urllib.request
import tiktoken

VOCAB_PATH = os.path.join(os.path.dirname(__file__), "gpt2.tiktoken")

def download_vocab(target_path: str = VOCAB_PATH):
    """
    Saves the official 50,256 mergeable ranks into a .tiktoken file.
    Format: Each line is <base64_token> <rank_int>
    """
    if os.path.exists(target_path):
        print(f"[Vocab] Found existing vocabulary file at: {target_path}")
        return target_path

    print("[Vocab] Preparing official OpenAI GPT-2 vocabulary...")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Use tiktoken's cached official ranks to serialize to data/gpt2.tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ranks = enc._mergeable_ranks

    print(f"[Vocab] Writing {len(ranks)} merge ranks to: {target_path}")
    with open(target_path, "w", encoding="utf-8") as f:
        for token_bytes, rank in ranks.items():
            b64_str = base64.b64encode(token_bytes).decode("utf-8")
            f.write(f"{b64_str} {rank}\n")

    print(f"[Vocab] Done! File size: {os.path.getsize(target_path) / 1024:.1f} KB")
    return target_path

if __name__ == "__main__":
    download_vocab()
