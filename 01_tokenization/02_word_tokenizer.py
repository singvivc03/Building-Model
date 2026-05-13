"""
Module 1.2 — Word-level tokenizer
==================================

Each word becomes one token. We'll deliberately demonstrate the
out-of-vocabulary (OOV) problem that motivates subword tokenizers.

Run:
    python 02_word_tokenizer.py
"""

from typing import Any


import re


class WordTokenizer:
    UNK_TOKEN = "<UNK>"   # used when we see a word we don't know

    def __init__(self, text):
        # Split text into words. Real systems use fancier rules; this is fine for demo.
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        unique_words = sorted(set[Any](words))

        # Reserve token id 0 for <UNK>
        self.word_to_id = {self.UNK_TOKEN: 0}
        for i, w in enumerate(unique_words, start=1):
            self.word_to_id[w] = i
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)

    def encode(self, text):
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        return [self.word_to_id.get(w, 0) for w in words]    # 0 = <UNK>

    def decode(self, ids):
        return " ".join(self.id_to_word[i] for i in ids)


# -------------------------------------------------------------------
# Demo
# -------------------------------------------------------------------
corpus = "the quick brown fox jumps over the lazy dog"
tok = WordTokenizer(corpus)

print(f"Vocab size: {tok.vocab_size}")
print(f"Vocab: {list(tok.word_to_id.keys())}")
print()

# Sentence with words we DID train on
sample1 = "the lazy fox"
print(f"{sample1!r} → {tok.encode(sample1)} → {tok.decode(tok.encode(sample1))!r}")

# Sentence with a NEW word — this is the OOV problem
sample2 = "the lazy unicorn"
ids = tok.encode(sample2)
print(f"{sample2!r} → {ids} → {tok.decode(ids)!r}")
print()
print("Notice 'unicorn' became <UNK>. The model would have no idea")
print("what word the user actually typed. This is unacceptable for a")
print("real LLM, where users type names, code, URLs, emoji, slang,")
print("other languages, typos — none of which fit in any fixed vocabulary.")
print()
print("Solution: BPE. See file 03.")
