"""
Module 1.1 — Character-level tokenizer
=======================================

The simplest tokenizer possible. Each character is one token.
Run:
    python 01_char_tokenizer.py
"""


class CharTokenizer:
    def __init__(self, text):
        # Build the vocabulary: every unique character that appears in `text`.
        unique_chars = sorted(set(text))

        # Two lookup tables: character → integer, and back.
        self.char_to_id = {ch: i for i, ch in enumerate(unique_chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)

    def encode(self, text):
        """text → list of integers"""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        """list of integers → text"""
        return "".join(self.id_to_char[i] for i in ids)


# -------------------------------------------------------------------
# Demo
# -------------------------------------------------------------------
corpus = "hello world! the quick brown fox jumps over the lazy dog."
tok = CharTokenizer(corpus)

print(f"Vocabulary size: {tok.vocab_size}")
print(f"Vocabulary: {list(tok.char_to_id.keys())}")
print()

sample = "hello dog"
ids = tok.encode(sample)
back = tok.decode(ids)

print(f"original : {sample!r}")
print(f"encoded  : {ids}")
print(f"decoded  : {back!r}")
print()
print(f"Notice: the sentence {sample!r} is {len(sample)} characters,")
print(f"so it's also {len(ids)} tokens. Character tokenizers make LONG sequences.")
print()

# Try to encode a character we didn't train on — this will crash.
print("Trying to encode 'Z' (uppercase, not in our corpus)...")
try:
    tok.encode("Z")
except KeyError as e:
    print(f"  → KeyError: {e}  (the tokenizer has no entry for this character)")
print()
print("Real LLMs solve this by including ALL bytes in the vocabulary,")
print("or by using subword tokenization (see file 03).")
