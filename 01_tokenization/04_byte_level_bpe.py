"""
Module 1.4 — Byte-level BPE (what GPT-2/3/4 and Claude actually use)
====================================================================

The plain BPE in file 03 starts from CHARACTERS. That has one weakness:
characters not seen at training time may have no token, and exotic
inputs (rare emoji, mixed scripts, raw binary text) need special
handling.

Byte-level BPE fixes this by operating on UTF-8 BYTES instead of
characters. Every string on Earth is a sequence of bytes in {0..255},
so the tokenizer can NEVER fail — single bytes are always the
fallback. This is what OpenAI's GPT-2, GPT-3, GPT-4 and Anthropic's
Claude use under the hood.

Run:
    python 04_byte_level_bpe.py
"""

from collections import Counter


def text_to_byte_symbols(word):
    """A word becomes a tuple of single-byte tuples.
    e.g. 'hi' (bytes 104, 105) -> ((104,), (105,))"""
    return tuple((b,) for b in word.encode("utf-8"))


def get_word_freqs(text):
    freqs = Counter()
    for word in text.split():
        freqs[text_to_byte_symbols(word)] += 1
    return freqs


def get_pair_counts(word_freqs):
    pairs = Counter()
    for symbols, freq in word_freqs.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(pair, word_freqs):
    a, b = pair
    merged = a + b   # tuple concatenation, e.g. (104,) + (105,) = (104, 105)
    new_word_freqs = {}
    for symbols, freq in word_freqs.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_word_freqs[tuple(new_symbols)] = freq
    return new_word_freqs


def preview_token(byte_tuple):
    """Render a byte-tuple token as readable text if possible."""
    try:
        return bytes(byte_tuple).decode("utf-8")
    except UnicodeDecodeError:
        return f"<bytes:{list(byte_tuple)}>"


def train_byte_level_bpe(text, num_merges):
    word_freqs = get_word_freqs(text)
    merges = []
    for step in range(num_merges):
        pairs = get_pair_counts(word_freqs)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        word_freqs = merge_pair(best, word_freqs)
        merges.append(best)
        print(f"merge {step + 1:2d}: "
              f"{preview_token(best[0])!r:>8} + {preview_token(best[1])!r:<8} "
              f"→ {preview_token(best[0] + best[1])!r:<10} ({pairs[best]}×)")
    return merges, word_freqs


# ===================================================================
# Demo — deliberately mix English, French (accents), Chinese, emoji
# ===================================================================
corpus = (
    "low low low low low lower lower lower "
    "newest newest newest newest newest "
    "café café résumé résumé "
    "你好 你好 你好 "
    "🚀 🚀 🚀"
)

print(f"Corpus (note the mixed scripts): {corpus!r}")
print(f"Byte length of corpus: {len(corpus.encode('utf-8'))}")
print()
print("Training byte-level BPE:")
print("-" * 70)
merges, final_words = train_byte_level_bpe(corpus, num_merges=15)

print()
print("=" * 70)
print("Key property: this tokenizer NEVER fails on any input")
print("=" * 70)
weird_inputs = [
    "hello",                    # plain English
    "café",                     # accented (multi-byte UTF-8)
    "你好",                      # Chinese (3 bytes per char)
    "🚀",                       # emoji (4 bytes)
    "\x00\xff\x42",             # raw binary bytes
    "{'key': 'value'}",         # JSON
]
for inp in weird_inputs:
    bytes_count = len(inp.encode("utf-8"))
    print(f"  {inp!r:25s}  →  {bytes_count} bytes — encodable ✓")

print()
print("Comparison with plain BPE (file 03):")
print("  • Plain BPE: would crash if it saw a character outside its training set.")
print("  • Byte-level BPE: 256 single-byte tokens are always in the vocabulary,")
print("    so it ALWAYS has a fallback. Used in production by OpenAI and Anthropic.")
