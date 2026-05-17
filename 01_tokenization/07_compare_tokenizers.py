"""
Module 1.7 — Side-by-side comparison of tokenization approaches
================================================================

Tokenize the SAME paragraph with every approach we've built, plus
GPT-4's real tokenizer (if installed), and report the token count
for each. Fewer tokens = less compute, lower API cost, faster output.

Run:
    python 07_compare_tokenizers.py
"""

import re
from collections import Counter


# Sample text — mixed English, French, Chinese, code, emoji
text = (
    "Tokenization is fun! It turns text into integers so the model "
    "can do math on language. Even unusual inputs like café, 你好, "
    "and 🚀 should encode cleanly. def f(x): return x**2"
)

print("=" * 75)
print(f"Sample text:")
print(f"  {text}")
print(f"Length: {len(text)} characters, {len(text.encode('utf-8'))} UTF-8 bytes")
print("=" * 75)


# -------------------------------------------------------------------
# 1. Character-level (one token per char)
# -------------------------------------------------------------------
char_tokens = list(text)
print(f"\n[1] Character-level     : {len(char_tokens):>4} tokens")
print(f"    preview: {char_tokens[:14]} ...")

# -------------------------------------------------------------------
# 2. Bytes-only (UTF-8) — what a tokenizer-free model would see
# -------------------------------------------------------------------
byte_tokens = list(text.encode("utf-8"))
print(f"\n[2] Bytes only          : {len(byte_tokens):>4} tokens   (vocab_size = 256)")
print(f"    preview: {byte_tokens[:14]} ...")

# -------------------------------------------------------------------
# 3. Word-level (whitespace + punctuation split)
# -------------------------------------------------------------------
word_tokens = re.findall(r"\w+|[^\w\s]", text)
print(f"\n[3] Word-level (regex)  : {len(word_tokens):>4} tokens")
print(f"    preview: {word_tokens[:8]} ...")
print(f"    note   : 'café', '你好', '🚀' may collapse to OOV in a real word tokenizer")

# -------------------------------------------------------------------
# 4. A toy byte-level BPE trained on this text only
# -------------------------------------------------------------------
def text_to_byte_symbols(word):
    return tuple((b,) for b in word.encode("utf-8"))


def train_tiny_bpe(corpus, num_merges):
    freqs = Counter()
    for w in corpus.split():
        freqs[text_to_byte_symbols(w)] += 1
    for _ in range(num_merges):
        pair_counts = Counter()
        for syms, f in freqs.items():
            for i in range(len(syms) - 1):
                pair_counts[(syms[i], syms[i + 1])] += f
        if not pair_counts:
            break
        a, b = max(pair_counts, key=pair_counts.get)
        merged = a + b
        new_freqs = {}
        for syms, f in freqs.items():
            new_syms, i = [], 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            new_freqs[tuple(new_syms)] = f
        freqs = new_freqs
    # Count total tokens
    return sum(len(syms) * f for syms, f in freqs.items())


toy_bpe_count = train_tiny_bpe(text, num_merges=50)
print(f"\n[4] Toy byte-level BPE  : {toy_bpe_count:>4} tokens   (after 50 merges)")

# -------------------------------------------------------------------
# 5. GPT-4's real tokenizer (if tiktoken is installed)
# -------------------------------------------------------------------
print()
try:
    import tiktoken
    for enc_name in ("cl100k_base", "o200k_base"):
        try:
            enc = tiktoken.get_encoding(enc_name)
            toks = enc.encode(text)
            previews = [enc.decode([t]) for t in toks[:10]]
            print(f"[5] tiktoken {enc_name:<14}: {len(toks):>4} tokens")
            print(f"    preview: {previews} ...")
        except Exception as e:
            print(f"[5] tiktoken {enc_name}: error: {e}")
except ImportError:
    print("[5] tiktoken not installed — `pip install tiktoken` to compare GPT-4 directly")

print()
print("=" * 75)
print("How to read this")
print("=" * 75)
print(" • Real production tokenizers (GPT-4 cl100k, Claude) compress to the")
print("   FEWEST tokens because they have a large learned vocab (~100k).")
print(" • Byte-level is the most compatible (never fails) but explodes seq len.")
print(" • Word-level is short but breaks on non-English / new words.")
print(" • Trade-off recap: cost per token ≈ token count × O(n²) attention.")
print("   For a 200k-context window, every extra token matters.")
