"""
Module 1.3 — Byte-Pair Encoding (BPE), trained from scratch
============================================================

This is (a simplified version of) the algorithm that produces Claude's,
GPT's, and Llama's tokenizers. Read the comments — the whole algorithm
fits in about 50 lines.

Run:
    python 03_bpe_tokenizer.py
"""

from collections import Counter


# ===================================================================
# Step 1. Get the initial "words", each split into its characters.
# We use '</w>' as a special end-of-word marker so the tokenizer
# can tell "low " (end of word) apart from "low" (start of "lower").
# ===================================================================
def get_word_freqs(text):
    freqs = Counter()
    for word in text.split():
        # Represent each word as a tuple of its chars, ending in </w>
        symbols = tuple(list(word) + ["</w>"])        
        freqs[symbols] += 1
    return freqs


# ===================================================================
# Step 2. Count every adjacent pair of symbols across all words.
# ===================================================================
def get_pair_counts(word_freqs):
    pairs = Counter()
    for symbols, freq in word_freqs.items():        
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


# ===================================================================
# Step 3. Merge the most common pair throughout the corpus.
# This is the "learn one new token" step.
# ===================================================================
def merge_pair(pair, word_freqs):
    a, b = pair    
    new_word_freqs = {}
    for symbols, freq in word_freqs.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            # If symbols[i:i+2] is the pair we're merging, glue them.                    
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(a + b)
                i += 2
            else:
                new_symbols.append(symbols[i]) #low
                i += 1
        new_word_freqs[tuple(new_symbols)] = freq
    return new_word_freqs


# ===================================================================
# Step 4. Train: just repeat steps 2 and 3 N times.
# ===================================================================
def train_bpe(text, num_merges):
    word_freqs = get_word_freqs(text)
    merges = []          # list of pairs we merged, in order
    for step in range(num_merges):
        pairs = get_pair_counts(word_freqs)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        word_freqs = merge_pair(best_pair, word_freqs)
        merges.append(best_pair)
        print(f"merge {step + 1:2d}: {best_pair[0]!r:>8} + {best_pair[1]!r:<8} "
              f"→ {(best_pair[0] + best_pair[1])!r}  "
              f"(appeared {pairs[best_pair]} times)")
    return merges, word_freqs


# ===================================================================
# Step 5. Use the learned merges to tokenize new text.
# ===================================================================
def encode(text, merges):
    out_tokens = []
    for word in text.split():
        symbols = list(word) + ["</w>"]
        # Apply each merge in the order it was learned.
        for a, b in merges:
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        out_tokens.extend(symbols)
    return out_tokens


# ===================================================================
# Demo
# ===================================================================
# A tiny "corpus" with deliberately repeated patterns so BPE has
# something obvious to learn.
corpus = (
    "low low low low low "
    "lower lower lower "
    "newest newest newest newest newest newest "
    "widest widest widest"
)

print("=" * 60)
print("Training BPE on a tiny corpus:")
print(f"  {corpus!r}")
print("=" * 60)

merges, final_word_freqs = train_bpe(corpus, num_merges=10)

print()
print("Final tokenized 'words' in the corpus:")
for symbols, freq in final_word_freqs.items():
    print(f"  {' '.join(symbols):30s}  (count={freq})")

print()
print("=" * 60)
print("Encoding new text with the learned tokenizer:")
print("=" * 60)
for sample in ["lowest", "newer", "wildest"]:
    print(f"  {sample!r:12s} → {encode(sample, merges)}")

print()
print("Observe:")
print(" • Frequent endings like 'est</w>' became a single token.")
print(" • A new word like 'wildest' still tokenizes fine — it just uses")
print("   smaller chunks for the unfamiliar 'wild' part.")
print(" • This is exactly why Claude can handle any input you throw at it.")
