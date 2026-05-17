"""
Module 1.5 — WordPiece (what BERT uses)
========================================

WordPiece is almost identical to BPE — same start, same merging step —
but uses a SMARTER rule for picking WHICH pair to merge.

  BPE:        pick the pair with the highest count
  WordPiece:  pick the pair with the highest "score"
                  score(a, b) = count(ab) / (count(a) * count(b))

That score is a likelihood ratio: "how much more common is (a,b)
together than if they were independent?" This prevents merging
pairs like (' ', 'e') just because both pieces are common — it
favors pairs that genuinely belong together as a unit.

Run:
    python 05_wordpiece.py
"""

from collections import Counter


# Same data structures as BPE: words are tuples of symbols.
def get_word_freqs(text):
    freqs = Counter()
    for word in text.split():
        symbols = tuple(list(word) + ["</w>"])
        freqs[symbols] += 1
    return freqs


def get_symbol_counts(word_freqs):
    counts = Counter()
    for symbols, freq in word_freqs.items():
        for s in symbols:
            counts[s] += freq
    return counts


def get_pair_counts(word_freqs):
    pairs = Counter()
    for symbols, freq in word_freqs.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(pair, word_freqs):
    a, b = pair
    new = {}
    for symbols, freq in word_freqs.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(a + b)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new[tuple(new_symbols)] = freq
    return new


def train_wordpiece(text, num_merges):
    word_freqs = get_word_freqs(text)
    for step in range(num_merges):
        symbol_counts = get_symbol_counts(word_freqs)
        pair_counts = get_pair_counts(word_freqs)
        if not pair_counts:
            break

        # WordPiece's key trick: score = count(ab) / (count(a) * count(b))
        def score(pair):
            a, b = pair
            return pair_counts[pair] / (symbol_counts[a] * symbol_counts[b])

        best = max(pair_counts, key=score)
        word_freqs = merge_pair(best, word_freqs)
        print(f"merge {step + 1:2d}: "
              f"{best[0]!r:>10} + {best[1]!r:<10} → {(best[0] + best[1])!r:<12} "
              f"count={pair_counts[best]:>2}  score={score(best):.4f}")
    return word_freqs


# ===================================================================
# Demo on the SAME corpus we used for BPE — compare merge orders!
# ===================================================================
corpus = (
    "low low low low low "
    "lower lower lower "
    "newest newest newest newest newest newest "
    "widest widest widest"
)
print(f"Corpus: {corpus!r}\n")
print("Running WordPiece (likelihood-ratio merges):")
print("-" * 70)
final = train_wordpiece(corpus, num_merges=10)

print()
print("=" * 70)
print("Compare to BPE's output for the same corpus (in 03_bpe_tokenizer.py):")
print("=" * 70)
print(" • BPE merged ('w','e') first because it had the highest RAW count (9).")
print(" • WordPiece may merge different pairs first because what wins is")
print("   the highest likelihood RATIO — pairs that occur 'unusually often'")
print("   relative to how common their pieces are individually.")
print()
print("In practice both produce similar-quality vocabularies. WordPiece")
print("is what BERT, DistilBERT, and ELECTRA use; BPE/byte-level BPE")
print("is what GPT/Claude/Llama use.")
