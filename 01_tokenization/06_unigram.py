"""
Module 1.6 — Unigram Language Model tokenization
================================================

Unigram is the OTHER major modern algorithm (alongside BPE/WordPiece).
It is the default in Google's SentencePiece library, used by T5,
ALBERT, mBART, and most multilingual LLMs.

Conceptual difference vs BPE/WordPiece:
  BPE/WordPiece BUILD UP a vocab by greedy merging.
  Unigram      START BIG (every plausible substring) and PRUNE down.

Big advantage: at tokenization time, Unigram considers MANY possible
segmentations of an input and picks the one with the highest total
probability. That uses Viterbi-style dynamic programming and is more
principled than BPE's greedy left-to-right application of merges.

This file implements the **Viterbi tokenizer** given a pre-built vocab
with probabilities. (Real Unigram training uses Expectation–
Maximization to estimate the probabilities; we skip that part for
clarity and just use observed substring frequencies.)

Run:
    python 06_unigram.py
"""

import math
from collections import Counter


# -------------------------------------------------------------------
# 1. Build a "trained" Unigram vocabulary.
#    In real Unigram, this is the result of an iterative EM/pruning
#    process. Here we just take all observed substrings up to length 6
#    from a corpus, with their frequencies as proxy probabilities.
# -------------------------------------------------------------------
corpus = (
    "the quick brown fox jumps over the lazy dog "
    "the cat sat on the mat "
    "the dog ran in the rain "
    "quickly quickly the fox jumps"
)

substrings = Counter()
MAX_LEN = 6
for word in corpus.split():
    for start in range(len(word)):
        for end in range(start + 1, min(start + MAX_LEN, len(word)) + 1):
            substrings[word[start:end]] += 1

# Always include single characters so we never fail to tokenize
for ch in set(corpus):
    if ch != " " and ch not in substrings:
        substrings[ch] = 1

total = sum(substrings.values())
vocab = {tok: count / total for tok, count in substrings.items()}
print(f"Vocabulary has {len(vocab)} candidate tokens")
print(f"Top 8 by probability: ")
for tok, p in sorted(vocab.items(), key=lambda x: -x[1])[:8]:
    print(f"   {tok!r:10s}  p={p:.4f}")
print()


# -------------------------------------------------------------------
# 2. Viterbi tokenization.
#    Find the segmentation of `word` that maximizes the sum of
#    log-probabilities. Classic dynamic programming:
#       best_score[i] = best log-prob for word[:i]
#    For each position i, try every possible "last token" ending at i.
# -------------------------------------------------------------------
def viterbi_tokenize(word, vocab):
    n = len(word)
    best_score = [-math.inf] * (n + 1)
    back = [0] * (n + 1)
    best_score[0] = 0.0

    for i in range(1, n + 1):
        for j in range(i):
            tok = word[j:i]
            if tok in vocab:
                score = best_score[j] + math.log(vocab[tok])
                if score > best_score[i]:
                    best_score[i] = score
                    back[i] = j

        # Fallback: if no in-vocab token ends here, take a single char.
        if best_score[i] == -math.inf:
            best_score[i] = best_score[i - 1] + math.log(1e-9)
            back[i] = i - 1

    # Reconstruct the chosen segmentation by walking the back-pointers.
    tokens = []
    i = n
    while i > 0:
        j = back[i]
        tokens.append(word[j:i])
        i = j
    return list(reversed(tokens))


# -------------------------------------------------------------------
# Demo
# -------------------------------------------------------------------
print("Tokenizing sample words with Viterbi:")
print("-" * 50)
for word in ["the", "quickly", "running", "jumping", "tokenization", "rainbow"]:
    tokens = viterbi_tokenize(word, vocab)
    print(f"  {word!r:15s} → {tokens}")

print()
print("Notice:")
print("  • Unigram allows MULTIPLE plausible segmentations and picks")
print("    the most probable one globally — not greedy left-to-right.")
print("  • That makes it robust to weird inputs and slightly more")
print("    consistent across languages than BPE.")
print("  • Real Unigram trains the probabilities with EM and prunes")
print("    the vocabulary to a target size (e.g. 32k).")
print("  • SentencePiece (the library) lets you choose BPE or Unigram.")
print("    T5 and many multilingual models pick Unigram.")
