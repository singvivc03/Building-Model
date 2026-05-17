"""
Module 2.1b — Watch embeddings actually LEARN
==============================================

In file 01 we hand-crafted an embedding matrix to demo the concept.
But in real LLMs, those numbers START RANDOM and are LEARNED from data.

This file shows the learning happening. We'll train a tiny embedding
from scratch using a simplified version of the Word2vec "skip-gram"
algorithm — the same algorithm that first demonstrated word arithmetic
back in 2013, and that is conceptually identical to how an LLM's input
embeddings get tuned during pretraining.

Run:
    python 01b_embedding_training.py
"""

import numpy as np

np.random.seed(0)


# -------------------------------------------------------------------
# 1. Vocabulary and "training data"
# -------------------------------------------------------------------
# In real Word2vec, the training pairs come from co-occurrence in
# actual text (words that appear near each other in real sentences).
# We'll just hard-code which words "belong together" semantically.
vocab = [
    "king", "queen", "prince", "princess", "man", "woman",
    "cat", "kitten", "dog", "puppy",
    "car", "truck", "bus", "van",
]
w2i = {w: i for i, w in enumerate(vocab)}

# Positive pairs: should end up CLOSE in embedding space.
positive_pairs = [
    ("king", "queen"), ("king", "prince"), ("queen", "princess"),
    ("prince", "princess"), ("man", "woman"), ("king", "man"),
    ("queen", "woman"),
    ("cat", "kitten"), ("dog", "puppy"), ("cat", "dog"),
    ("kitten", "puppy"),
    ("car", "truck"), ("car", "bus"), ("truck", "van"),
    ("bus", "van"),
]


# -------------------------------------------------------------------
# 2. Random initialization of the embedding matrix
# -------------------------------------------------------------------
# In real LLMs this is typically Normal(0, 0.02). Small starting
# values prevent activations from exploding through the network.
d_model = 6     # embedding dimension
E = np.random.randn(len(vocab), d_model) * 0.3


def cosine(a, b):
    """Cosine similarity in [-1, 1]."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def show_neighbors(label):
    print(f"\n{label}")
    for w in ["king", "cat", "car"]:
        sims = [(other, cosine(E[w2i[w]], E[w2i[other]]))
                for other in vocab if other != w]
        sims.sort(key=lambda x: -x[1])
        top3 = ", ".join(f"{other}({s:+.2f})" for other, s in sims[:3])
        print(f"  closest to {w:<8s} → {top3}")


show_neighbors("BEFORE TRAINING (random init — no semantic structure):")


# -------------------------------------------------------------------
# 3. Training: pull together / push apart
# -------------------------------------------------------------------
# Intuition: for every "positive" pair we move the two vectors a step
# toward each other; for randomly-sampled "negative" pairs we move them
# slightly apart. This is the heart of Word2vec's skip-gram-with-
# negative-sampling. We use plain numpy and explicit updates so you
# can see the gradient story.
#
# The same update rule is what happens inside an LLM during pretraining
# — except the "positive pair" signal comes from the next-token
# prediction loss, propagated back through the network into the
# embedding rows.

lr_pos = 0.05      # how hard to pull paired words together
lr_neg = 0.02      # how hard to push random pairs apart
epochs = 800
neg_per_epoch = 25

for epoch in range(epochs):
    # POSITIVE updates — pull paired words toward each other
    for w1, w2 in positive_pairs:
        i, j = w2i[w1], w2i[w2]
        diff = E[j] - E[i]
        E[i] += lr_pos * diff
        E[j] -= lr_pos * diff

    # NEGATIVE updates — push random pairs apart
    for _ in range(neg_per_epoch):
        i = np.random.randint(len(vocab))
        j = np.random.randint(len(vocab))
        if i == j:
            continue
        if (vocab[i], vocab[j]) in positive_pairs or (vocab[j], vocab[i]) in positive_pairs:
            continue
        diff = E[j] - E[i]
        E[i] -= lr_neg * diff
        E[j] += lr_neg * diff

    # Periodic normalization to keep magnitudes sane
    if epoch % 50 == 0:
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E = E / (norms + 1e-9)


show_neighbors("\nAFTER TRAINING (embeddings have organized themselves):")


# -------------------------------------------------------------------
# 4. The famous "word arithmetic"
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("Word arithmetic test:  king - man + woman = ?")
print("=" * 60)
analogy = E[w2i["king"]] - E[w2i["man"]] + E[w2i["woman"]]
sims = [(w, cosine(analogy, E[w2i[w]])) for w in vocab]
sims.sort(key=lambda x: -x[1])
print("Closest words to (king - man + woman):")
for w, s in sims[:5]:
    print(f"  {w:<10s}  {s:+.3f}")
print()
print("If 'queen' is at the top (or very close), the embedding space")
print("has actually learned the 'gender' direction as a vector!")
print()
print("Takeaways:")
print("  • The numbers in an embedding row are NOT designed — they're")
print("    a side-effect of optimizing some objective (here: pull pairs")
print("    together; in real LLMs: predict the next token).")
print("  • Geometric relationships in embedding space carry meaning")
print("    that nobody hand-coded — they emerge.")
print("  • An LLM's input embedding matrix is the SAME idea, just bigger:")
print("    100k-vocab × 4096+ dims, trained on trillions of tokens.")
