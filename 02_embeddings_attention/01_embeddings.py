"""
Module 2.1 — Embeddings
=======================

An embedding layer is just a (vocab_size × embedding_dim) matrix.
Row `i` is the vector for token `i`. To embed a sentence, look up rows.

In a real LLM, this matrix starts random and is LEARNED during training.
For this demo we'll hand-craft a tiny one so you can see the structure.

Run:
    python 01_embeddings.py
"""

import numpy as np


# -------------------------------------------------------------------
# A tiny "trained" embedding table.
# Pretend training has finished and these are the learned vectors.
# We pick numbers by hand so related words end up nearby.
# -------------------------------------------------------------------

# Our vocabulary: 6 tokens, each represented as a 4-dim vector.
# Imagine the 4 dimensions roughly mean:
#   [animal-ness, vehicle-ness, royalty-ness, gender-male-ness]
vocab = ["king", "queen", "man", "woman", "cat", "car"]

embedding_matrix = np.array([
    # animal  vehicle  royalty  male
    [0.0,     0.0,     0.9,     0.9],   # king
    [0.0,     0.0,     0.9,    -0.9],   # queen
    [0.0,     0.0,     0.0,     0.9],   # man
    [0.0,     0.0,     0.0,    -0.9],   # woman
    [0.9,     0.0,     0.0,     0.0],   # cat
    [0.0,     0.9,     0.0,     0.0],   # car
])

token_to_id = {tok: i for i, tok in enumerate(vocab)}
vocab_size, embedding_dim = embedding_matrix.shape
print(f"vocab_size={vocab_size}, embedding_dim={embedding_dim}")
print()


# -------------------------------------------------------------------
# Step 1. Look up an embedding.
# -------------------------------------------------------------------
def embed(token):
    return embedding_matrix[token_to_id[token]]

print("Embedding for 'king':", embed("king"))
print("Embedding for 'cat' :", embed("cat"))
print()


# -------------------------------------------------------------------
# Step 2. Embed a whole sentence by stacking rows.
# -------------------------------------------------------------------
sentence = ["king", "queen", "cat"]
sentence_embeddings = np.array([embed(tok) for tok in sentence])
print(f"Sentence: {sentence}")
print(f"Embedded shape: {sentence_embeddings.shape}   "
      f"(num_tokens × embedding_dim)")
print(sentence_embeddings)
print()


# -------------------------------------------------------------------
# Step 3. The famous geometric magic.
# In a properly trained embedding space, related concepts line up,
# and you can do arithmetic on word vectors.
# -------------------------------------------------------------------

def cosine_sim(a, b):
    """Cosine similarity: dot product normalized by lengths.
    Returns 1.0 if parallel, 0 if perpendicular, -1 if opposite."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Closest neighbors of 'cat'
target = embed("cat")
print("Cosine similarity of 'cat' with everyone else:")
for tok in vocab:
    sim = cosine_sim(target, embed(tok))
    print(f"  {tok:<8s}  {sim:+.3f}")
print()

# The classic analogy: king - man + woman ≈ queen
analogy = embed("king") - embed("man") + embed("woman")
print("king - man + woman =", analogy)
print("queen              =", embed("queen"))
print()
# Find the closest vocab word to the analogy result
print("Closest tokens to (king - man + woman):")
sims = [(tok, cosine_sim(analogy, embed(tok))) for tok in vocab]
for tok, s in sorted(sims, key=lambda x: -x[1]):
    print(f"  {tok:<8s}  {s:+.3f}")
print()
print("This 'word arithmetic' was the original demonstration that")
print("neural networks were learning genuine semantic structure.")
print("Real LLMs do this in 4096+ dimensions, learned from terabytes")
print("of text. The principle is identical.")
