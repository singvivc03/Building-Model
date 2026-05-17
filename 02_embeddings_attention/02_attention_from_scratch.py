"""
Module 2.2 — Scaled dot-product attention, from scratch
========================================================

The single most important formula in modern AI:

    Attention(Q, K, V) = softmax( Q · K^T / sqrt(d) ) · V

We'll build it step by step in NumPy. After this file you'll have
SEEN every matrix involved. No magic left.

Run:
    python 02_attention_from_scratch.py
"""

import numpy as np

np.random.seed(0)         # so the random numbers are reproducible


# -------------------------------------------------------------------
# Setup: 4 tokens, each embedded as a 6-dim vector.
# This is the OUTPUT of the embedding layer — the "X" matrix.
# -------------------------------------------------------------------
n = 4          # sequence length (number of tokens)
d = 6          # embedding dimension

X = np.random.randn(n, d)
print(f"Input X (the sentence's embeddings):  shape {X.shape}")
print(X.round(2))
print()


# -------------------------------------------------------------------
# Step 1. The three learned projection matrices.
# In a real model these are TRAINED. Here we just init them randomly.
# -------------------------------------------------------------------
W_Q = np.random.randn(d, d) * 0.1
W_K = np.random.randn(d, d) * 0.1
W_V = np.random.randn(d, d) * 0.1

# Project X into queries, keys, values.
Q = X @ W_Q                # (n, d)
K = X @ W_K                # (n, d)
V = X @ W_V                # (n, d)
print(f"Q shape: {Q.shape}     K shape: {K.shape}     V shape: {V.shape}")
print()


# -------------------------------------------------------------------
# Step 2. Score every token against every other token.
# scores[i, j] = how much should token i attend to token j?
# -------------------------------------------------------------------
scores = Q @ K.T                       # (n, n)
scores = scores / np.sqrt(d)           # the famous sqrt(d) scaling

print(f"Raw scores (Q · K^T / sqrt(d)):  shape {scores.shape}")
print(scores.round(2))
print()


# -------------------------------------------------------------------
# Step 3. Causal mask — token i may only attend to positions 0..i.
# We set future positions to a huge negative number so that after
# softmax they become ~0.
#
# The mask looks like (1 = allowed, 0 = forbidden):
#   row 0 attends to: 0
#   row 1 attends to: 0, 1
#   row 2 attends to: 0, 1, 2
#   row 3 attends to: 0, 1, 2, 3
# -------------------------------------------------------------------
mask = np.tril(np.ones((n, n)))        # lower-triangular matrix of 1s
print("Causal mask (1=allowed, 0=forbidden):")
print(mask)
print()

scores_masked = np.where(mask == 1, scores, -1e9)
print("Scores after causal masking:")
print(scores_masked.round(2))
print()


# -------------------------------------------------------------------
# Step 4. Softmax along each row to get attention weights.
# Each row sums to 1.0 — it's a probability distribution over
# which tokens to attend to.
# -------------------------------------------------------------------
def softmax(x, axis=-1):
    # subtract the max for numerical stability before exp
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

attn_weights = softmax(scores_masked, axis=-1)
print("Attention weights (each row sums to 1):")
print(attn_weights.round(3))
print("Row sums:", attn_weights.sum(axis=1).round(3))   # should be [1, 1, 1, 1]
print()


# -------------------------------------------------------------------
# Step 5. Use the attention weights to take a weighted average of V.
# -------------------------------------------------------------------
output = attn_weights @ V              # (n, d)
print(f"Attention output: shape {output.shape}")
print(output.round(2))
print()


# -------------------------------------------------------------------
# Bundle it all up into one clean function.
# -------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V, mask=None):
    d = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d)
    if mask is not None:
        scores = np.where(mask == 1, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights

out, w = scaled_dot_product_attention(Q, K, V, mask=mask)
assert np.allclose(out, output)
print("✓ Clean function gives identical output. Attention is built.")
print()
print("What you just saw:")
print("  • Every output row is a blend of V's rows,")
print("    weighted by how relevant each token is to the current one.")
print("  • Look at the attention_weights matrix — that's literally")
print("    'who is the model paying attention to.' This is what gets")
print("    visualized in those famous transformer attention diagrams.")
