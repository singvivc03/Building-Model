"""
Module 2.3 — Multi-head attention
==================================

One attention "head" learns one relationship. Multi-head attention runs
several heads in parallel, each with its own W_Q, W_K, W_V — so the model
can track multiple kinds of relationships at once (syntax, coreference,
topic, etc.) — then concatenates and projects the result.

Run:
    python 03_multi_head_attention.py
"""

import numpy as np

np.random.seed(0)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d = Q.shape[-1]
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d)
    if mask is not None:
        scores = np.where(mask == 1, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V


class MultiHeadAttention:
    """
    Shapes:
      input X         : (n, d_model)
      per-head dim    : d_head = d_model // num_heads
      after splitting : (num_heads, n, d_head)
      output          : (n, d_model)
    """

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must divide evenly by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Single big projection matrices — we'll RESHAPE to split into heads.
        # This is a standard trick: it's mathematically identical to having
        # num_heads separate small matrices, but more efficient.
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1   # final projection

    def __call__(self, X, mask=None):
        n = X.shape[0]

        # Project to Q, K, V — shape (n, d_model)
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        # Split into heads: (n, d_model) → (n, num_heads, d_head) → (num_heads, n, d_head)
        def split_heads(M):
            return M.reshape(n, self.num_heads, self.d_head).transpose(1, 0, 2)

        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        # Per-head attention — runs all heads in parallel via broadcasting.
        head_outs = scaled_dot_product_attention(Qh, Kh, Vh, mask=mask)
        # head_outs shape: (num_heads, n, d_head)

        # Concatenate heads back together: (n, num_heads * d_head) = (n, d_model)
        concat = head_outs.transpose(1, 0, 2).reshape(n, self.d_model)

        # Final linear projection
        return concat @ self.W_O


# -------------------------------------------------------------------
# Demo
# -------------------------------------------------------------------
n = 5           # 5 tokens in our sentence
d_model = 8     # embedding dimension
num_heads = 4   # 4 attention heads, each of size 8/4 = 2

X = np.random.randn(n, d_model)
mask = np.tril(np.ones((n, n)))           # causal mask

mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
out = mha(X, mask=mask)

print(f"Input  shape: {X.shape}")
print(f"Output shape: {out.shape}     (should match input shape)")
print()
print("Output:")
print(out.round(2))
print()
print("Key takeaways:")
print(f"  • {num_heads} heads ran in parallel, each looking at a different")
print(f"    'subspace' of the {d_model}-dim representation.")
print("  • Output shape matches input shape — multi-head attention is a")
print("    shape-preserving transformation. That's why we can stack many")
print("    of these layers in a row (which is exactly what Module 3 does).")
print()
print("✓ You now have all the pieces of a transformer except the wrapper.")
print("  Next module: positional encoding + feedforward + residuals + LayerNorm,")
print("  assembled into a full decoder-only transformer block. We'll also")
print("  switch from NumPy to PyTorch so we can train it on real text.")
