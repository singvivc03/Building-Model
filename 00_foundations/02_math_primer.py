"""
Module 0 — Math primer (vectors, matrices, gradients) in NumPy
==============================================================

If NumPy is not installed, run once:
    pip install numpy

Then run this file:
    python 02_math_primer.py
"""

import numpy as np


# ===============================================================
# Part A — Vectors
# ===============================================================
print("=" * 50)
print("A. Vectors")
print("=" * 50)

cat     = np.array([0.9, 0.1, 0.2])   # imagine: [furry, vehicle-ness, food-ness]
kitten  = np.array([0.8, 0.1, 0.3])
car     = np.array([0.1, 0.95, 0.0])

# Dot product measures "how aligned" two vectors are.
# Big positive = similar. Near zero = unrelated. Negative = opposite.
print(f"cat · kitten = {np.dot(cat, kitten):.3f}   (should be HIGH — both furry animals)")
print(f"cat · car    = {np.dot(cat, car):.3f}    (should be LOW  — unrelated)")

# This is literally how attention "decides" what to pay attention to.
# We'll see it again in Module 2.


# ===============================================================
# Part B — Matrices and matrix multiplication
# ===============================================================
print()
print("=" * 50)
print("B. Matrices")
print("=" * 50)

# A "weight matrix" W maps a 3-dim vector to a 2-dim vector.
# This is the fundamental operation of a neural network layer.
W = np.array([
    [1.0, 0.0, -1.0],
    [0.5, 0.5,  0.5],
])

x = np.array([1.0, 2.0, 3.0])
y = W @ x   # the @ symbol means matrix multiply
print(f"W shape: {W.shape}")        # (2, 3)
print(f"x shape: {x.shape}")        # (3,)
print(f"y = W @ x = {y}  shape: {y.shape}")   # (2,)

# A full LLM is just hundreds of these matrix multiplications, stacked,
# with some non-linearities and attention sprinkled in. Truly.


# ===============================================================
# Part C — Gradient descent (the heart of training)
# ===============================================================
print()
print("=" * 50)
print("C. Gradient descent — learning by going downhill")
print("=" * 50)

# Goal: find the value of w that minimizes f(w) = (w - 4)^2
# The minimum is obviously at w = 4. Let's see if the machine finds it.

# The gradient (derivative) of (w-4)^2 with respect to w is 2*(w-4).
# If gradient is positive → we're too high, step LEFT.
# If gradient is negative → we're too low,  step RIGHT.

w = 0.0              # start far from the answer
learning_rate = 0.1  # how big a step we take each time

for step in range(20):
    loss = (w - 4) ** 2          # how wrong are we?
    grad = 2 * (w - 4)           # which way is downhill?
    w = w - learning_rate * grad # take a step downhill
    print(f"step {step:2d}  w={w:6.3f} grad={grad:6.3f} loss={loss:7.4f}")

print()
print(f"Final w = {w:.4f}   (true answer is 4.0)")
print("This tiny loop IS what training does, just with billions of parameters")
print("and a far more complex loss function.")
