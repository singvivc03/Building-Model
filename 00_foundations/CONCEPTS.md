# Module 0 — Foundations

Before we build anything, three short pieces of background. Don't memorize — just get a feel.

---

## 1. What *is* an AI model like Claude, really?

Strip away the marketing and a large language model (LLM) is one thing: **a function that predicts the next token given a sequence of previous tokens.**

```
input:  "The capital of France is"
output: probability distribution over every possible next token
        "Paris" → 0.91
        "the"   → 0.02
        "a"     → 0.01
        ...
```

That's it. The model picks the most likely (or samples) and appends it. Then it does it again. And again. Out comes a paragraph, an essay, a code review.

Everything we'll build — tokenizers, attention, transformers, training, RLHF, agents — exists to make that next-token prediction **better, faster, and more useful**.

Three layers of "better":

| Layer | What it does | We'll build it in |
|---|---|---|
| Architecture | How the model is shaped | Modules 1–3 |
| Training | How it learns from data | Modules 4–5 |
| Scaffolding | How we wrap it to do real work (agents, tools) | Module 6 |

---

## 2. The minimum math you need

Don't panic. We only need three concepts, all at the intuition level.

### Vectors

A vector is just a list of numbers: `[0.2, -1.4, 3.0]`. In LLMs, *everything* eventually becomes a vector. The word "cat" might become `[0.1, -0.3, 0.8, ...]` (in real models, around 4096 numbers). Vectors that point in similar directions are "similar". The vector for "cat" and "kitten" point in roughly the same direction. The vector for "cat" and "spreadsheet" don't.

### Matrices

A matrix is a grid of numbers — a list of vectors. The core operation in all of deep learning is **matrix multiplication**: combining a vector of inputs with a matrix of "weights" to produce a vector of outputs. The "weights" *are* what the model has learned. Training = adjusting the weights.

### Gradients (the calculus bit)

When the model is wrong, we need to know **which direction to nudge each weight** to make it less wrong. That direction is called the *gradient*. Imagine you're on a hillside in fog, trying to reach the valley. You feel the slope under your feet and step downhill. Repeat millions of times. That's training. The technical name is **gradient descent**.

That's it. Vectors, matrices, gradients. No calculus textbook required.

---

## 3. Why we use Python (and NumPy / PyTorch)

- **Python** — easy syntax, dominant language for ML
- **NumPy** — fast operations on vectors and matrices. We'll start here because the code is closest to the math.
- **PyTorch** — adds two superpowers: (a) it runs on a GPU, and (b) it computes gradients *automatically* (called "autograd"). We'll switch to PyTorch once concepts are solid.

---

## What's in this module

- `01_python_primer.py` — the Python you need to follow along. Runs in seconds.
- `02_math_primer.py` — vectors, matrices, and a tiny gradient descent example, in NumPy.

Read this file, then run the two scripts. When the output makes sense to you, move on to **Module 1 — Tokenization**.
