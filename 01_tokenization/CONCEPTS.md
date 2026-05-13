# Module 1 — Tokenization: how text becomes numbers

## The problem

A neural network is a giant pile of matrix multiplications. Matrix multiplications need **numbers**. Text is not numbers — it's strings of characters.

So before a model can read anything, we need a procedure that turns text into a sequence of integers, and then back the other way:

```
"hello world"  →  [15, 23, 4, ...]   ← what the model actually sees
[15, 23, 4]    →  "hello world"      ← how we read its output
```

A **tokenizer** does exactly this. A **token** is one unit (a number) that maps to a piece of text. The full mapping between tokens and text pieces is the **vocabulary**.

Three flavors to know, from simplest to most realistic:

---

## 1. Character-level tokenization

Each character is its own token.

```
"hello"  →  ['h', 'e', 'l', 'l', 'o']  →  [7, 4, 11, 11, 14]
```

**Pros:** dead simple. Tiny vocabulary (~100 tokens for English). Can never see a "new" word it can't represent — every word is just a sequence of letters.

**Cons:** sequences are very long. "Tokenization is fun" is 19 tokens this way. That means more compute per sentence. The model also has to "learn" what a word even is from scratch.

---

## 2. Word-level tokenization

Each whole word is a token.

```
"hello world"  →  ['hello', 'world']  →  [42, 87]
```

**Pros:** sequences are short and intuitive. Each token carries a lot of meaning.

**Cons:**
- Vocabulary explodes (hundreds of thousands of words).
- **Out-of-vocabulary (OOV) problem**: anything the tokenizer didn't see during training (typos, names, new slang, code, other languages) becomes `<UNK>` — the model literally can't read it.

That's a dealbreaker. Real LLMs need to handle *anything* a user types.

---

## 3. Subword tokenization — **Byte-Pair Encoding (BPE)**

The trick used by Claude, GPT, Llama, and almost every modern LLM. The idea: tokens are **chunks of characters** of varying length. Common chunks (like `" the"`, `"ing"`, `"tion"`) become single tokens. Rare or new strings get broken into smaller chunks down to individual bytes.

```
"tokenization"  →  ['token', 'ization']        →  [1234, 5678]
"asdfqwerty"    →  ['as', 'df', 'q', 'wer', 'ty']  →  [...]   (never seen before, still works)
```

**Pros:**
- No OOV problem. Worst case, a word falls back to byte-level.
- Short sequences for common words, longer for rare ones — efficient.
- The vocabulary is *learned from data*, so it adapts to the corpus.

**Cons:**
- Training the tokenizer is its own little algorithm (we'll build it).
- Token boundaries can be unintuitive (the famous "tokens aren't words" thing — a single space is often part of the next token).

### How BPE actually works (the algorithm, plain English)

1. Start with the vocabulary = every individual byte/character that appears in your text.
2. Count every adjacent pair of tokens in the corpus. e.g. `'l','o'` appears 9000 times.
3. **Merge** the most common pair into a single new token. Add it to the vocabulary.
4. Repeat steps 2–3 until you have the vocabulary size you want (often 32k–100k).

Once trained, encoding is just "apply the merges in the order you learned them."

The first merge might combine `' '` + `'t'` → `' t'`. A later merge might combine `' t'` + `'he'` → `' the'`. After a few thousand merges, `' the'` becomes one of the most common single tokens in English text.

---

## What's in this module

- `01_char_tokenizer.py` — the trivial case, in ~30 lines
- `02_word_tokenizer.py` — see the OOV problem in action
- `03_bpe_tokenizer.py` — train a real BPE tokenizer from scratch on a small text. Watch the merges happen.

Run them in order. After this module, every piece of text in our project is just a list of integers. Onward to **Module 2 — Embeddings & Attention**, where those integers become *meaning*.
