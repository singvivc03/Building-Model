"""
Module 1.8 — What if we had NO tokenizer at all?
=================================================

A growing line of research asks: do we really need tokenization?
The model could just read raw bytes directly. Examples:

  • ByT5     (Google, 2021) — byte-level T5
  • CANINE   (Google, 2021) — character-level BERT with downsampling
  • MEGABYTE (Meta, 2023)   — hierarchical byte transformer
  • BLT      (Meta, 2024)   — Byte Latent Transformer with dynamic patching

This file shows what "no tokenizer" actually looks like — and the
brutal sequence-length penalty that's the reason it hasn't replaced
BPE in production yet.

Run:
    python 08_byte_only_no_tokenizer.py
"""


def show(text):
    chars = len(text)
    bytes_seq = list(text.encode("utf-8"))
    bytes_count = len(bytes_seq)
    print(f"Text: {text!r}")
    print(f"  chars: {chars}  |  bytes: {bytes_count}  "
          f"|  expansion ratio: {bytes_count/chars:.2f}x")
    print(f"  byte sequence: {bytes_seq}")

    # Show the recovery
    recovered = bytes(bytes_seq).decode("utf-8")
    assert recovered == text, "round-trip failed!"
    print(f"  round-trip OK ✓")
    print()


# -------------------------------------------------------------------
# Demo across different scripts
# -------------------------------------------------------------------
print("=" * 70)
print("Bytes-only 'tokenization' is just UTF-8 encoding")
print("=" * 70 + "\n")

show("hello")                # plain ASCII — 1 byte/char
show("café")                 # accented Latin — some 2-byte chars
show("你好")                  # Chinese — 3 bytes per char
show("🚀")                    # emoji — 4 bytes per char
show("Hello, café 你好 🚀!")   # mixed

# -------------------------------------------------------------------
# The cost story
# -------------------------------------------------------------------
print("=" * 70)
print("Why this is appealing")
print("=" * 70)
print(" + Vocabulary is exactly 256 (all byte values). Trivially small.")
print(" + Out-of-vocabulary is IMPOSSIBLE — every input is valid bytes.")
print(" + No tokenizer training step, no merges to maintain.")
print(" + Identical treatment for all languages and scripts.")
print(" + No 'tokens don't equal words' weirdness for users.")
print()
print("=" * 70)
print("Why production LLMs (Claude, GPT-4, Llama) still use BPE")
print("=" * 70)
print(" – Sequence length explodes. English BPE: ~0.75 tokens/word.")
print("   Bytes-only: ~5 bytes/word. Roughly 4–7× more tokens for same text.")
print(" – Attention is O(n²). 5× more tokens = ~25× more compute.")
print(" – Generation speed is roughly linear in tokens produced.")
print(" – Byte-level models like ByT5 work but are SLOW and need")
print("   special architectures (patching, hierarchical attention) to scale.")
print(" – Existing pretrained weights, fine-tuning tooling, and APIs")
print("   are all built around tokens. Switching costs are enormous.")
print()
print("=" * 70)
print("Where the field may be heading")
print("=" * 70)
print(" • MEGABYTE (2023) groups bytes into 'patches' and runs a fast")
print("   small model within patches + a larger model across patches.")
print(" • BLT (2024) makes the patch boundaries DYNAMIC — splitting on")
print("   high-entropy bytes — which beats fixed-size patches.")
print(" • If long-context attention keeps getting cheaper (FlashAttention,")
print("   sliding window, state-space models), byte-level may become viable.")
print(" • The big bet: removing tokenization would eliminate a whole class")
print("   of bugs (tokenizer drift, multilingual unfairness, math failure).")
