# Building a Model From Scratch — End-to-End Journey

Goal: understand how an AI model like Claude is built, from the very bottom (turning text into numbers) all the way up to **agents and MCP**. Balanced theory + working code. Every module has a `CONCEPTS.md` you read first, then Python files you run and tinker with.

---

## How to use this project

1. Start at Module 0. Read `CONCEPTS.md`. Then run the Python files in order.
2. Don't skip modules — each one builds on the previous.
3. Tinker. Change a number. Break things. Read the error. That's how intuition forms.
4. By Module 4 you'll have a tiny LLM that generates Shakespeare-like text. By Module 6 you'll have wrapped it in an agent with tools.

---

## The roadmap

### Module 0 — Foundations
What an AI model actually *is*, plus the minimum Python and math you need. Vectors, matrices, gradients — at the intuition level.

### Module 1 — Tokenization (text → numbers)
Computers can't read words. They read numbers. We'll build three tokenizers, smallest to most realistic:
- Character-level (the trivial case)
- Word-level (intuitive but breaks on new words)
- **Byte-Pair Encoding (BPE)** — what real LLMs like Claude and GPT use

### Module 2 — Embeddings & Attention
- Embeddings: how a token ID becomes a "meaning vector"
- The big idea: **attention**. We'll build it from scratch in NumPy first, then PyTorch.
- Multi-head attention — why multiple "perspectives" matter

### Module 3 — The Transformer Block
Putting attention together with feedforward layers, residual connections, and layer norm. By the end of this module you'll have a full **decoder-only transformer** (the GPT/Claude family architecture) coded from scratch.

### Module 4 — Training
- Loss functions (cross-entropy: "how wrong were we?")
- Optimizers (SGD → Adam)
- The training loop
- **Pretraining**: next-token prediction at scale
- Train our tiny transformer on a real text corpus

### Module 5 — From raw model to helpful assistant
A pretrained model just predicts the next token. To get something useful like Claude, you need:
- **Supervised fine-tuning (SFT)** — show it examples of good answers
- **RLHF** (Reinforcement Learning from Human Feedback) — what made ChatGPT work
- **DPO** — a simpler modern alternative
- **Constitutional AI** — Anthropic's approach (what shapes Claude's character)

### Module 6 — Agents, Tools & MCP
- What makes a model an **agent** (not just a chatbot)
- **Tool use** / function calling — how the model decides to call a calculator or search the web
- The **ReAct** pattern (reasoning + acting in a loop)
- Memory and context management
- **MCP — Model Context Protocol** — the open standard for connecting models to tools and data sources. We'll build a tiny MCP server and client.

---

## Progress tracker

- [ ] Module 0 — Foundations
- [ ] Module 1 — Tokenization
- [ ] Module 2 — Embeddings & Attention
- [ ] Module 3 — The Transformer Block
- [ ] Module 4 — Training
- [ ] Module 5 — Fine-tuning, RLHF, DPO, Constitutional AI
- [ ] Module 6 — Agents, Tools, MCP

Tick them off as you go. Ask follow-up questions any time — depth, math derivations, alternative explanations, more exercises. This is *your* journey.
