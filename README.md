# gpt-zero

A full GPT-style transformer built from scratch in PyTorch, trained on Shakespeare. This is where [nanoLM](https://github.com/marzoukbaig14/nanoLM) ends up.

~10M parameters. No abstractions, no shortcuts. Just the math and backprop.

---

## What this is

A ground-up implementation of a character-level GPT. Every component is built and understood before use, following the full progression from [nanoLM](https://github.com/marzoukbaig14/nanoLM) — where the same approach was applied to bigrams, MLPs, and WaveNet-style models — all the way to a real transformer.

Built by following Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series. All his lectures and notes are freely available online.

---

## What's implemented

- **Bigram baseline** — simplest possible language model, predicts next token from current token only
- **Self-attention** — queries, keys, and values; scaled dot-product attention
- **Multi-head self-attention** — parallel attention heads concatenated and projected
- **Transformer block** — attention + feedforward network + residual connections + layer normalization
- **Positional encoding** — learned embeddings so the model knows token order
- **Autoregressive text generation** — sampling one character at a time at inference

---

## Architecture

| Hyperparameter | Value |
|---|---|
| Parameters | ~10M |
| Context length | 256 tokens |
| Embedding dimension | 384 |
| Attention heads | 6 |
| Transformer layers | 6 |
| Dropout | 0.2 |
| Training data | Shakespeare (~1MB) |

---

## Sample output

After training, the model generates text like:

```
GLOUCESTER:
What say you to the king? I pray thee, speak,
For I have heard the commons speak of it.

KING RICHARD II:
My lord, I am not well; I cannot speak.
```

Not Shakespeare. But close enough to fool your English teacher.

---

## Getting started

**Install dependencies**
```bash
pip install torch
```

**Prepare the data**
```bash
python data/prepare.py
```

**Train**
```bash
python train.py
```

**Generate text**
```bash
python sample.py
```

Training takes ~15 minutes on a GPU, longer on CPU.

---

## File structure

```
gpt-zero/
├── data/
│   └── prepare.py       # Downloads and tokenizes Shakespeare
├── model.py             # Full transformer implementation
├── train.py             # Training loop
├── sample.py            # Text generation
└── README.md
```

---

## Credits

- Architecture and approach from Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) and [nanoGPT](https://github.com/karpathy/nanoGPT)
- Built as the capstone of [nanoLM](https://github.com/marzoukbaig14/nanoLM)
