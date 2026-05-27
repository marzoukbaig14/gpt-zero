# gpt-zero

A full GPT-style transformer built from scratch in PyTorch, trained on Shakespeare. This is where [nanoLM](https://github.com/marzoukbaig14/nanoLM) ends up.

~10M parameters. No abstractions, no shortcuts. Just the math and backprop.

## What this is

Ground-up implementation of a character-level GPT. Every component is built and understood before use, following the full progression from [nanoLM](https://github.com/marzoukbaig14/nanoLM) where the same approach was applied to MLPs and WaveNet-style models.

## What's implemented

- Bigram baseline
- Self-attention (queries, keys, values)
- Multi-head self-attention
- Transformer block (attention + feedforward + residual connections + layernorm)
- Positional encoding
- Autoregressive text generation

The trained model generates coherent Shakespearean text.

## Structure
