# gpt-zero

Character-level GPT built from scratch in PyTorch.
No abstractions, no shortcuts, just math and backprop.

## What This Is

Ground-up implementation of a GPT-style transformer — 
every component built and understood before use. 
Follows from [nano-LM](https://github.com/marzoukbaig14/nano-lm), 
where the same approach was applied to MLPs and WaveNet-style 
hierarchical models.

## Concepts Implemented

- Bigram language model baseline
- Self-attention (queries, keys, values)
- Multi-head attention
- Transformer block (attention + feedforward + residual + layernorm)
- Positional encoding
- Autoregressive text generation

## Stack

Python · PyTorch · Jupyter

## Structure

notebooks/    # step by step build
gpt.py        # final clean implementation

## Credits

Inspired by Andrej Karpathy's 
[Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series.
