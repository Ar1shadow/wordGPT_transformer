# wordGPT_transformer

A from-scratch, character-level language model built step by step in PyTorch — from a simple Bigram baseline up to a full decoder-only Transformer (GPT). Trained on a character corpus (`input.txt`, Tiny Shakespeare).

English | [中文](README.zh-CN.md)

## Overview

This project implements a GPT-style autoregressive language model from the ground up, with no high-level transformer libraries. It is meant as a learning project that traces the path from n-gram modeling to modern attention-based architectures.

```
input -> token embedding -> [Transformer block] x N -> LayerNorm -> logits
block: LayerNorm -> Multi-Head Self-Attention -> residual
       LayerNorm -> FeedForward (GELU) -> residual
```

## Files

| File | Description |
|------|-------------|
| `Bigram.py` | Bigram baseline (1st-order Markov, embedding lookup table). |
| `GPT.py` | Early transformer implementation. |
| `gpt_v2.py` | Iteration on the transformer architecture. |
| `gpt_v3.py` | Latest model: 6-layer, 6-head decoder-only Transformer with RoPE, pre-LN, GELU FFN, dropout, gradient clipping, cosine LR schedule. |
| `input.txt` | Training corpus (character-level). |
| `more.txt` | Additional text data. |
| `*.pth` | Saved model checkpoints. |
| `training_history*.png` | Training/validation loss curves. |

## Model (gpt_v3)

| Component | Detail |
|-----------|--------|
| Tokenizer | Character-level (vocab = unique chars in corpus) |
| Context length | 256 |
| Embedding dim | 384 |
| Heads | 6 |
| Layers | 6 |
| Positional encoding | Rotary Position Embedding (RoPE) |
| Normalization | Pre-LayerNorm |
| FeedForward | Linear → GELU → Linear, 4x expansion, dropout 0.1 |
| Masking | Causal (upper-triangular) |
| Optimizer | AdamW, lr 2e-3, cosine annealing |
| Regularization | Dropout + gradient clipping (max_norm 1.0) |

## Requirements

- Python 3.10+
- PyTorch
- matplotlib
- numpy

```bash
pip install torch matplotlib numpy
```

Device is auto-selected: CUDA → MPS (Apple Silicon) → CPU.

## Usage

Train the latest model:

```bash
python gpt_v3.py
```

This trains for `max_iters` steps, prints train/val loss at each `eval_interval`, generates a 300-character sample, and saves a loss curve to `training_history_transformer_v3.png`.

Train the bigram baseline:

```bash
python Bigram.py
```

## License

MIT
