# wordGPT_transformer

从零开始用 PyTorch 实现的字符级语言模型，逐步演进：从简单的 Bigram 基线模型到完整的仅解码器 Transformer（GPT）。在字符语料（`input.txt`，Tiny Shakespeare）上训练。

[English](README.md) | 中文

## 概览

本项目从底层实现一个 GPT 风格的自回归语言模型，不依赖任何高层 Transformer 库。作为学习项目，它展示了从 n-gram 建模到现代注意力架构的完整路径。

```
输入 -> token embedding -> [Transformer block] x N -> LayerNorm -> logits
block: LayerNorm -> 多头自注意力 -> 残差连接
       LayerNorm -> 前馈网络 (GELU) -> 残差连接
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `Bigram.py` | Bigram 基线模型（一阶马尔可夫，embedding 查表）。 |
| `GPT.py` | 早期 Transformer 实现。 |
| `gpt_v2.py` | Transformer 架构的迭代版本。 |
| `gpt_v3.py` | 最新模型：6 层、6 头的仅解码器 Transformer，使用 RoPE、Pre-LN、GELU 前馈网络、dropout、梯度裁剪、余弦学习率调度。 |
| `input.txt` | 训练语料（字符级）。 |
| `more.txt` | 额外文本数据。 |
| `*.pth` | 保存的模型检查点。 |
| `training_history*.png` | 训练/验证损失曲线。 |

## 模型 (gpt_v3)

| 组件 | 细节 |
|------|------|
| 分词器 | 字符级（词表 = 语料中的唯一字符） |
| 上下文长度 | 256 |
| 嵌入维度 | 384 |
| 注意力头数 | 6 |
| 层数 | 6 |
| 位置编码 | 旋转位置编码 (RoPE) |
| 归一化 | Pre-LayerNorm |
| 前馈网络 | Linear → GELU → Linear，4 倍扩展，dropout 0.1 |
| 掩码 | 因果掩码（上三角） |
| 优化器 | AdamW，学习率 2e-3，余弦退火 |
| 正则化 | Dropout + 梯度裁剪 (max_norm 1.0) |

## 环境依赖

- Python 3.10+
- PyTorch
- matplotlib
- numpy

```bash
pip install torch matplotlib numpy
```

设备自动选择：CUDA → MPS（Apple Silicon）→ CPU。

## 使用方法

训练最新模型：

```bash
python gpt_v3.py
```

将训练 `max_iters` 步，每 `eval_interval` 打印一次训练/验证损失，生成 300 字符的样本，并将损失曲线保存到 `training_history_transformer_v3.png`。

训练 Bigram 基线模型：

```bash
python Bigram.py
```

## 许可证

MIT
