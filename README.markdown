# Transformer from Scratch

基于 PyTorch 从零实现 **Attention Is All You Need** 论文中的 Transformer 模型，附带逐行中文注释与交互式分析 Notebook。

## 项目概述

本项目旨在帮助开发者**深入理解 Transformer 的每一个实现细节**，而非仅仅调用高层 API。所有核心组件均从基础张量运算构建，并配有：

- **逐行中文注释**：解释每一行代码的数学原理与工程考量
- **交互式 Notebook**：包含可视化图表与可逐步执行的代码单元格
- **完整训练流程**：从数据生成到模型评估的端到端演示

## 核心功能

| 模块 | 类/函数 | 说明 |
|------|---------|------|
| 注意力机制 | `attention()` | 缩放点积注意力（Scaled Dot-Product Attention） |
| 多头注意力 | `MultiHeadedAttention` | 并行多头注意力 + 线性投影拼接 |
| 位置编码 | `PositionalEncoding` | 正弦/余弦位置编码（对数空间优化） |
| 前馈网络 | `PositionwiseFeedForward` | 逐位置两层 MLP（升维 → ReLU → 降维） |
| 层归一化 | `LayerNorm` | 可学习缩放/偏移的层归一化 |
| 残差连接 | `SublayerConnection` | Pre-Norm 残差连接（LayerNorm → Sublayer → Add） |
| 编码器 | `Encoder` / `EncoderLayer` | N 层自注意力 + FFN 堆叠 |
| 解码器 | `Decoder` / `DecoderLayer` | N 层掩码自注意力 + 交叉注意力 + FFN 堆叠 |
| 完整模型 | `EncoderDecoder` | 编码器-解码器架构整合 |
| 模型构建 | `make_model()` | 一键构建 + Xavier 参数初始化 |
| 贪心解码 | `greedy_decode()` | 自回归逐步生成 |

## 技术栈

- **Python** 3.10+
- **PyTorch** 2.x
- **Matplotlib** / **NumPy**（Notebook 可视化）
- **Jupyter Notebook**（交互式分析）

## 快速开始

### 安装依赖

```bash
pip install torch matplotlib numpy jupyter
```

### 运行基础模型

```bash
python Transformer_basic.py
```

预期输出：

```
==================================================
Transformer Basic Model
==================================================

[1/4] Building the model...
Number of parameters: 14731787

[2/4] Testing forward pass...
Input shape: torch.Size([1, 10])
Output shape: torch.Size([1, 10, 512])

[3/4] Testing positional encoding...
PE output shape: torch.Size([1, 100, 20])
PE range: [-1.8403, 1.8403]

[4/4] Testing greedy decode (untrained)...
Input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Output: [1, 10, 10, 10, 10, 10, 10, 10, 10, 10]
(Random output expected since model is untrained)

==================================================
All tests completed successfully!
==================================================
```

### 启动交互式 Notebook

```bash
jupyter notebook Transformer_analysis.ipynb
```

Notebook 包含 13 个章节、64 个单元格，涵盖模型实现、结构可视化、训练演示与性能评估。

## 项目结构

```
Transformer_from_scratch/
├── Transformer_basic.py          # 核心实现：完整 Transformer 模型代码
├── Transformer_analysis.ipynb    # 交互式分析 Notebook（含可视化）
├── Transformer_full_structure.md # 逐模块中文详解文档
├── assets/                       # 架构图与截图
│   ├── Snipaste_2026-04-22_19-02-51-1776896980586-1.png
│   ├── Snipaste_2026-04-24_23-37-26.png
│   └── Snipaste_2026-04-25_01-27-53.png
└── README.markdown               # 项目说明文档
```

### 文件说明

- **`Transformer_basic.py`**：完整 Transformer 实现，包含模型定义、前向传播测试与贪心解码验证。可直接运行。
- **`Transformer_analysis.ipynb`**：13 章交互式分析，包括位置编码热力图、因果掩码可视化、架构示意图、多头注意力权重图、Noam 学习率曲线、训练损失曲线等。
- **`Transformer_full_structure.md`**：面向初学者的逐模块深度解析，涵盖数学公式推导、代码逐行注释与设计原理阐述。

## 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N` | 6 | 编码器/解码器堆叠层数 |
| `d_model` | 512 | 模型隐藏维度 |
| `d_ff` | 2048 | 前馈网络中间层维度 |
| `h` | 8 | 多头注意力头数 |
| `d_k` | 64 | 每个头的维度（= d_model / h） |
| `dropout` | 0.1 | Dropout 比率 |

不同配置下的参数量对比：

| 配置 | N | d_model | d_ff | h | 参数量 |
|------|---|---------|------|---|--------|
| Tiny | 1 | 64 | 256 | 2 | ~40K |
| Small | 2 | 128 | 512 | 4 | ~500K |
| Base | 6 | 512 | 2048 | 8 | ~59.5M |
| Big | 6 | 1024 | 4096 | 16 | ~268.5M |

## 关键设计说明

### Pre-Norm vs Post-Norm

本实现采用 **Pre-Norm**（先归一化再经过子层）：

```
Pre-Norm:  output = x + Sublayer(LayerNorm(x))
Post-Norm: output = LayerNorm(x + Sublayer(x))
```

Pre-Norm 为原始输入提供无阻碍的梯度通道，训练更稳定，是 GPT、LLaMA 等现代大模型的标配。

### Xavier 初始化

`make_model()` 中对所有维度大于 1 的参数使用 Xavier 均匀初始化，确保各层输出的方差大致相等，有助于深层网络的稳定训练。

### log_softmax

注意力计算中使用 `log_softmax` 而非 `softmax`，在某些场景下数值更稳定（避免浮点溢出），但会影响注意力权重的概率解释。

## Notebook 章节导航

| 章节 | 内容 | 可视化 |
|------|------|--------|
| 1 | 环境准备与工具函数 | — |
| 2 | 层归一化 & 残差连接 | LayerNorm 效果演示 |
| 3 | 注意力机制 | — |
| 4 | 前馈网络 & 嵌入层 | — |
| 5 | 位置编码 | 热力图 + 多维度波形图 |
| 6 | 编码器 | — |
| 7 | 解码器 | 因果掩码矩阵图 |
| 8 | 完整模型组装 | — |
| 9 | 模型结构可视化 | 架构示意图 + 注意力权重热力图 |
| 10 | 关键参数说明 | 配置对比表 |
| 11 | 训练过程分析 | Noam 学习率曲线 + 训练损失曲线 |
| 12 | 模型性能评估 | 训练后注意力权重对比 |
| 13 | 综合测试与总结 | — |

## 参考文献

- Vaswani, A., et al. **"Attention Is All You Need"**. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Harvard NLP. **"The Annotated Transformer"**. [nlp.seas.harvard.edu](https://nlp.seas.harvard.edu/annotated-transformer/)

## 许可证

MIT License
