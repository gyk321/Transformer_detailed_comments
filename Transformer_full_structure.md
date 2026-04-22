# Transformer全结构

![Snipaste_2026-04-22_19-02-51](C:\Users\10540\Desktop\Transformer_from_scratch\assets\Snipaste_2026-04-22_19-02-51-1776896980586-1.png)

## make_model 函数详解

```python
def make_model(
    src_vocab,    # 源语言词汇表大小
    tgt_vocab,    # 目标语言词汇表大小
    N=6,          # Encoder 和 Decoder 的层数
    d_model=512,  # 模型隐藏维度
    d_ff=2048,    # 前馈网络的中间维度
    h=8,          # 注意力头数
    dropout=0.1   # Dropout 概率
):
    """
    构建完整的 Transformer 模型
    
    参数:
        src_vocab: 源语言词汇表大小
        tgt_vocab: 目标语言词汇表大小
        N: Encoder/Decoder 的层数（论文中是 6）
        d_model: 模型隐藏维度（论文中是 512）
        d_ff: 前馈网络中间层维度（论文中是 2048）
        h: 多头注意力头数（论文中是 8）
        dropout: Dropout 概率
    
    返回:
        完整的 EncoderDecoder 模型
    """
    
    # ============================================
    # 步骤 1: 准备基础组件
    # ============================================
    
    # 定义 deepcopy 的简写，后面会频繁用到
    c = copy.deepcopy
    
    # 创建多头注意力模块
    # 这个模块会被多次复制使用
    attn = MultiHeadedAttention(h, d_model)
    
    # 创建前馈网络模块
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    # 创建位置编码模块
    position = PositionalEncoding(d_model, dropout)
    
    # ============================================
    # 步骤 2: 组装完整的 Transformer 模型
    # ============================================
    
    model = EncoderDecoder(
        # --- Encoder ---
        # Encoder 由 N 个 EncoderLayer 层组成
        Encoder(
            EncoderLayer(
                d_model,        # 模型维度
                c(attn),        # 自注意力（深拷贝！）
                c(ff),          # 前馈网络（深拷贝！）
                dropout         # Dropout
            ),
            N  # Encoder 层数
        ),
        
        # --- Decoder ---
        # Decoder 由 N 个 DecoderLayer 层组成
        Decoder(
            DecoderLayer(
                d_model,        # 模型维度
                c(attn),        # 掩码自注意力（深拷贝！）
                c(attn),        # 源-目标注意力（深拷贝！）
                c(ff),          # 前馈网络（深拷贝！）
                dropout         # Dropout
            ),
            N  # Decoder 层数
        ),
        
        # --- 源语言嵌入 ---
        # 词嵌入 + 位置编码
        nn.Sequential(
            Embeddings(d_model, src_vocab),  # 词嵌入
            c(position)                      # 位置编码（深拷贝！）
        ),
        
        # --- 目标语言嵌入 ---
        # 词嵌入 + 位置编码
        nn.Sequential(
            Embeddings(d_model, tgt_vocab),  # 词嵌入
            c(position)                      # 位置编码（深拷贝！）
        ),
        
        # --- Generator ---
        # 将最后输出映射回词汇表概率分布
        Generator(d_model, tgt_vocab)
    )
    
    # ============================================
    # 步骤 3: 初始化参数
    # ============================================
    
    # 对所有参数进行 Xavier 初始化（论文中的做法）
    for p in model.parameters():
        if p.dim() > 1:  # 只初始化矩阵参数，偏置不初始化
            nn.init.xavier_uniform_(p)
    
    # 返回完整模型
    return model
```

## 词嵌入层（Word Embeddings Layer）

将离散的单词（通常表示为整数 ID）转换为模型可以处理的连续且具有固定维度的稠密向量，并对向量进行特定的缩放。

```python
class Embeddings(nn.Module):
    """词嵌入层：将离散的 token ID 转换为连续的向量表示"""
    
    def __init__(self, d_model, vocab):
        """
        初始化词嵌入层
        
        参数:
            d_model: 模型的隐藏维度 (论文中为 512)
                     每个 token 会被映射为一个 d_model 维的向量
            vocab: 词汇表大小 (例如 37000)
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)  # Look-Up Table 查找表
        self.d_model = d_model

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: token ID 序列，形状为 (batch_size, sequence_length)
               例如: [[1, 7, 3, 9], [4, 2, 8, 5]]
        
        返回:
            嵌入向量，形状为 (batch_size, sequence_length, d_model)
            每个 token ID 被替换为对应的 d_model 维向量
        """
        # 1. 查表：将每个 token ID 替换为对应的嵌入向量
        # 2. 缩放：乘以 √(d_model) 进行数值缩放
        return self.lut(x) * math.sqrt(self.d_model)
```

## Transformer 位置编码 (Positional Encoding) 详解

### 一、 核心数学公式

Transformer 论文中使用了不同频率的正弦和余弦函数来生成位置编码：

$PE_{(pos, 2i)} = \sin(pos / 10000^{2i / d_{\text{model}}})$

$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i / d_{\text{model}}})$

- $pos$：当前词在句子中的绝对位置（例如第 0 个词，第 1 个词）。
- $i$：词向量的维度索引（从 $0$ 到 $d_{\text{model}}/2$）。
- $d_{\text{model}}$：词向量的总维度（代码中通常是 512）。

这个公式的精妙之处在于，对于任何固定的偏移量 $k$，$PE_{pos+k}$ 都可以表示为 $PE_{pos}$ 的线性函数，这有助于模型学习词与词之间的**相对位置关系**。

------

### 二、 代码逐行解析

这份代码位于你上传的文件的 `PositionalEncoding` 类中。为了兼顾计算效率，这段 PyTorch 代码并没有直接使用上述公式进行 `for` 循环，而是使用了一些矩阵运算的技巧。

#### 1. 初始化部分 (`__init__`)

```Python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
```

- `d_model`: 模型的维度（如 512）。
- `dropout`: 丢弃率，用于防止过拟合。
- `max_len`: 预设的序列最大长度，默认 5000 足够绝大多数 NLP 任务使用。我们会在初始化时一次性计算好这 5000 个位置的编码，避免每次前向传播时重复计算。

#### 2. 核心张量运算 (对数空间优化)

```Python
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
```

- `pe`: 创建一个大小为 `(5000, 512)` 的全 0 矩阵，用来存放所有位置的编码。
- `position`: 创建一个从 0 到 4999 的一维张量，然后用 `unsqueeze(1)` 把它变成形状为 `(5000, 1)` 的列向量，代表绝对位置 $pos$。

```Python
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
```

- **这是整段代码最难懂的地方：对数空间运算。**

  代码没有直接计算 $\frac{1}{10000^{2i / d_{\text{model}}}}$，因为在计算机中直接计算大指数的分数容易出现数值不稳定或溢出。

  根据数学对数性质：

  $\frac{1}{10000^{2i/d}} = 10000^{-2i/d} = e^{\ln(10000^{-2i/d})} = e^{-\frac{2i}{d} \ln(10000)}$

  对应到代码里：

  - `torch.arange(0, d_model, 2)` 生成了数列 `[0, 2, 4, ...]`，这就对应了公式里的 $2i$。
  - `-(math.log(10000.0) / d_model)` 对应了 $-\frac{\ln(10000)}{d}$。
  - 外面套上 `torch.exp` 即可得到分母的除数因子 `div_term`。

```Python
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
```

- `0::2` 表示从索引 0 开始，步长为 2（即 0, 2, 4, 6... 的偶数列）。将这些列赋值为 $\sin$ 函数的值。
- `1::2` 表示从索引 1 开始，步长为 2（即 1, 3, 5, 7... 的奇数列）。将这些列赋值为 $\cos$ 函数的值。
- 这里利用了 PyTorch 的广播机制（Broadcasting），`position` 矩阵 `(5000, 1)` 乘以 `div_term` 矩阵 `(256,)`，会自动生成 `(5000, 256)` 的结果矩阵。

#### 3. 注册为 Buffer (`register_buffer`)

```Python
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
```

- `unsqueeze(0)`：在最外层增加一个 Batch 维度，将形状变成 `(1, 5000, 512)`，以便后续与输入张量（Batch Size, Seq Length, d_model）直接相加。
- `self.register_buffer`：这是 PyTorch 的一个重要特性。位置编码 `pe` 是固定的（不可训练的参数），但我们需要它和模型一起保存在 `state_dict` 中，且能够在调用 `.cuda()` 时自动移动到 GPU 上。使用 `register_buffer` 可以完美实现这一点。

#### 4. 前向传播 (`forward`)

```Python
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
```

- `x`: 模型的输入，形状通常是 `(batch_size, sequence_length, d_model)`。
- `self.pe[:, : x.size(1)]`: 因为我们预先计算了 5000 个位置，这里根据当前输入的实际长度 `x.size(1)` 将多余的切片截断掉。
- `requires_grad_(False)`: 再次确保反向传播时不会对位置编码求梯度（节省显存和算力）。
- **最后一步：** 将词嵌入表示 `x` 与位置编码 `pe` **直接相加**（注意是相加，不是拼接），然后通过 Dropout 层返回。

## 多头注意力机制

多头注意力机制（Multi-Head Attention）是 Transformer 能够如此强大的**绝对核心**。

如果在上一步骤中，位置编码是为了让模型知道词的“位置”，那么多头注意力机制就是为了让模型理解词与词之间的**“多重复杂关系”**。

### 一、 为什么需要“多头”？

想象你在阅读句子 **"The animal didn't cross the street because it was too tired."** 当你试图理解 "it" 指代什么时：

- **注意力头 A** 可能关注语法结构（寻找最近的名词 "animal"）。
- **注意力头 B** 可能关注语义逻辑（谁会 "tired"？也是 "animal"）。
- **注意力头 C** 可能关注行为关联（谁在 "cross the street"？）。

如果只有一个注意力头，所有的信息融合在一起容易变得模糊。**多头机制允许模型将数据映射到不同的子空间中，并行地关注不同维度的特征。**

------

### 二、 核心数学公式

首先，标准缩放点积注意力（Scaled Dot-Product Attention）的公式是：

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

而多头注意力的做法是，将 $Q, K, V$ 分别通过不同的线性变换映射 $h$ 次（$h$ 为头的数量），并行计算上述注意力，最后将结果拼接（Concat）并再次线性映射：

$$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O$$

其中

$$\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

------

### 三、 代码逐段深度解析

这份代码将原本复杂的“并行映射”转化为了极其优雅的**矩阵维度变换（Reshape / View）技巧**。这也是 PyTorch 工程实现中最精彩的部分之一。

#### 1. 初始化阶段 (`__init__`)

Python

```
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
```

- **`assert d_model % h == 0`**: 模型的总维度（通常是 512）必须能被头数（通常是 8）整除。这样每个头分到的维度就是 $512 / 8 = 64$ 维（即 `self.d_k`）。

- **`self.linears = clones(nn.Linear(d_model, d_model), 4)`**: **（高能预警）** 按照理论公式，我们需要为 8 个头分别建立 $W^Q, W^K, W^V$ 的权重矩阵。但在工程上那样写非常慢！

  这里的聪明做法是：直接克隆 4 个尺寸为 `(d_model, d_model)` 的全连接层。前 3 个用来同时处理所有头的 $Q, K, V$ 的线性映射，第 4 个用来做最后的输出映射 $W^O$。

#### 2. 前向传播 (`forward`)

这是数据真正流动的过程。假设输入的形状是 `(batch_size, seq_len, d_model)`。

**步骤 1：线性映射与维度拆分 (The Reshape Magic)**

Python

```
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
```

这是全篇最难懂的一行代码，我们拆开来看：

1. `lin(x)`: 数据先经过全连接层，形状仍为 `(batch_size, seq_len, 512)`。
2. `.view(nbatches, -1, self.h, self.d_k)`: 将最后的 512 维拆开，变成 `(batch_size, seq_len, 8, 64)`。这标志着我们将词向量切分给了 8 个不同的头。
3. `.transpose(1, 2)`: 交换第 1 和第 2 维度。形状变成了 **`(batch_size, 8, seq_len, 64)`**。
   - **为什么要交换？** 因为我们要让第 8 个头（维度索引为 1）表现得像批次（Batch）一样，这样在下一步计算矩阵乘法时，PyTorch 会自动在 8 个头上**并行独立计算**注意力得分！

**步骤 2：并行计算注意力**

Python

```
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
```

将形状为 `(batch_size, 8, seq_len, 64)` 的 $Q, K, V$ 送入基础的 `attention` 函数。内部通过 `torch.matmul` 会对最后两个维度进行矩阵乘法。得到的结果 `x` 形状依然是 `(batch_size, 8, seq_len, 64)`。

**步骤 3：拼接头并输出**

Python

```
        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        # ... (del 释放显存操作)
        return self.linears[-1](x)
```

1. `x.transpose(1, 2)`: 先把维度换回来，变成 `(batch_size, seq_len, 8, 64)`。
2. `.contiguous()`: 内存连续化处理（PyTorch中转置后改变形状前的常规操作）。
3. `.view(nbatches, -1, self.h * self.d_k)`: 把最后两个维度重新合并成 `8 * 64 = 512`。这一步**等价于公式中的 Concat（拼接）操作**，形状变回 `(batch_size, seq_len, 512)`。
4. 最后，通过第 4 个线性层 `self.linears[-1](x)` 做最后一次信息融合输出。

### 总结

这份代码完美地展现了**“空间换时间”**与**“批处理并行”**的深度学习工程哲学。它没有写任何一个 `for` 循环去挨个处理 8 个头，而是通过精妙的 `view` 和 `transpose`，让底层 GPU 矩阵加速运算一次性搞定了多头注意力的全过程。
