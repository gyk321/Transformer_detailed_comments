import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import math
import copy

def clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
  def __init__(self, features, eps=1e-6):
    super(LayerNorm, self).__init__()
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
  
def attention(query, key, value, mask=None, dropout=None):
  d_k = query.size(-1)
  scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
  p_attn = log_softmax(scores, dim=-1)
  if dropout is not None:
    p_attn = dropout(p_attn)
  return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
  def __init__(self, h, d_model, dropout=0.1):
    super(MultiHeadedAttention, self).__init__()
    assert d_model % h == 0
    self.d_k = d_model // h
    self.h = h
    self.linears = clones(nn.Linear(d_model, d_model), 4)
    self.attn = None
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, query, key, value, mask=None):
    if mask is not None:
      mask = mask.unsqueeze(1)
    nbatches = query.size(0)
    query, key, value = [l(x)
      .view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
      for l, x in zip(self.linears, (query, key, value))]
    x, self.attn = attention(
      query, 
      key, 
      value, 
      mask=mask, 
      dropout=self.dropout)
    x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
    return self.linears[-1](x)
  
class PositionwiseFeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.w_1 = nn.Linear(d_model, d_ff)
    self.w_2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.w_2(self.dropout(torch.relu(self.w_1(x))))
  
class Embeddings(nn.Module):
  def __init__(self, d_model, vocab):
    """
    初始化词嵌入层
    :param d_model: 词嵌入的维度
    :param vocab: 词汇表的大小
    """
    super(Embeddings, self).__init__()
    self.lut = nn.Embedding(vocab, d_model)
    self.d_model = d_model

  def forward(self, x):
    return self.lut(x) * math.sqrt(self.d_model)
  
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:, :x.size(1)].requires_grad_(False)
    return self.dropout(x)

def subsequent_mask(size):
  attn_shape = (1, size, size)
  subsequent_mask = torch.triu(
    torch.ones(attn_shape), diagonal=1).type(torch.uint8)
  return subsequent_mask == 0

class SublayerConnection(nn.Module):
  def __init__(self, size, dropout):
    super(SublayerConnection, self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))
  
class Encoder(nn.Module):
  def __init__(self, layer, N):
    super(Encoder, self).__init__()
    self.layers = clones(layer, N)
    self.norm = LayerNorm(layer.size)

  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)
  
class EncoderLayer(nn.Module):
  def __init__(self, size, self_attn, feed_forward, dropout):
    super(EncoderLayer, self).__init__()
    self.self_attn = self_attn
    self.feed_forward = feed_forward
    self.sublayer = clones(SublayerConnection
                            (size, dropout), 
                            2)
    self.size = size

  def forward(self, x, mask):
    x = self.sublayer[0](x,
                         lambda x: 
                         self.self_attn(x, x, x, mask))
    return self.sublayer[1](x, self.feed_forward)
  
class Decoder(nn.Module):
  def __init__(self, layer, N):
    super(Decoder, self).__init__()
    self.layers = clones(layer, N)
    self.norm = LayerNorm(layer.size)

  def forward(self, x, memory, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x, memory, src_mask, tgt_mask)
    return self.norm(x)
  
class DecoderLayer(nn.Module):
  def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
    super(DecoderLayer, self).__init__()
    self.size = size
    self.self_attn = self_attn
    self.src_attn = src_attn
    self.feed_forward = feed_forward
    self.sublayer = clones(SublayerConnection(size,
                                              dropout), 3)
  def forward(self, x, memory, src_mask, tgt_mask):
    m = memory
    x = self.sublayer[0](x,
                         lambda x: 
                         self.self_attn(x, x, x, tgt_mask))
    x = self.sublayer[1](x,
                         lambda x: 
                         self.src_attn(x, m, m, src_mask))
    return self.sublayer[2](x, self.feed_forward)
  
class EncoderDecoder(nn.Module):
  def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
    super(EncoderDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.generator = generator
  def forward(self, src, tgt, src_mask, tgt_mask):
    return self.decode(self.encode(src, src_mask),
                       src_mask, 
                       tgt, 
                       tgt_mask)
  def encode(self, src, src_mask):
    return self.encoder(self.src_embed(src), 
                        src_mask)
  def decode(self, memory, src_mask, tgt, tgt_mask):
    return self.decoder(self.tgt_embed(tgt), 
                        memory, 
                        src_mask, 
                        tgt_mask)
  
class Generator(nn.Module):
  def __init__(self, d_model, vocab):
    super(Generator, self).__init__()
    self.proj = nn.Linear(d_model, vocab)

  def forward(self, x):
    return log_softmax(self.proj(x), dim=-1)
  
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
  c = copy.deepcopy
  attn = MultiHeadedAttention(h, d_model)
  ff = PositionwiseFeedForward(d_model, d_ff, dropout)
  position = PositionalEncoding(d_model, dropout)
  model = EncoderDecoder(
    Encoder(EncoderLayer(d_model, 
                   c(attn), 
                   c(ff), 
                   dropout), 
                N),
    Decoder(DecoderLayer(d_model, 
                   c(attn), 
                   c(attn), 
                   c(ff), 
                   dropout), 
                N),
    nn.Sequential(Embeddings(d_model, src_vocab), 
                  c(position)),
    nn.Sequential(Embeddings(d_model, tgt_vocab), 
                  c(position)),
    Generator(d_model, tgt_vocab)
                )
  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  return model

print("=" * 50)
print("Transformer Basic Model")
print("=" * 50)

print("\n[1/4] Building the model...")
model = make_model(11, 11, N=2)
params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {params}")

print("\n[2/4] Testing forward pass...")
src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
tgt = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
src_mask = torch.ones(1, 1, 10)
tgt_mask = subsequent_mask(10)
output = model(src, tgt, src_mask, tgt_mask)
print("Input shape:", src.shape)
print("Output shape:", output.shape)

print("\n[3/4] Testing positional encoding...")
pe = PositionalEncoding(20, 0)
y = pe.forward(torch.zeros(1, 100, 20))
print("PE output shape:", y.shape)
print(f"PE range: [{y.min():.4f}, {y.max():.4f}]")

print("\n[4/4] Testing greedy decode (untrained)...")
def greedy_decode(model, src, src_mask, max_len, start_symbol):
  memory = model.encode(src, src_mask)
  ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
  for i in range(max_len-1):
    out = model.decode(memory, src_mask, 
                       ys, 
                       subsequent_mask(ys.size(1)).type_as(src.data))
    prob = model.generator(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word.data[0]
    ys = torch.cat([ys, 
                    torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
  return ys

test_model = make_model(11, 11, N=2)
test_model.eval()
result = greedy_decode(test_model, 
                       src, src_mask, 
                       max_len=10, 
                       start_symbol=1)
print("Input:", src.tolist()[0])
print("Output:", result.tolist()[0])
print("(Random output expected since model is untrained)")

print("\n" + "=" * 50)
print("All tests completed successfully!")
print("=" * 50)