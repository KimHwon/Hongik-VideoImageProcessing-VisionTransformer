import torch
from torch import nn

import math


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention
    """
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # stack weight 
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -1e10)

        attention = torch.nn.functional.softmax(attn_logits, dim=-1)
        
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # separate Q, K, V 
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, self.head_dim * 3)
        qkv = qkv.permute(0, 2, 1, 3) # (batch, head, seq_len, dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # determine value output
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [batch, seq_len, Head, dim]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        output = self.o_proj(values)

        if return_attention:
            return output, attention
        else:
            return output

class MLP(nn.Sequential):
    """
    MLP
    """
    def __init__(self, emb_dim, ff_dim):
        super().__init__(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, emb_dim)
        )

class ResidualAdd(nn.Module):
    """
    ResidualAdd
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        res = x
        x = self.fn(x)
        x += res
        return x

class EncoderBlock(nn.Sequential):
    """
    EncoderBlock
    """
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(dim),
                MultiHeadAttention(dim, dim, num_heads),
                nn.Dropout(dropout)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(dim),
                MLP(dim, ff_dim),
                nn.Dropout(dropout)
            ))
        )

class Transformer(nn.Sequential):
    """
    Transformer
    """
    def __init__(self, depth, dim, num_heads, ff_dim, dropout):
        super().__init__(*[
            EncoderBlock(dim, num_heads, ff_dim, dropout) for _ in range(depth)
        ])
        
