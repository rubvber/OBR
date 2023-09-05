import torch
from torch import nn
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.

    Code due to Giorgia Corrado with slight modifications from Ruben van Bergen
    """
    def __init__(self, dim: int, d_k=64):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_d_k = math.sqrt(d_k)
        self.wq, self.wk, self.wv = [nn.Linear(dim,d_k) for _ in range(3)]
  
    def forward(self, x):

        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_d_k
        
        attn = score.softmax(-1)
        context = torch.bmm(attn, value)
        return context, attn
    

class MultiHeadAttention(nn.Module):
    # Code added by Ruben van Bergen
    def __init__(self, in_dim, d_k=64, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.sqrt_d_k = math.sqrt(d_k)
        self.num_heads = num_heads
        self.d_k = d_k
        self.wq, self.wk, self.wv = [nn.Linear(in_dim,d_k*num_heads) for _ in range(3)]

    def forward(self, x):
        N,K = x.shape[:2]
        s = [N, K, self.d_k, self.num_heads]
        query = self.wq(x).view(s)
        key = self.wk(x).view(s)
        value = self.wv(x).view(s)

        score = (query.unsqueeze(2)*key.unsqueeze(1)).sum(3) / self.sqrt_d_k
        
        attn = score.softmax(-2)

        context = ((attn.unsqueeze(-2) * value.unsqueeze(2)).sum(2)).view(N,K,self.d_k*self.num_heads)
        
        return context, attn