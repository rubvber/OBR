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
        self.wq = nn.Linear(dim,d_k)
        self.wk = nn.Linear(dim,d_k)
        self.wv = nn.Linear(dim,d_k)

    def forward(self, x):

        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_d_k
        
        attn = score.softmax(-1)
        context = torch.bmm(attn, value)
        return context, attn