import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    k_T = K.transpose(-2, -1)
    scores = torch.matmul(Q, k_T) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights
