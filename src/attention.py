import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention.
    Q, K, V: [batch, heads, seq, d_k] or [batch, seq, d_k]
    mask: optional, same shape as scores
    Returns: (output, attention weights)
    """
    d_k = Q.shape[-1]
    k_T = K.transpose(-2, -1)
    scores = torch.matmul(Q, k_T) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer.
    Splits input into multiple attention heads.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        """x: [batch, seq, d_model], mask: optional [batch, seq, seq]
        Returns: (output, attention weights)
        """
        batch_size = x.size(0)
        
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, -1, self.num_heads, self.d_k)
        q = q.transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k)
        k = k.transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k)
        v = v.transpose(1, 2)
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            elif mask.dim() == 2:
                # If mask is (batch, seq_len), expand to (batch, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
                mask = mask.expand(-1, 1, mask.size(-1), mask.size(-1))
        out, weights = scaled_dot_product_attention(q, k, v, mask)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(out), weights