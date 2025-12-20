import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    """Transformer Encoder Block: Multi-Head Attention, Feedforward, LayerNorm, Dropout, Residuals."""
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        """
        Forward pass for one encoder block.
        Args:
            x: Tensor of shape [batch, seq, d_model]
            mask: Optional attention mask [batch, seq, seq]
        Returns:
            Tensor of shape [batch, seq, d_model]
        """
        attn_out, attn_weights = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights