import torch
import torch.nn as nn
from .transformer import Transformer

class Model(nn.Module):
    """
    Transformer encoder for sequence prediction tasks.
    Includes positional encoding, stacked encoder layers, and output head.
    """
    def __init__(self, d_model, num_heads, num_layers):
        super(Model, self).__init__()
        self.transformer = Transformer(d_model, num_heads, num_layers)
        
        
    def forward(self, x, positions, segments, mask=None):
        output, _ = self.transformer(x, positions, segments, mask)
        return output

    def get_attention_weights(self, x, positions, segments, mask=None):
        _, attn_weights = self.transformer(x, positions, segments, mask)
        return attn_weights