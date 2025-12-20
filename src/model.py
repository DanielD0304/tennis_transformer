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
        self.softmax = nn.Softmax(dim = -1)
        
        
    def forward(self, x, positions, mask=None):
        output, _ = self.transformer(x, positions, mask)
        output_pred = self.softmax(output)
        return output_pred
    
    def get_attention_weights(self, x, positions, mask=None):
        _, attn_weights = self.transformer(x, positions, mask)
        return attn_weights