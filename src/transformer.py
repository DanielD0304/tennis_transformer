import torch
import torch.nn as nn
from .encoderlayer import EncoderLayer

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.encoder_layers = nn.ModuleList([
    EncoderLayer(d_model, num_heads) for _ in range(num_layers)
])