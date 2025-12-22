import torch
import torch.nn as nn
from .encoderlayer import EncoderLayer

class Transformer(nn.Module):
    """
    Transformer encoder for sequence prediction tasks.
    Includes positional encoding, stacked encoder layers, and output head.
    """
    def __init__(self, d_model, num_heads, num_layers, max_len=15, output_dim=2):
        super(Transformer, self).__init__()
        self.input_proj = nn.Linear(5, d_model)
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_len = max_len
        self.output_dim = output_dim
        self.encoder_layers = nn.ModuleList([
    EncoderLayer(d_model, num_heads) for _ in range(num_layers)
])
        self.dropout = nn.Dropout(0.1)
        self.positionalEncoding = nn.Embedding(max_len, d_model)
        self.output = nn.Linear(d_model, output_dim)
        
    def forward(self, x, positions, mask=None):
        """
        Forward pass for the Transformer encoder.
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            positions: Position indices tensor of shape [batch, seq_len]
            mask: Optional attention mask [batch, seq_len, seq_len]
        Returns:
            Output tensor of shape [batch, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        assert positions.shape == (batch_size, seq_len)
        pos_emb = self.positionalEncoding(positions)
        x = self.input_proj(x)
        x = x + pos_emb
        x = self.dropout(x)
        all_attn_weights = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)
        x = x.mean(dim=1)
        x = self.output(x)
        return x, all_attn_weights