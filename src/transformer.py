import torch
import torch.nn as nn
from .encoderlayer import EncoderLayer

class Transformer(nn.Module):
    """
    Transformer encoder for sequence prediction tasks.
    Includes positional encoding, stacked encoder layers, and output head.
    """
    def __init__(self, d_model, num_heads, num_layers, input_dim=6, max_len=50, output_dim=2):
        super(Transformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        # 0: CLS, 1: A-Surface, 2: A-Recent, 3: B-Surface, 4: B-Recent
        self.segmentEmbedding = nn.Embedding(5, d_model)
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_len = max_len
        self.output_dim = output_dim
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.2)
        self.positionalEncoding = nn.Embedding(max_len + 1, d_model)  # +1 für CLS
        self.output = nn.Linear(d_model, output_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # [1, 1, d_model]
        
    def forward(self, x, positions, segments, mask=None):
        """
        Forward pass for the Transformer encoder mit [CLS]-Token.
        Args:
            x: Input tensor of shape [batch, seq_len, input_dim]
            positions: Position indices tensor of shape [batch, seq_len]
            mask: Optional attention mask [batch, seq_len+1, seq_len+1]
        Returns:
            Output tensor of shape [batch, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        assert positions.shape == (batch_size, seq_len)
        assert segments.shape == (batch_size, seq_len)

        # [CLS]-Token an den Anfang der Sequenz einfügen
        cls_tokens = self.cls_token.expand(batch_size, 1, self.d_model)  # [batch, 1, d_model]
        x = self.input_proj(x)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, seq_len+1, d_model]

        cls_pos = torch.zeros(batch_size, 1, dtype=positions.dtype, device=positions.device)
        positions = positions + 1
        positions = torch.cat([cls_pos, positions], dim=1)  # [batch, seq_len+1]
        pos_emb = self.positionalEncoding(positions)

        # Segment-IDs für [CLS] anpassen: 0 für CLS, Rest wie übergeben (1,2,3,4)
        cls_seg = torch.zeros(batch_size, 1, dtype=segments.dtype, device=segments.device)
        segments = torch.cat([cls_seg, segments], dim=1)  # [batch, seq_len+1]
        seg_emb = self.segmentEmbedding(segments)

        x = x + pos_emb + seg_emb
        x = self.dropout(x)
        all_attn_weights = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)
        x = x[:, 0, :]
        x = self.output(x)
        return x, all_attn_weights