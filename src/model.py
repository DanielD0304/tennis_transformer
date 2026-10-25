import os

import torch
import torch.nn as nn
from .transformer import Transformer

class Model(nn.Module):
    """
    Transformer encoder for sequence prediction tasks.
    Includes positional encoding, stacked encoder layers, and output head.
    """
    def __init__(self, d_model, num_heads, num_layers, input_dim=6, max_len=50, output_dim=2, extra_dim=0):
        super(Model, self).__init__()
        self.transformer = Transformer(d_model, num_heads, num_layers, input_dim, max_len, output_dim, extra_dim)
        
        
    def forward(self, x, positions, segments, mask=None, extra=None):
        output, _ = self.transformer(x, positions, segments, mask, extra)
        return output

    def get_attention_weights(self, x, positions, segments, mask=None, extra=None):
        _, attn_weights = self.transformer(x, positions, segments, mask, extra)
        return attn_weights


def save_checkpoint(model, path, temperature=1.0, extra_dim=0, input_dim=None):
    """Gewichte plus Temperatur (Temperature Scaling) und Input-Dimensionen in einer Datei."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'temperature': float(temperature), 'extra_dim': extra_dim,
                'input_dim': input_dim}, path)


def load_model(path, config, device='cpu'):
    """
    -> (Model im eval-Modus, Temperatur). Liest auch alte Checkpoints (nur state_dict, ohne
    Elo2-Input, Temperatur 1).
    """
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        weights, temperature, extra_dim = state['state_dict'], state.get('temperature', 1.0), state.get('extra_dim', 0)
    else:
        weights, temperature, extra_dim = state, 1.0, 0
    model = Model(config.d_model, config.num_heads, config.num_layers, config.input_dim, config.max_len,
                  config.output_dim, extra_dim).to(device)
    model.load_state_dict(weights)
    model.eval()
    return model, temperature
