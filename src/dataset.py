import torch
from . import preprocessing as pre
from . import data_loader as dl
from .preprocessing import FEATURE_NAMES

class DataSet(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        def feature_list(d):
            return [d.get(k, 0.0) for k in FEATURE_NAMES]

        # Player A
        player_a_surface = torch.tensor([feature_list(d) for d in sample['player_a_surface']], dtype=torch.float)
        player_a_surface_mask = torch.tensor(sample['player_a_surface_mask'], dtype=torch.float)
        player_a_surface_pos = torch.arange(player_a_surface.shape[0])
        
        player_a_recent = torch.tensor([feature_list(d) for d in sample['player_a_recent']], dtype=torch.float)
        player_a_recent_mask = torch.tensor(sample['player_a_recent_mask'], dtype=torch.float)
        player_a_recent_pos = torch.arange(player_a_recent.shape[0])

        # Player B
        player_b_surface = torch.tensor([feature_list(d) for d in sample['player_b_surface']], dtype=torch.float)
        player_b_surface_mask = torch.tensor(sample['player_b_surface_mask'], dtype=torch.float)
        player_b_surface_pos = torch.arange(player_b_surface.shape[0])
        
        player_b_recent = torch.tensor([feature_list(d) for d in sample['player_b_recent']], dtype=torch.float)
        player_b_recent_mask = torch.tensor(sample['player_b_recent_mask'], dtype=torch.float)
        player_b_recent_pos = torch.arange(player_b_recent.shape[0])

        segment_ids = torch.tensor(sample['segment_ids'], dtype=torch.long)
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        odds_a = torch.tensor(sample.get('odds_a', 0.0), dtype=torch.float)
        odds_b = torch.tensor(sample.get('odds_b', 0.0), dtype=torch.float)

        return {
            'player_a_surface': player_a_surface,
            'player_a_surface_mask': player_a_surface_mask,
            'player_a_surface_pos': player_a_surface_pos,
            'player_a_recent': player_a_recent,
            'player_a_recent_mask': player_a_recent_mask,
            'player_a_recent_pos': player_a_recent_pos,
            'player_b_surface': player_b_surface,
            'player_b_surface_mask': player_b_surface_mask,
            'player_b_surface_pos': player_b_surface_pos,
            'player_b_recent': player_b_recent,
            'player_b_recent_mask': player_b_recent_mask,
            'player_b_recent_pos': player_b_recent_pos,
            'segment_ids': segment_ids,
            'label': label,
            'odds_a': odds_a,
            'odds_b': odds_b
        }
        
        