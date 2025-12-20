import torch
import preprocessing as pre
import data_loader as dl

class DataSet(torch.utils.data.Dataset):
    def __init__(self):
        self.data = dl.load_all_matches()
        self.processed_data = pre.create_training_samples(self.data)
        
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, index):
        sample = self.processed_data[index]
        # Player A
        player_a_surface = torch.tensor([list(d.values()) for d in sample['player_a_surface']], dtype=torch.float)
        player_a_surface_mask = torch.tensor(sample['player_a_surface_mask'], dtype=torch.float)
        player_a_surface_pos = torch.arange(player_a_surface.shape[0])
        player_a_recent = torch.tensor([list(d.values()) for d in sample['player_a_recent']], dtype=torch.float)
        player_a_recent_mask = torch.tensor(sample['player_a_recent_mask'], dtype=torch.float)
        player_a_recent_pos = torch.arange(player_a_recent.shape[0])
        # Player B
        player_b_surface = torch.tensor([list(d.values()) for d in sample['player_b_surface']], dtype=torch.float)
        player_b_surface_mask = torch.tensor(sample['player_b_surface_mask'], dtype=torch.float)
        player_b_surface_pos = torch.arange(player_b_surface.shape[0])
        player_b_recent = torch.tensor([list(d.values()) for d in sample['player_b_recent']], dtype=torch.float)
        player_b_recent_mask = torch.tensor(sample['player_b_recent_mask'], dtype=torch.float)
        player_b_recent_pos = torch.arange(player_b_recent.shape[0])
        # Label
        label = torch.tensor(sample['label'], dtype=torch.long)
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
            'label': label
        }
        
        