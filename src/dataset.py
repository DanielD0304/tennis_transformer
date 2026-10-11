import torch
from . import preprocessing as pre
from . import data_loader as dl
from .preprocessing import FEATURE_NAMES, STAT_FEATURES

# Spalten, die beim Stats-Dropout genullt werden (inkl. has_stats)
_STAT_IDX = torch.tensor([FEATURE_NAMES.index(f) for f in STAT_FEATURES + ['has_stats']])
_HAS_STATS = FEATURE_NAMES.index('has_stats')


class DataSet(torch.utils.data.Dataset):
    def __init__(self, samples, stats_dropout=0.0):
        """
        stats_dropout: Anteil der Historien-Matches, deren Aufschlag-Stats zufällig entfernt werden.
        Nur fürs Training. Live fehlen die Stats für neuere Matches (tennis-data, ESPN),
        so lernt das Modell, damit umzugehen.
        """
        self.samples = samples
        self.stats_dropout = stats_dropout

    def _drop_stats(self, x):
        if self.stats_dropout <= 0:
            return x
        drop = (torch.rand(x.shape[0]) < self.stats_dropout) & (x[:, _HAS_STATS] > 0)
        x[drop.nonzero(as_tuple=True)[0].unsqueeze(1), _STAT_IDX] = 0.0
        return x
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        def feature_list(d):
            return [d.get(k, 0.0) for k in FEATURE_NAMES]

        # Player A
        player_a_surface = self._drop_stats(torch.tensor([feature_list(d) for d in sample['player_a_surface']], dtype=torch.float))
        player_a_surface_mask = torch.tensor(sample['player_a_surface_mask'], dtype=torch.float)
        player_a_surface_pos = torch.arange(player_a_surface.shape[0])
        
        player_a_recent = self._drop_stats(torch.tensor([feature_list(d) for d in sample['player_a_recent']], dtype=torch.float))
        player_a_recent_mask = torch.tensor(sample['player_a_recent_mask'], dtype=torch.float)
        player_a_recent_pos = torch.arange(player_a_recent.shape[0])

        # Player B
        player_b_surface = self._drop_stats(torch.tensor([feature_list(d) for d in sample['player_b_surface']], dtype=torch.float))
        player_b_surface_mask = torch.tensor(sample['player_b_surface_mask'], dtype=torch.float)
        player_b_surface_pos = torch.arange(player_b_surface.shape[0])
        
        player_b_recent = self._drop_stats(torch.tensor([feature_list(d) for d in sample['player_b_recent']], dtype=torch.float))
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
        
        