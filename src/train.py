import os

import numpy as np
import torch
from torch.utils.data import DataLoader as dl
from torch.nn.utils.rnn import pad_sequence

from . import preprocessing as pre
from .config import TrainingConfig, default_config
from .dataset import DataSet
from .loss import make_criterion
from .model import Model, load_model, save_checkpoint
from .models.calibration import fit_temperature


SEQUENCE_KEYS = ['player_a_surface', 'player_a_recent', 'player_b_surface', 'player_b_recent']


def _has_current_features(samples):
    """Passen gespeicherte Samples zu preprocessing.FEATURE_NAMES/EXTRA_FEATURES (z.B. nach neuen Features)?"""
    for s in samples[:50]:
        if not set(pre.EXTRA_FEATURES) <= set(s):
            return False
        for item in s['player_a_recent']:
            return set(pre.FEATURE_NAMES) <= set(item)
    return True


def load_or_preprocess_data(config):
    """
    Load preprocessed data if available, otherwise preprocess and save.
    """
    all_samples = None
    if os.path.exists(config.preprocessed_data_path):
        print(f"Loading preprocessed data from {config.preprocessed_data_path}...")
        all_samples = torch.load(config.preprocessed_data_path)
        if not _has_current_features(all_samples):
            print("Gespeicherte Samples haben einen alten Feature-Satz, werden neu erzeugt.")
            all_samples = None
    if all_samples is None:
        print("Preprocessed data not found. Processing now...")
        from . import data_loader as dl_module
        train_years = list(range(config.data_years_start, config.test_year))
        
        test_year = config.test_year
        raw_data = dl_module.load_merged_data(train_years, test_year)
        
        # ELO (Features der Historie) und Elo2 (zusätzlicher Input, Stand vor dem Match)
        from .elo import compute_elo_ratings, elo2_features
        raw_data = compute_elo_ratings(raw_data)
        raw_data = raw_data.join(elo2_features(raw_data))
        
        print("Processing samples...")
        all_samples = pre.create_training_samples(
            raw_data,
            n_matches=config.n_surface_matches,
            n_recent=config.n_recent_matches,
            max_rank=config.max_rank,
        )
        
        print(f"Saving to {config.preprocessed_data_path}...")
        os.makedirs(os.path.dirname(config.preprocessed_data_path), exist_ok=True)
        torch.save(all_samples, config.preprocessed_data_path)
        print("Done!")
    
    return all_samples


def model_inputs(batch, device):
    """Batch aus custom_collate_fn -> Argumente für Model.forward (inkl. [CLS]-Maske und Extra-Features)."""
    features = torch.cat([batch[k] for k in SEQUENCE_KEYS], dim=1).to(device)
    positions = torch.cat([batch[k + '_pos'] for k in SEQUENCE_KEYS], dim=1).to(device)
    masks = torch.cat([batch[k + '_mask'] for k in SEQUENCE_KEYS], dim=1).to(device)
    masks = torch.cat([torch.ones(masks.shape[0], 1, device=device, dtype=masks.dtype), masks], dim=1)
    extra = batch['extra'].to(device) if 'extra' in batch else None
    return features, positions, batch['segment_ids'].to(device).long(), masks, extra


def swapped_sample(s):
    """Dasselbe Match aus Sicht von B: Sequenzen tauschen, Extra-Features (Logits für A) negieren."""
    out = dict(s)
    for part in ('surface', 'recent'):
        for suffix in ('', '_mask'):
            a, b = f'player_a_{part}{suffix}', f'player_b_{part}{suffix}'
            out[a], out[b] = s[b], s[a]
    for k in pre.EXTRA_FEATURES:
        out[k] = -s.get(k, 0.0)
    out['label'] = 1 - s['label']
    return out


@torch.no_grad()
def raw_logits(model, samples, device, batch_size=256):
    """Logit von P(Sample-Spieler A gewinnt) = z1 - z0, ohne Symmetrisierung und Temperatur."""
    model.eval()
    extra = getattr(model.transformer, 'extra_dim', 0)
    loader = dl(DataSet(samples), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    out = []
    for batch in loader:
        x, pos, seg, mask, ex = model_inputs(batch, device)
        logits = model(x, pos, seg, mask, ex if extra else None)
        out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def symmetric_logits(model, samples, device, batch_size=256):
    """(z(A, B) - z(B, A)) / 2: unabhängig davon, wer Spieler A ist."""
    return (raw_logits(model, samples, device, batch_size)
            - raw_logits(model, [swapped_sample(s) for s in samples], device, batch_size)) / 2


def _log_loss(z, y):
    p = np.clip(1 / (1 + np.exp(-z)), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train(config=default_config):
    """
    Train the tennis match prediction model.

    Checkpoint nach dem Log Loss auf dem Validierungsjahr. Danach Temperature Scaling auf demselben
    Jahr (symmetrische Logits), die Temperatur steht mit im Checkpoint. Streng out-of-sample sind
    damit erst die Jahre nach config.val_year.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load or preprocess data
    all_samples = load_or_preprocess_data(config)
    
    # Split data by year
    train_samples = [s for s in all_samples if s['year'] <= config.train_years_end]
    val_samples = [s for s in all_samples if s['year'] == config.val_year]
    test_samples = [s for s in all_samples if s['year'] == config.test_year]
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    train_dataset = DataSet(train_samples, stats_dropout=config.stats_dropout)
    train_loader = dl(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    model = Model(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        input_dim=config.input_dim,
        max_len=config.max_len,
        output_dim=config.output_dim,
        extra_dim=config.extra_dim,
    ).to(device)
    
    criterion = make_criterion(config.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    val_labels = np.array([s['label'] for s in val_samples], float)

    patience_counter = 0
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x, pos, seg, mask, extra = model_inputs(batch, device)
            outputs = model(x, pos, seg, mask, extra if config.extra_dim else None)
            loss = criterion(outputs, batch['label'].to(device))

            if torch.isnan(loss):
                print("Warning: NaN loss encountered. Skipping batch.")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Validierung: Log Loss der symmetrischen Vorhersage (das, was später benutzt wird)
        z_val = symmetric_logits(model, val_samples, device)
        val_loss = _log_loss(z_val, val_labels)
        val_acc = float(np.mean((z_val > 0) == (val_labels == 1)))
        print(f"Validation Log Loss: {val_loss:.4f}, Accuracy: {100 * val_acc:.2f}%, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, config.best_model_path, extra_dim=config.extra_dim, input_dim=config.input_dim)
            print(f"New best model saved! Validation Log Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience: {patience_counter}/{config.patience}")
            if patience_counter >= config.patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    # Temperature Scaling auf dem Validierungsjahr mit dem besten Checkpoint
    model, _ = load_model(config.best_model_path, config, device)
    z_val = symmetric_logits(model, val_samples, device)
    temperature = fit_temperature(z_val, val_labels)
    save_checkpoint(model, config.best_model_path, temperature, config.extra_dim, config.input_dim)
    print(f"Temperatur {temperature:.3f}: Val Log Loss {_log_loss(z_val, val_labels):.4f} -> "
          f"{_log_loss(z_val / temperature, val_labels):.4f}")
    if test_samples:
        z_test = symmetric_logits(model, test_samples, device) / temperature
        test_labels = np.array([s['label'] for s in test_samples], float)
        print(f"Test {config.test_year}: Log Loss {_log_loss(z_test, test_labels):.4f}, "
              f"Accuracy {100 * np.mean((z_test > 0) == (test_labels == 1)):.2f}% (n={len(test_samples)})")
    return model, temperature


def custom_collate_fn(batch):
    keys = batch[0].keys()
    batched_data = {}
    for key in keys:
        features = [sample[key] for sample in batch]
        if features[0].dim() > 0:
            batched_data[key] = pad_sequence(features, batch_first=True, padding_value=0)
        else:
            batched_data[key] = torch.stack(features)
    return batched_data
        
if __name__ == "__main__":
    train()