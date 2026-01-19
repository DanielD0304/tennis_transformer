import torch
from torch.utils.data import DataLoader as dl
from torch.nn.utils.rnn import pad_sequence
from .dataset import DataSet
from .model import Model
import matplotlib.pyplot as plt
from . import preprocessing as pre
from .config import TrainingConfig, default_config
import os


def load_or_preprocess_data(config):
    """
    Load preprocessed data if available, otherwise preprocess and save.
    """
    if os.path.exists(config.preprocessed_data_path):
        print(f"Loading preprocessed data from {config.preprocessed_data_path}...")
        all_samples = torch.load(config.preprocessed_data_path)
        print(f"Loaded {len(all_samples)} samples.")
    else:
        print("Preprocessed data not found. Processing now...")
        from . import data_loader as dl_module
        years = list(range(config.data_years_start, config.data_years_end + 1))
        raw_data = dl_module.load_all_matches(years)
        
        # WICHTIG: ELO berechnen, falls noch nicht geschehen (für neue Features)
        from .elo import compute_elo_ratings
        raw_data = compute_elo_ratings(raw_data)
        
        print("Processing samples...")
        all_samples = pre.create_training_samples(
            raw_data,
            n_matches=config.n_surface_matches,
            n_recent=config.n_recent_matches
        )
        
        print(f"Saving to {config.preprocessed_data_path}...")
        
        # --- FIX 1: Ordner 'data/' erstellen ---
        os.makedirs(os.path.dirname(config.preprocessed_data_path), exist_ok=True)
        # ---------------------------------------
        
        torch.save(all_samples, config.preprocessed_data_path)
        print("Done!")
    
    return all_samples


def train(config=default_config):
    """
    Train the tennis match prediction model.
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

    # Create datasets
    train_dataset = DataSet(train_samples)
    val_dataset = DataSet(val_samples)
    test_dataset = DataSet(test_samples)

    # Create loaders
    train_loader = dl(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = dl(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn) 
    test_loader = dl(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Initialize model
    model = Model(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        input_dim=config.input_dim,
        max_len=config.max_len,
        output_dim=config.output_dim
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )
    
    # Tracking variables
    best_val_accuracy = 0.0
    patience_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # ... (Dein Training Loop Code bleibt gleich) ...
            player_a_surface = batch['player_a_surface'].to(device)
            player_a_recent = batch['player_a_recent'].to(device)
            player_a_surface_pos = batch['player_a_surface_pos'].to(device)
            player_a_recent_pos = batch['player_a_recent_pos'].to(device)
            player_a_surface_mask = batch['player_a_surface_mask'].to(device)
            player_a_recent_mask = batch['player_a_recent_mask'].to(device)
            player_b_surface = batch['player_b_surface'].to(device)
            player_b_recent = batch['player_b_recent'].to(device)
            player_b_surface_pos = batch['player_b_surface_pos'].to(device)
            player_b_recent_pos = batch['player_b_recent_pos'].to(device)
            player_b_surface_mask = batch['player_b_surface_mask'].to(device)
            player_b_recent_mask = batch['player_b_recent_mask'].to(device)
            segment_ids = batch['segment_ids'].to(device).long()
            labels = batch['label'].to(device)

            features = torch.cat([
                player_a_surface, player_a_recent, player_b_surface, player_b_recent
            ], dim=1)

            positions = torch.cat([
                player_a_surface_pos, player_a_recent_pos, player_b_surface_pos, player_b_recent_pos
            ], dim=1)

            masks = torch.cat([
                player_a_surface_mask, player_a_recent_mask, player_b_surface_mask, player_b_recent_mask
            ], dim=1)
            cls_mask = torch.ones(masks.shape[0], 1, device=masks.device, dtype=masks.dtype)
            masks = torch.cat([cls_mask, masks], dim=1)

            outputs = model(features, positions, segment_ids, masks)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                print("Warning: NaN loss encountered. Skipping batch.")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                # ... (Dein Validation Loop Code bleibt gleich) ...
                player_a_surface = batch['player_a_surface'].to(device)
                player_a_recent = batch['player_a_recent'].to(device)
                player_a_surface_pos = batch['player_a_surface_pos'].to(device)
                player_a_recent_pos = batch['player_a_recent_pos'].to(device)
                player_a_surface_mask = batch['player_a_surface_mask'].to(device)
                player_a_recent_mask = batch['player_a_recent_mask'].to(device)
                player_b_surface = batch['player_b_surface'].to(device)
                player_b_recent = batch['player_b_recent'].to(device)
                player_b_surface_pos = batch['player_b_surface_pos'].to(device)
                player_b_recent_pos = batch['player_b_recent_pos'].to(device)
                player_b_surface_mask = batch['player_b_surface_mask'].to(device)
                player_b_recent_mask = batch['player_b_recent_mask'].to(device)
                segment_ids = batch['segment_ids'].to(device).long()
                labels = batch['label'].to(device)

                features = torch.cat([
                    player_a_surface, player_a_recent, player_b_surface, player_b_recent
                ], dim=1)

                positions = torch.cat([
                    player_a_surface_pos, player_a_recent_pos, player_b_surface_pos, player_b_recent_pos
                ], dim=1)

                masks = torch.cat([
                    player_a_surface_mask, player_a_recent_mask, player_b_surface_mask, player_b_recent_mask
                ], dim=1)
                cls_mask = torch.ones(masks.shape[0], 1, device=masks.device, dtype=masks.dtype)
                masks = torch.cat([cls_mask, masks], dim=1)

                outputs = model(features, positions, segment_ids, masks)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        current_val_acc = val_correct / val_total
        current_val_loss = val_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Validation Loss: {current_val_loss:.4f}, Accuracy: {100 * current_val_acc:.2f}%, LR: {current_lr:.6f}")
        
        scheduler.step(current_val_loss)
        
        # Best Model Checkpointing
        if current_val_acc > best_val_accuracy:
            best_val_accuracy = current_val_acc
            
            # --- FIX 2: Ordner 'models/' erstellen ---
            os.makedirs(os.path.dirname(config.best_model_path), exist_ok=True)
            # -----------------------------------------
            
            torch.save(model.state_dict(), config.best_model_path)
            print(f"New best model saved! Validation Accuracy: {100 * best_val_accuracy:.2f}%")
        
        # Early Stopping
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience: {patience_counter}/{config.patience}")
            if patience_counter >= config.patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
        
        # ... (Dein Visualization Code bleibt gleich) ...
        # Visualize attention weights
        val_batch = next(iter(val_loader))
        # (Lade hier die Daten wie oben für Visualisierung...)
        # Ich habe das hier aus Platzgründen gekürzt, da der Fehler oben lag. 
        # Dein Visualisierungscode unten war okay, solange er funktioniert.
        
        # Basic test evaluation (am Ende der Epoche)
        # ...

        # Falls du hier auch speicherst:
        # torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt") 
        # Das speichert im Root-Verzeichnis, das ist okay.

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