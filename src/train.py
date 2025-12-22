import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from .dataset import DataSet
from .model import Model
import matplotlib.pyplot as plt

def train():
    num_epochs = 10
    batch_size = 32
    d_model = 64
    num_heads = 4
    num_layers = 2
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DataSet()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn = custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn = custom_collate_fn)

    model = Model(d_model, num_heads, num_layers).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
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
            labels = batch['label'].to(device)

            features = torch.cat([
                player_a_surface, player_a_recent, player_b_surface, player_b_recent
            ], dim=1)  # (batch, total_seq_len, feature_dim)

            positions = torch.cat([
                player_a_surface_pos, player_a_recent_pos, player_b_surface_pos, player_b_recent_pos
            ], dim=1)  # (batch, total_seq_len)

            masks = torch.cat([
                player_a_surface_mask, player_a_recent_mask, player_b_surface_mask, player_b_recent_mask
            ], dim=1)  # (batch, total_seq_len)

            outputs = model(features, positions, masks)
            attn_weights = model.get_attention_weights(features, positions, masks)
            layer = 0
            head = 0
            sample = 0
            attn = attn_weights[layer][sample, head].detach().cpu().numpy()
            plt.imshow(attn, cmap='viridis')
            plt.colorbar()
            plt.title(f'Attention Weights (Layer {layer}, Head {head}, Sample {sample})')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.show()
            loss = criterion(outputs, labels)

            # Check for NaN loss
            if torch.isnan(loss):
                print("Warning: NaN loss encountered. Skipping batch.")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Basic test evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
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
                labels = batch['label'].to(device)

                features = torch.cat([
                player_a_surface, player_a_recent, player_b_surface, player_b_recent
                ], dim=1)  # (batch, total_seq_len, feature_dim)

                positions = torch.cat([
                    player_a_surface_pos, player_a_recent_pos, player_b_surface_pos, player_b_recent_pos
                ], dim=1)  # (batch, total_seq_len)

                masks = torch.cat([
                    player_a_surface_mask, player_a_recent_mask, player_b_surface_mask, player_b_recent_mask
                ], dim=1)  # (batch, total_seq_len)

                outputs = model(features, positions, masks)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%")

        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")


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