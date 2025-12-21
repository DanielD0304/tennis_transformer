import torch
from torch.utils.data import DataLoader
from .dataset import DataSet
from .model import Model

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Model(d_model, num_heads, num_layers).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            # Helper to pad tensors to the same sequence length
            def pad_to_max_seq(tensors):
                max_len = max(t.shape[0] for t in tensors)
                padded = []
                for t in tensors:
                    if t.shape[0] < max_len:
                        if t.dim() == 1:
                            # 1D tensor (positions, masks)
                            pad_width = (0, max_len - t.shape[0])
                        elif t.dim() == 2:
                            # 2D tensor (features)
                            pad_width = (0, 0, 0, max_len - t.shape[0])
                        else:
                            raise ValueError("Unsupported tensor dimension for padding")
                        t = torch.nn.functional.pad(t, pad_width)
                    padded.append(t)
                return padded

            # Prepare player A
            a_feats = [batch['player_a_surface'], batch['player_a_recent']]
            a_pos = [batch['player_a_surface_pos'], batch['player_a_recent_pos']]
            a_mask = [batch['player_a_surface_mask'], batch['player_a_recent_mask']]
            a_feats = pad_to_max_seq(a_feats)
            a_pos = pad_to_max_seq(a_pos)
            a_mask = pad_to_max_seq(a_mask)
            player_a_features = torch.cat(a_feats, dim=0).to(device)
            player_a_positions = torch.cat(a_pos, dim=0).to(device)
            player_a_mask = torch.cat(a_mask, dim=0).to(device)

            # Prepare player B
            b_feats = [batch['player_b_surface'], batch['player_b_recent']]
            b_pos = [batch['player_b_surface_pos'], batch['player_b_recent_pos']]
            b_mask = [batch['player_b_surface_mask'], batch['player_b_recent_mask']]
            b_feats = pad_to_max_seq(b_feats)
            b_pos = pad_to_max_seq(b_pos)
            b_mask = pad_to_max_seq(b_mask)
            player_b_features = torch.cat(b_feats, dim=0).to(device)
            player_b_positions = torch.cat(b_pos, dim=0).to(device)
            player_b_mask = torch.cat(b_mask, dim=0).to(device)

            # Stack for batch (2, seq, features)
            inputs = torch.stack([player_a_features, player_b_features], dim=0)
            positions = torch.stack([player_a_positions, player_b_positions], dim=0)
            masks = torch.stack([player_a_mask, player_b_mask], dim=0)

            labels = batch['label'].to(device)

            outputs = model(inputs, positions, masks)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
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
                a_feats = [batch['player_a_surface'], batch['player_a_recent']]
                a_pos = [batch['player_a_surface_pos'], batch['player_a_recent_pos']]
                a_mask = [batch['player_a_surface_mask'], batch['player_a_recent_mask']]
                a_feats = pad_to_max_seq(a_feats)
                a_pos = pad_to_max_seq(a_pos)
                a_mask = pad_to_max_seq(a_mask)
                player_a_features = torch.cat(a_feats, dim=0).to(device)
                player_a_positions = torch.cat(a_pos, dim=0).to(device)
                player_a_mask = torch.cat(a_mask, dim=0).to(device)

                b_feats = [batch['player_b_surface'], batch['player_b_recent']]
                b_pos = [batch['player_b_surface_pos'], batch['player_b_recent_pos']]
                b_mask = [batch['player_b_surface_mask'], batch['player_b_recent_mask']]
                b_feats = pad_to_max_seq(b_feats)
                b_pos = pad_to_max_seq(b_pos)
                b_mask = pad_to_max_seq(b_mask)
                player_b_features = torch.cat(b_feats, dim=0).to(device)
                player_b_positions = torch.cat(b_pos, dim=0).to(device)
                player_b_mask = torch.cat(b_mask, dim=0).to(device)

                inputs = torch.stack([player_a_features, player_b_features], dim=0)
                positions = torch.stack([player_a_positions, player_b_positions], dim=0)
                masks = torch.stack([player_a_mask, player_b_mask], dim=0)

                labels = batch['label'].to(device)

                outputs = model(inputs, positions, masks)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%")

        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()