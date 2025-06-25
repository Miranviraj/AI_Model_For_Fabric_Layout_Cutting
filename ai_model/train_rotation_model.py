
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RotationDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        self.samples = [
            (
                [d["fabric_width"], d["fabric_height"], d["pattern_width"], d["pattern_height"]],
                [d["should_rotate"]]
            )
            for d in data
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class RotationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Load dataset
dataset = RotationDataset("rotation_training_data.json")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train model
model = RotationModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

for epoch in range(10):
    total_loss = 0
    for x, y in loader:
        pred = model(x).squeeze()
        loss = loss_fn(pred, y.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "rotation_model.pth")
print("✅ Model saved to rotation_model.pth")
