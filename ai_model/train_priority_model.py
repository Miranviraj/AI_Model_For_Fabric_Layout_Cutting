
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class PriorityDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        self.samples = [
            (
                [d["fabric_width"], d["fabric_height"], d["pattern_width"], d["pattern_height"]],
                [d["priority"]]
            )
            for d in data
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class PriorityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Load data
dataset = PriorityDataset("hybrid_training_data.json")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = PriorityModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Train
for epoch in range(10):
    total_loss = 0
    for x, y in loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "priority_model.pth")
print("✅ Model saved to priority_model.pth")
