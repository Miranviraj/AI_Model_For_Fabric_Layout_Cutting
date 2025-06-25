import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class CuttingDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        self.samples = [
            (
                [r["fabric_width"], r["fabric_height"], r["pattern_width"], r["pattern_height"]],
                [r["x"], r["y"]]
            )
            for r in raw
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class AIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

# Train the model
dataset = CuttingDataset("cutting_dataset.json")
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = AIModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(10):
    total_loss = 0
    for x, y in loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

# Save model
torch.save(model.state_dict(), "fabric_ai_model.pth")
