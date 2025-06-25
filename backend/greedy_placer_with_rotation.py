import torch
import torch.nn as nn

# AI model for rotation prediction
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

def load_rotation_model(path="rotation_model.pth"):
    model = RotationModel()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

def greedy_place_with_rotation(fabric, patterns, rotation_model):
    placed = []
    grid = [[0]*fabric["width"] for _ in range(fabric["height"])]

    def can_place(x, y, w, h):
        if x + w > fabric["width"] or y + h > fabric["height"]:
            return False
        for i in range(y, y+h):
            for j in range(x, x+w):
                if grid[i][j] == 1:
                    return False
        return True

    def place(x, y, w, h):
        for i in range(y, y+h):
            for j in range(x, x+w):
                grid[i][j] = 1

    for pattern in patterns:
        # Predict rotation
        input_tensor = torch.tensor([
            fabric["width"],
            fabric["height"],
            pattern["width"],
            pattern["height"]
        ], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            rotate_prob = rotation_model(input_tensor).item()
        should_rotate = rotate_prob > 0.5

        # Try placing rotated dimensions first if predicted
        w, h = pattern["width"], pattern["height"]
        if should_rotate:
            w, h = h, w

        placed_flag = False
        for y in range(fabric["height"]):
            for x in range(fabric["width"]):
                if can_place(x, y, w, h):
                    place(x, y, w, h)
                    placed.append({
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "rotated": should_rotate
                    })
                    placed_flag = True
                    break
            if placed_flag:
                break

    return placed
