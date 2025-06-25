import torch
import torch.nn as nn

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

def load_model(model_path="priority_model.pth"):
    model = PriorityModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def greedy_place(fabric, patterns, model):
    scored_patterns = []
    for pattern in patterns:
        input_tensor = torch.tensor([
            fabric["width"],
            fabric["height"],
            pattern["width"],
            pattern["height"]
        ], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            priority = model(input_tensor).item()
        scored_patterns.append((priority, pattern))

    scored_patterns.sort(reverse=True, key=lambda x: x[0])
    placed = []
    skipped = 0

    grid = [[0] * fabric["width"] for _ in range(fabric["height"])]

    def can_place(x, y, w, h):
        if x + w > fabric["width"] or y + h > fabric["height"]:
            return False
        for i in range(y, y + h):
            for j in range(x, x + w):
                if grid[i][j] == 1:
                    return False
        return True

    def place(x, y, w, h):
        for i in range(y, y + h):
            for j in range(x, x + w):
                grid[i][j] = 1

    for _, pattern in scored_patterns:
        w_orig, h_orig = pattern["width"], pattern["height"]
        placed_flag = False

        # Try original
        for y in range(fabric["height"]):
            for x in range(fabric["width"]):
                if can_place(x, y, w_orig, h_orig):
                    place(x, y, w_orig, h_orig)
                    placed.append({
                        "x": x,
                        "y": y,
                        "width": w_orig,
                        "height": h_orig,
                        "rotated": False
                    })
                    placed_flag = True
                    break
            if placed_flag:
                break

        # Try rotated
        if not placed_flag:
            w_alt, h_alt = h_orig, w_orig
            for y in range(fabric["height"]):
                for x in range(fabric["width"]):
                    if can_place(x, y, w_alt, h_alt):
                        place(x, y, w_alt, h_alt)
                        placed.append({
                            "x": x,
                            "y": y,
                            "width": w_alt,
                            "height": h_alt,
                            "rotated": True
                        })
                        placed_flag = True
                        break
                if placed_flag:
                    break

        if not placed_flag:
            skipped += 1  

    return placed, skipped
