from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn

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

# Load the model
model = AIModel()
model.load_state_dict(torch.load("../ai_model/fabric_ai_model.pth"))
model.eval()

app = Flask(__name__)
CORS(app)

@app.route('/api/cut-fabric', methods=['POST'])
def cut_fabric():
    data = request.get_json()
    fabric_width = data['fabric']['width']
    fabric_height = data['fabric']['height']
    patterns = data['patterns']

    layout = []
    for p in patterns:
        input_tensor = torch.tensor(
            [[fabric_width, fabric_height, p["width"], p["height"]]],
            dtype=torch.float32
        )
        with torch.no_grad():
            output = model(input_tensor).numpy().tolist()[0]
        layout.append({
            "x": int(abs(output[0])) % fabric_width,
            "y": int(abs(output[1])) % fabric_height,
            "width": p["width"],
            "height": p["height"]
        })

    used_area = sum([r["width"] * r["height"] for r in layout])
    total_area = fabric_width * fabric_height
    waste = total_area - used_area

    return jsonify({
        "layout": layout,
        "used_area": used_area,
        "waste_area": waste,
        "waste_percentage": round((waste / total_area) * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
