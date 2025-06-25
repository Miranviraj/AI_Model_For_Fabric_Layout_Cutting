from flask import Flask, request, jsonify
from flask_cors import CORS

from greedy_placer_with_rotation import load_rotation_model, greedy_place_with_rotation

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# Load model
rotation_model = load_rotation_model("rotation_model.pth")

@app.route("/api/ai-rotation-layout", methods=["POST"])
def ai_rotation_layout():
    data = request.get_json()
    fabric = data.get("fabric")
    patterns = data.get("patterns")

    if not fabric or not patterns:
        return jsonify({"error": "Missing fabric or patterns"}), 400

    layout = greedy_place_with_rotation(fabric, patterns, rotation_model)

    used_area = sum(p["width"] * p["height"] for p in layout)
    total_area = fabric["width"] * fabric["height"]
    waste = total_area - used_area

    return jsonify({
        "layout": layout,
        "used_area": used_area,
        "waste_area": waste,
        "waste_percentage": round((waste / total_area) * 100, 2)
    })

@app.route("/api/test")
def test():
    return "✅ AI server is working."

if __name__ == "__main__":
    print("🔍 Registered Routes:")
    for rule in app.url_map.iter_rules():
        print(rule)
    app.run(debug=True)
