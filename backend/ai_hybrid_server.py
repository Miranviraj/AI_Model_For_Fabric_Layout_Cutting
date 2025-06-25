from flask import Flask, request, jsonify
from flask_cors import CORS
from greedy_placer import load_model, greedy_place

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "http://localhost:5173"}})

model = load_model("priority_model.pth")


# Optimize waste usage using dynamic programming
def get_best_cut_mix(waste_area, pattern_options):
    patterns = []
    for name, (w, h) in pattern_options.items():
        patterns.append({
            "name": name,
            "area": w * h
        })

    dp = [0] * (waste_area + 1)
    decision = [{} for _ in range(waste_area + 1)]

    for i in range(waste_area + 1):
        for pat in patterns:
            if pat["area"] <= i:
                prev = i - pat["area"]
                new_count = decision[prev].copy()
                new_count[pat["name"]] = new_count.get(pat["name"], 0) + 1
                new_area = dp[prev] + pat["area"]
                if new_area > dp[i]:
                    dp[i] = new_area
                    decision[i] = new_count

    return decision[waste_area]


@app.route('/api/hybrid-layout', methods=['POST'])
def hybrid_layout():
    data = request.get_json()
    fabric = data.get("fabric")
    patterns = data.get("patterns")

    if not fabric or not patterns:
        return jsonify({"error": "Missing fabric or patterns"}), 400

    layout, skipped = greedy_place(fabric, patterns, model)

    used_area = sum([r["width"] * r["height"] for r in layout])
    total_area = fabric["width"] * fabric["height"]
    waste = total_area - used_area

    pattern_options = {
        "Small": (40, 40),
        "Medium": (60, 60),
        "Large": (70, 70),
        "ShortSleeve": (25, 20),
        "LongSleeve": (20, 40)
    }

    possible_cuts = get_best_cut_mix(waste, pattern_options)

    return jsonify({
        "layout": layout,
        "used_area": used_area,
        "waste_area": waste,
        "waste_percentage": round((waste / total_area) * 100, 2),
        "possible_cuts": possible_cuts,
        "unplaced_patterns": skipped 
    })


if __name__ == '__main__':
    app.run(debug=True)
