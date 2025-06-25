import json
from rectpack import newPacker
import random

def generate_data(samples=1000):
    dataset = []

    for _ in range(samples):
        fabric = {
            "width": random.randint(100, 200),
            "height": random.randint(100, 200)
        }

        max_w = fabric["width"] - 10
        max_h = fabric["height"] - 10

        if max_w < 20 or max_h < 20:
            continue

        num_patterns = random.randint(3, 10)
        patterns = []
        for i in range(num_patterns):
            width = random.randint(10, max_w)
            height = random.randint(10, max_h)
            patterns.append({ "id": i, "width": width, "height": height })

        # Setup rectpack packer
        packer = newPacker()
        packer.add_bin(fabric["width"], fabric["height"])
        for p in patterns:
            packer.add_rect(p["width"], p["height"], rid=p["id"])

        packer.pack()

        # Extract layout
        layout = []
        for rect in packer.rect_list():
            b, x, y, w, h, rid = rect
            layout.append({
                "id": rid,
                "x": x,
                "y": y,
                "width": w,
                "height": h
            })

        # Ensure all patterns were packed
        if len(layout) != len(patterns):
            continue

        # Create training samples
        for pattern in layout:
            dataset.append({
                "fabric_width": fabric["width"],
                "fabric_height": fabric["height"],
                "pattern_width": pattern["width"],
                "pattern_height": pattern["height"],
                "x": pattern["x"],
                "y": pattern["y"]
            })

    # Save to file
    with open("cutting_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Generated {len(dataset)} samples and saved to cutting_dataset.json")

if __name__ == "__main__":
    generate_data()
