
import json
from rectpack import newPacker
import random

def generate_hybrid_training_data(samples=1000):
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

        packer = newPacker()
        packer.add_bin(fabric["width"], fabric["height"])
        for p in patterns:
            packer.add_rect(p["width"], p["height"], rid=p["id"])

        packer.pack()

        rects = packer.rect_list()
        if len(rects) != len(patterns):
            continue  # skip if some patterns were not placed

        # Sort rects by placement order (top to bottom)
        rects.sort(key=lambda r: (r[2], r[1]))  # sort by y, then x

        # Assign priority (normalized placement order)
        for i, rect in enumerate(rects):
            b, x, y, w, h, rid = rect
            priority = 1 - (i / (len(rects) - 1)) if len(rects) > 1 else 1.0
            dataset.append({
                "fabric_width": fabric["width"],
                "fabric_height": fabric["height"],
                "pattern_width": w,
                "pattern_height": h,
                "priority": priority
            })

    with open("hybrid_training_data.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Generated {len(dataset)} samples for hybrid AI training.")

if __name__ == "__main__":
    generate_hybrid_training_data()
