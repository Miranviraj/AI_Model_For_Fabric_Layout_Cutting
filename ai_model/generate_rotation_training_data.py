
import json
import random
from rectpack import newPacker

def generate_rotation_training_data(samples=1000):
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

        width = random.randint(10, max_w)
        height = random.randint(10, max_h)

        # Try placing without rotation
        packer1 = newPacker()
        packer1.add_bin(fabric["width"], fabric["height"])
        packer1.add_rect(width, height)
        packer1.pack()
        placed_normal = len(packer1.rect_list()) > 0

        # Try placing with rotation
        packer2 = newPacker()
        packer2.add_bin(fabric["width"], fabric["height"])
        packer2.add_rect(height, width)
        packer2.pack()
        placed_rotated = len(packer2.rect_list()) > 0

        # If both succeed, label rotate if rotation gave better position (closer to top-left)
        should_rotate = 0
        if placed_rotated and not placed_normal:
            should_rotate = 1
        elif placed_rotated and placed_normal:
            x1, y1 = packer1.rect_list()[0][1:3]
            x2, y2 = packer2.rect_list()[0][1:3]
            should_rotate = 1 if (y2, x2) < (y1, x1) else 0

        dataset.append({
            "fabric_width": fabric["width"],
            "fabric_height": fabric["height"],
            "pattern_width": width,
            "pattern_height": height,
            "should_rotate": should_rotate
        })

    with open("rotation_training_data.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Generated {len(dataset)} samples for rotation AI training.")

if __name__ == "__main__":
    generate_rotation_training_data()
