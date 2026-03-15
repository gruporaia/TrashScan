from pathlib import Path
import subprocess
from collections import Counter

DATASET = Path("datasets/taco_clean")

CLASS_NAMES = [
    "plastic",
    "metal",
    "paper",
    "glass",
    "organic",
    "other"
]

YAML_FILE = "taco.yaml"

MODEL = "yolov8s.pt"
EPOCHS = 200
IMG_SIZE = 960
BATCH = 16
PATIENCE = 30


# --------------------------------------------------
# 1. dataset sanity check
# --------------------------------------------------

print("\n--- DATASET CHECK ---\n")

train_imgs = list((DATASET/"images/train").glob("*.jpg"))
val_imgs = list((DATASET/"images/val").glob("*.jpg"))

print("train images:", len(train_imgs))
print("val images:", len(val_imgs))


# --------------------------------------------------
# 2. class distribution
# --------------------------------------------------

print("\n--- CLASS DISTRIBUTION ---\n")

counter = Counter()

for label in (DATASET/"labels/train").glob("*.txt"):

    with open(label) as f:

        for line in f:

            cls = int(line.split()[0])
            counter[cls] += 1


total_objects = sum(counter.values())

for c in sorted(counter):

    name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else "unknown"

    print(f"class {c} ({name}): {counter[c]}")

print("\nTotal objects:", total_objects)


# --------------------------------------------------
# 3. generate YAML
# --------------------------------------------------

print("\n--- GENERATING taco.yaml ---\n")

yaml_text = f"""
path: {DATASET}

train: images/train
val: images/val

nc: {len(CLASS_NAMES)}

names:
"""

for i, name in enumerate(CLASS_NAMES):
    yaml_text += f"  {i}: {name}\n"

with open(YAML_FILE, "w") as f:
    f.write(yaml_text)

print("✓ taco.yaml created")


# --------------------------------------------------
# 4. training
# --------------------------------------------------

print("\n--- STARTING TRAINING ---\n")

train_cmd = [
    "yolo", "detect", "train",
    f"data={YAML_FILE}",
    f"model={MODEL}",
    f"epochs={EPOCHS}",
    f"imgsz={IMG_SIZE}",
    f"batch={BATCH}",
    f"patience={PATIENCE}",
    "mosaic=1",
    "mixup=0.3",
    "fliplr=0.5",
    "scale=0.5",
    "degrees=10"
]

print(" ".join(train_cmd))
print()

subprocess.run(train_cmd)


# --------------------------------------------------
# 5. evaluation
# --------------------------------------------------

print("\n--- FINAL VALIDATION ---\n")

val_cmd = [
    "yolo", "detect", "val",
    "model=runs/detect/train/weights/best.pt",
    f"data={YAML_FILE}",
    f"imgsz={IMG_SIZE}"
]

print(" ".join(val_cmd))
print()

subprocess.run(val_cmd)


print("\n--- TRAINING PIPELINE COMPLETED ---\n")
