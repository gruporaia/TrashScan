from pathlib import Path
import sys

DATASET = Path("datasets/taco_clean")

EXPECTED = {
    "train": 1200,
    "val": 300
}

def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)

def check_dir(path):
    if not path.exists():
        fail(f"missing directory: {path}")

print("\n--- DATASET STRUCTURE CHECK ---\n")

# directories
dirs = [
    DATASET/"images/train",
    DATASET/"images/val",
    DATASET/"labels/train",
    DATASET/"labels/val"
]

for d in dirs:
    check_dir(d)

print("✓ directory structure OK")

# count files
img_train = list((DATASET/"images/train").glob("*.jpg"))
img_val   = list((DATASET/"images/val").glob("*.jpg"))

lbl_train = list((DATASET/"labels/train").glob("*.txt"))
lbl_val   = list((DATASET/"labels/val").glob("*.txt"))

print("\n--- FILE COUNTS ---\n")

print("train images:", len(img_train))
print("train labels:", len(lbl_train))

print("val images:", len(img_val))
print("val labels:", len(lbl_val))

# exact counts
if len(img_train) != EXPECTED["train"]:
    fail("train image count mismatch")

if len(img_val) != EXPECTED["val"]:
    fail("val image count mismatch")

if len(lbl_train) != EXPECTED["train"]:
    fail("train label count mismatch")

if len(lbl_val) != EXPECTED["val"]:
    fail("val label count mismatch")

print("✓ file counts correct")

# image ↔ label matching
print("\n--- IMAGE / LABEL MATCHING ---\n")

img_train_names = {p.stem for p in img_train}
lbl_train_names = {p.stem for p in lbl_train}

img_val_names = {p.stem for p in img_val}
lbl_val_names = {p.stem for p in lbl_val}

if img_train_names != lbl_train_names:
    missing = img_train_names - lbl_train_names
    fail(f"missing train labels: {list(missing)[:5]}")

if img_val_names != lbl_val_names:
    missing = img_val_names - lbl_val_names
    fail(f"missing val labels: {list(missing)[:5]}")

print("✓ images and labels match")

# overlap
print("\n--- TRAIN/VAL LEAKAGE CHECK ---\n")

overlap = img_train_names & img_val_names

if overlap:
    fail(f"data leakage detected: {list(overlap)[:5]}")

print("✓ no train/val overlap")

print("\n--- DATASET VERIFIED SUCCESSFULLY ---\n")
