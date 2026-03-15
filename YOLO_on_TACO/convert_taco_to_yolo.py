import json
import os
import shutil
from pathlib import Path
import cv2

BASE_DIR = "datasets/TACO/data"
ANNOT_FILE = f"{BASE_DIR}/annotations.json"

OUT_DIR = "datasets/taco_yolo"

IMG_OUT = f"{OUT_DIR}/images/train"
LBL_OUT = f"{OUT_DIR}/labels/train"

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

print("Loading annotations...")

with open(ANNOT_FILE) as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
categories = {c["id"]: c["name"] for c in coco["categories"]}

# agrupar anotações por imagem
image_annotations = {}
for ann in coco["annotations"]:
    image_annotations.setdefault(ann["image_id"], []).append(ann)


# ------------------------------------------------
# class grouping (60 → 6 classes)
# ------------------------------------------------

def map_category(name):

    name = name.lower()

    if "plastic" in name or "styrofoam" in name:
        return 0

    if "metal" in name or "can" in name or "aluminium" in name:
        return 1

    if "paper" in name or "cardboard" in name:
        return 2

    if "glass" in name:
        return 3

    if "food" in name or "organic" in name:
        return 4

    return 5


copied = 0

for img_id, img_info in images.items():

    file_name = img_info["file_name"]

    img_path = os.path.join(BASE_DIR, file_name)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)

    if img is None:
        continue

    h, w = img.shape[:2]

    # cria nome único usando batch
    unique_name = file_name.replace("/", "_")

    stem = Path(unique_name).stem

    label_file = f"{LBL_OUT}/{stem}.txt"

    anns = image_annotations.get(img_id, [])

    with open(label_file, "w") as f:

        for ann in anns:

            x, y, bw, bh = ann["bbox"]

            x_center = (x + bw/2) / w
            y_center = (y + bh/2) / h
            bw /= w
            bh /= h

            cat_name = categories[ann["category_id"]]

            class_id = map_category(cat_name)

            f.write(f"{class_id} {x_center} {y_center} {bw} {bh}\n")

    shutil.copy(img_path, f"{IMG_OUT}/{stem}.jpg")

    copied += 1


print("Images processed:", copied)
