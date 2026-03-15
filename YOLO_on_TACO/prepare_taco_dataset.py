import os
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np

DATASET_DIR = "datasets/taco_yolo"
IMG_DIR = f"{DATASET_DIR}/images/train"
LBL_DIR = f"{DATASET_DIR}/labels/train"

OUT_DIR = "datasets/taco_prepared"

TRAIN_RATIO = 0.8
TARGET_IMAGES_PER_CLASS = 400

random.seed(42)

os.makedirs(f"{OUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/images/val", exist_ok=True)

os.makedirs(f"{OUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/val", exist_ok=True)


# ------------------------------------------------
# utility
# ------------------------------------------------

def load_labels(path):

    classes = []

    with open(path) as f:
        for line in f:
            c = int(line.split()[0])
            classes.append(c)

    return classes


# ------------------------------------------------
# load dataset
# ------------------------------------------------

images = list(Path(IMG_DIR).glob("*.jpg"))

image_classes = {}

class_images = defaultdict(list)

for img_path in images:

    label_path = Path(LBL_DIR) / (img_path.stem + ".txt")

    if not label_path.exists():
        continue

    classes = load_labels(label_path)

    image_classes[img_path] = classes

    for c in set(classes):
        class_images[c].append(img_path)


print("dataset size:", len(images))


# ------------------------------------------------
# stratified split
# ------------------------------------------------

train_set = set()
val_set = set()

for c, imgs in class_images.items():

    random.shuffle(imgs)

    split = int(len(imgs) * TRAIN_RATIO)

    train_set.update(imgs[:split])
    val_set.update(imgs[split:])


train_set = list(train_set)
val_set = list(val_set)

print("train images:", len(train_set))
print("val images:", len(val_set))


# ------------------------------------------------
# copy base dataset
# ------------------------------------------------

def copy_pair(img_path, split):

    label_path = Path(LBL_DIR) / (img_path.stem + ".txt")

    shutil.copy(img_path, f"{OUT_DIR}/images/{split}/{img_path.name}")
    shutil.copy(label_path, f"{OUT_DIR}/labels/{split}/{img_path.stem}.txt")


for img in train_set:
    copy_pair(img, "train")

for img in val_set:
    copy_pair(img, "val")


# ------------------------------------------------
# augmentations
# ------------------------------------------------

def augment(img):

    augments = []

    # horizontal flip
    augments.append(cv2.flip(img, 1))

    # brightness
    augments.append(cv2.convertScaleAbs(img, alpha=1.2, beta=30))

    # blur
    augments.append(cv2.GaussianBlur(img, (5,5), 0))

    # noise
    noise = np.random.normal(0,10,img.shape).astype(np.uint8)
    augments.append(cv2.add(img, noise))

    return augments


# ------------------------------------------------
# class balancing via oversampling
# ------------------------------------------------

train_labels = list(Path(f"{OUT_DIR}/labels/train").glob("*.txt"))

class_counts = Counter()

for label in train_labels:

    classes = load_labels(label)

    for c in classes:
        class_counts[c] += 1


print("class distribution before balance:", class_counts)


for c, count in class_counts.items():

    if count >= TARGET_IMAGES_PER_CLASS:
        continue

    imgs = class_images[c]

    needed = TARGET_IMAGES_PER_CLASS - count

    for i in range(needed):

        img_path = random.choice(imgs)

        img = cv2.imread(str(img_path))

        aug_imgs = augment(img)

        aug = random.choice(aug_imgs)

        new_name = f"{img_path.stem}_aug_{i}.jpg"

        cv2.imwrite(f"{OUT_DIR}/images/train/{new_name}", aug)

        label_path = Path(LBL_DIR) / (img_path.stem + ".txt")

        shutil.copy(label_path,
                    f"{OUT_DIR}/labels/train/{Path(new_name).stem}.txt")


print("dataset preparation finished")
