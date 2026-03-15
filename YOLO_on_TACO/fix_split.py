import random
import shutil
from pathlib import Path

DATASET="datasets/taco_yolo"

IMG=Path(DATASET+"/images/train")
LBL=Path(DATASET+"/labels/train")

OUT="datasets/taco_clean"

train_ratio=0.8

random.seed(42)

imgs=list(IMG.glob("*.jpg"))

random.shuffle(imgs)

split=int(len(imgs)*train_ratio)

train_imgs=imgs[:split]
val_imgs=imgs[split:]

print("train:",len(train_imgs))
print("val:",len(val_imgs))

for p in [
    "images/train","images/val",
    "labels/train","labels/val"
]:
    Path(OUT+"/"+p).mkdir(parents=True,exist_ok=True)

def copy_pair(img,split):

    label=LBL/(img.stem+".txt")

    shutil.copy(img,f"{OUT}/images/{split}/{img.name}")
    shutil.copy(label,f"{OUT}/labels/{split}/{img.stem}.txt")

for img in train_imgs:
    copy_pair(img,"train")

for img in val_imgs:
    copy_pair(img,"val")

print("done")
