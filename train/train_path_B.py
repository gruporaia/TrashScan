"""
train_path_B.py  —  Path B: Two-stage detect + classify pipeline.

Stage 1: YOLO detector (reuses best Path A weights) — localizes objects
Stage 2: Crop classifier (ResNet-50 or ViT-B/16) — refines class prediction

The detector is frozen. Only the classifier is trained on 224x224 crops
extracted from processed/train/path_B/crops/.

Usage:
  python train_path_B.py \
      --detector_weights ~/runs/path_A/yolov9s/weights/best.pt \
      --crops_dir        ~/processed \
      --output           ~/runs/path_B \
      --classifiers      resnet50 vit_b16 \
      --epochs           50 \
      --batch            32 \
      --device           0
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm
import timm
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

COARSE_CLASSES = ["plastic", "paper", "metal", "other"]
NUM_CLASSES    = len(COARSE_CLASSES)
CROP_SIZE      = 224

CLASSIFIER_REGISTRY = {
    # B1: CNN classifier — strong ImageNet pretrained baseline
    "resnet50":         ("resnet50.a1_in1k",                             True),
    # B2: ViT random init — tests transformer architecture without pretraining
    "vit_b16_scratch":  ("vit_base_patch16_224",                         False),
    # B3: ViT ImageNet pretrained — tests value of large-scale pretraining
    "vit_b16_imagenet": ("vit_base_patch16_224.augreg_in21k_ft_in1k",    True),
}

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class CropDataset(Dataset):
    """
    Loads 224x224 instance crops from processed/{split}/path_B/crops/{class_idx}/.
    Each subfolder name is the integer class index.
    """
    def __init__(self, crops_root: Path, split: str, transform=None):
        self.transform = transform
        self.samples   = []

        split_dir = crops_root / split / "path_B" / "crops"
        assert split_dir.exists(), f"Crops dir not found: {split_dir}"

        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            try:
                cls_idx = int(cls_dir.name)
            except ValueError:
                continue
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((img_path, cls_idx))

        print(f"  [{split}] {len(self.samples)} crops across "
              f"{NUM_CLASSES} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = read_image(str(img_path)).float() / 255.0
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.3, hue=0.1
            ),
            transforms.RandomRotation(30),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
        ])


def get_class_weights(crops_root: Path) -> torch.Tensor:
    """Load class weights from meta/class_weights.yaml."""
    weights_path = crops_root / "meta" / "class_weights.yaml"
    if not weights_path.exists():
        print("[warn] class_weights.yaml not found, using uniform weights")
        return torch.ones(NUM_CLASSES)
    with open(weights_path) as f:
        data = yaml.safe_load(f)
    cw = data.get("coarse_weights", {})
    weights = torch.tensor(
        [cw.get(cls, 1.0) for cls in COARSE_CLASSES],
        dtype=torch.float32
    )
    print(f"Class weights: { {c: round(float(w),3) for c,w in zip(COARSE_CLASSES, weights)} }")
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Builds a timm classifier with a replaced head for num_classes.
    """
    model_str, pretrained = CLASSIFIER_REGISTRY[name]
    model = timm.create_model(
        model_str,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    print(f"  Built {name} ({model_str})")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int,
    lr:           float,
    device:       torch.device,
    class_weights:torch.Tensor,
    patience:     int,
    output_dir:   Path,
) -> dict:

    model = model.to(device)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc  = 0.0
    best_epoch    = 0
    no_improve    = 0
    history       = []

    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader,
                                  desc=f"  Epoch {epoch}/{epochs} train",
                                  leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()
        train_acc  = train_correct / train_total
        train_loss = train_loss    / train_total

        # ── val ───────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader,
                                      desc=f"  Epoch {epoch}/{epochs} val",
                                      leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs      = model(imgs)
                loss         = criterion(outputs, labels)
                val_loss    += loss.item() * imgs.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc  = val_correct / val_total
        val_loss = val_loss    / val_total

        labels_range = list(range(NUM_CLASSES))
        val_f1 = f1_score(all_labels, all_preds,
                          labels=labels_range,
                          average="macro", zero_division=0)

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc,  5),
            "val_loss":   round(val_loss,   5),
            "val_acc":    round(val_acc,    5),
            "val_f1":     round(val_f1,     5),
            "lr":         round(scheduler.get_last_lr()[0], 8),
        })

        # ── save best ─────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            no_improve   = 0
            torch.save(model.state_dict(),
                       weights_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:  # early stopping
                print(f"  Early stopping at epoch {epoch} "
                      f"(best epoch {best_epoch}, "
                      f"best val_acc={best_val_acc:.4f})")
                break


    # save training history CSV
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    return {
        "best_epoch":   best_epoch,
        "best_val_acc": round(best_val_acc, 5),
        "stopped_early": no_improve >= patience,
        "epochs_trained": epoch,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(
    model:       nn.Module,
    test_loader: DataLoader,
    device:      torch.device,
    output_dir:  Path,
    classifier_name: str,
) -> dict:

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="  Test inference"):
            imgs   = imgs.to(device)
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    labels_range = list(range(NUM_CLASSES))

    # per-class AP
    per_class_ap = {}
    for c in range(NUM_CLASSES):
        gt_binary = (all_labels == c).astype(int)
        if gt_binary.sum() == 0:
            per_class_ap[COARSE_CLASSES[c]] = 0.0
            continue
        per_class_ap[COARSE_CLASSES[c]] = round(
            float(average_precision_score(gt_binary, all_probs[:, c])), 5
        )

    # latency
    dummy     = torch.zeros(1, 3, CROP_SIZE, CROP_SIZE).to(device)
    for _ in range(10):
        model(dummy)
    t0 = time.perf_counter()
    for _ in range(100):
        model(dummy)
    latency_ms = (time.perf_counter() - t0) / 100 * 1000

    metrics = {
        "classifier":    classifier_name,
        "accuracy":      round(float((all_preds == all_labels).mean()), 5),
        "precision":     round(float(precision_score(all_labels, all_preds,
                                labels=labels_range, average="macro",
                                zero_division=0)), 5),
        "recall":        round(float(recall_score(all_labels, all_preds,
                                labels=labels_range, average="macro",
                                zero_division=0)), 5),
        "f1":            round(float(f1_score(all_labels, all_preds,
                                labels=labels_range, average="macro",
                                zero_division=0)), 5),
        "latency_ms":    round(latency_ms, 3),
    }
    metrics.update({f"AP_{cls}": v for cls, v in per_class_ap.items()})

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=labels_range)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=COARSE_CLASSES).plot(
        ax=ax, colorbar=True, xticks_rotation=45
    )
    ax.set_title(f"Path B — {classifier_name} — test confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{classifier_name}.png", dpi=150)
    plt.close()

    # save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  [{classifier_name}] "
          f"accuracy={metrics['accuracy']:.4f}  "
          f"f1={metrics['f1']:.4f}  "
          f"latency={latency_ms:.2f}ms")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(output_dir: Path):
    records = []
    for p in sorted(output_dir.glob("*/metrics.json")):
        with open(p) as f:
            m = json.load(f)
        m["classifier"] = p.parent.name
        records.append(m)
    if not records:
        print("No metrics.json files found.")
        return
    df = pd.DataFrame(records).set_index("classifier")
    df = df.sort_values("accuracy", ascending=False)
    print("\n" + "="*70)
    print("PATH B  —  CLASSIFIER BENCHMARK SUMMARY")
    print("="*70)
    print(df.to_string(float_format="{:.4f}".format))
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Path B: two-stage detect+classify")
    p.add_argument("--detector_weights", type=Path, required=True,
                   help="best.pt from best Path A model")
    p.add_argument("--crops_dir",        type=Path, required=True,
                   help="processed/ root containing train/val/test path_B crops")
    p.add_argument("--output",           type=Path,
                   default=Path("runs/path_B"))
    p.add_argument("--classifiers",      nargs="+",
                   default=["resnet50", "vit_b16_scratch", "vit_b16_imagenet"],
                   choices=list(CLASSIFIER_REGISTRY.keys()))
    p.add_argument("--epochs",           type=int,  default=50)
    p.add_argument("--batch",            type=int,  default=32)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--patience",         type=int,  default=10)
    p.add_argument("--device",           type=str,  default="0")
    p.add_argument("--mlflow_uri",       type=str,  default="./mlruns")
    p.add_argument("--summarize",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.summarize:
        print_summary(args.output)
        raise SystemExit(0)

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device : {device}")
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(int(args.device))
        print(f"GPU    : {dev.name}  VRAM: {dev.total_memory/1e9:.1f}GB")

    mlflow.set_tracking_uri(args.mlflow_uri)
    class_weights = get_class_weights(args.crops_dir)
    args.output.mkdir(parents=True, exist_ok=True)

    # build datasets once, shared across classifiers
    train_ds = CropDataset(args.crops_dir, "train",
                           transform=get_transforms("train"))
    val_ds   = CropDataset(args.crops_dir, "val",
                           transform=get_transforms("val"))
    test_ds  = CropDataset(args.crops_dir, "test",
                           transform=get_transforms("test"))

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=4,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=4,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch,
                              shuffle=False, num_workers=4,
                              pin_memory=True)

    all_metrics = {}

    for clf_name in args.classifiers:
        print(f"\n{'─'*60}")
        print(f"  Classifier : {clf_name}")
        print(f"{'─'*60}")

        run_dir = args.output / clf_name
        run_dir.mkdir(parents=True, exist_ok=True)

        model = build_classifier(clf_name, NUM_CLASSES)

        train_info = train_classifier(
            model         = model,
            train_loader  = train_loader,
            val_loader    = val_loader,
            epochs        = args.epochs,
            lr            = args.lr,
            device        = device,
            class_weights = class_weights,
            patience      = args.patience,
            output_dir    = run_dir,
        )

        # load best weights for test evaluation
        model.load_state_dict(
            torch.load(run_dir / "weights" / "best.pt",
                       map_location=device)
        )

        metrics = evaluate_on_test(
            model            = model,
            test_loader      = test_loader,
            device           = device,
            output_dir       = run_dir,
            classifier_name  = clf_name,
        )
        metrics.update(train_info)
        all_metrics[clf_name] = metrics

        mlflow.set_experiment("litter_path_B")
        with mlflow.start_run(run_name=clf_name):
            mlflow.log_params({
                "classifier": clf_name,
                "epochs":     args.epochs,
                "batch":      args.batch,
                "lr":         args.lr,
                "patience":   args.patience,
                "device":     args.device,
            })
            mlflow.log_metrics({
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float))
            })
            best_pt = run_dir / "weights" / "best.pt"
            if best_pt.exists():
                mlflow.log_artifact(str(best_pt), "weights")

        print(f"\n  [{clf_name}] "
              f"best_epoch={train_info['best_epoch']}  "
              f"val_acc={train_info['best_val_acc']:.4f}  "
              f"test_acc={metrics['accuracy']:.4f}  "
              f"f1={metrics['f1']:.4f}")

    print_summary(args.output)
