"""
train_path_B_cascade.py — Path B cascade classifier: coarse -> fine.

Pipeline:
  Stage 1: frozen YOLO detector supplies boxes (same as old Path B).
  Stage 2: coarse crop classifier predicts material: plastic/paper/metal/glass/other.
  Stage 3: one fine classifier per coarse material refines into subclasses inside that material.

Training uses GT crops from preprocess_cascade.py because fine labels are known there.
Evaluation with detector boxes is handled by evaluate_path_B_cascade_combined.py.

Usage:
  python train_path_B_cascade.py \
    --crops_dir /workspace/processed_5cls \
    --output /workspace/runs/path_B_cascade \
    --backbone resnet50 \
    --epochs 30 --batch 32 --device 0
"""

import argparse
import json
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

COARSE_CLASSES = ["plastic", "paper", "metal", "glass", "other"]
CROP_SIZE = 224

BACKBONES = {
    "resnet50": ("resnet50.a1_in1k", True),
    "vit_b16_scratch": ("vit_base_patch16_224", False),
    "vit_b16_imagenet": ("vit_base_patch16_224.augreg_in21k_ft_in1k", True),
}


def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomRotation(30),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class CropIndexDataset(Dataset):
    def __init__(self, data_root: Path, split: str, target: str, parent: str | None = None, fine_subset: list[int] | None = None, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples = []
        csv_path = self.data_root / split / "path_B" / "crop_index.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing crop index: {csv_path}. Run preprocess_cascade.py first.")
        df = pd.read_csv(csv_path)
        if parent is not None:
            df = df[df["coarse_label"] == parent]
        if fine_subset is not None:
            df = df[df["fine_idx"].isin(fine_subset)]
        if df.empty:
            print(f"  [{split}] empty dataset target={target} parent={parent}")
            return
            
        if target == "coarse":
            for _, r in df.iterrows():
                self.samples.append((self.data_root / str(r["crop_path"]), int(r["coarse_idx"])))
            self.class_ids = list(range(len(COARSE_CLASSES)))
            self.global_to_local = {i: i for i in self.class_ids}
            
        elif target == "fine":
            # CORREÇÃO: Usar o fine_subset fixo se fornecido, garantindo o mesmo gabarito para train/val/test
            if fine_subset is not None:
                class_ids = sorted(fine_subset)
            else:
                class_ids = sorted(int(x) for x in df["fine_idx"].unique())
                
            self.class_ids = class_ids
            self.global_to_local = {gid: i for i, gid in enumerate(class_ids)}
            
            for _, r in df.iterrows():
                gid = int(r["fine_idx"])
                self.samples.append((self.data_root / str(r["fine_crop_path"]), self.global_to_local[gid]))
                
        else:
            raise ValueError("target must be 'coarse' or 'fine'")
            
        print(f"  [{split}] target={target} parent={parent} crops={len(self.samples)} classes={len(self.class_ids)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = read_image(str(img_path)).float() / 255.0
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if img.shape[0] == 4:
            img = img[:3]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_maps(data_root: Path):
    path = data_root / "meta" / "fine_class_map.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run preprocess_cascade.py first.")
    data = json.loads(path.read_text())
    return data


def build_model(backbone: str, num_classes: int) -> nn.Module:
    model_name, pretrained = BACKBONES[backbone]
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Built {backbone} ({model_name}) num_classes={num_classes} pretrained={pretrained} params={n:.1f}M")
    return model


def class_weights_from_dataset(ds: CropIndexDataset, num_classes: int) -> torch.Tensor:
    if len(ds) == 0:
        return torch.ones(num_classes)
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, label in ds.samples:
        counts[label] += 1
    counts[counts == 0] = 1
    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(freq)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_one(model, train_loader, val_loader, epochs, lr, device, patience, output_dir, class_weights):
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = tr_total = tr_correct = 0
        for imgs, labels in tqdm(train_loader, desc=f"  epoch {epoch}/{epochs} train", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total += imgs.size(0)
        scheduler.step()

        model.eval()
        val_loss = val_total = val_correct = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"  epoch {epoch}/{epochs} val", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                preds = logits.argmax(1)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        tr_acc = tr_correct / max(1, tr_total)
        val_acc = val_correct / max(1, val_total)
        val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
        row = {
            "epoch": epoch,
            "train_loss": tr_loss / max(1, tr_total),
            "train_acc": tr_acc,
            "val_loss": val_loss / max(1, val_total),
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(f"  ep={epoch:03d} train_acc={tr_acc:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), weights_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stopping at epoch {epoch}; best_epoch={best_epoch} best_val_f1={best_f1:.4f}")
                break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    return {"best_epoch": best_epoch, "best_val_f1": round(best_f1, 5), "epochs_trained": epoch}


def evaluate_classifier(model, loader, device, class_names, output_dir, title):
    model.eval()
    y_true, y_pred = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"  test {title}"):
            logits = model(imgs.to(device))
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(labels.numpy())
    elapsed = time.perf_counter() - t0
    labels = list(range(len(class_names)))
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5) if y_true else 0.0,
        "precision_macro": round(float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 5) if y_true else 0.0,
        "recall_macro": round(float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 5) if y_true else 0.0,
        "f1_macro": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 5) if y_true else 0.0,
        "n_samples": len(y_true),
        "fps_crops": round(len(y_true) / max(elapsed, 1e-9), 2),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if y_true and len(class_names) <= 40:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.6), max(6, len(class_names) * 0.5)))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=True, xticks_rotation=45)
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    print(f"  {title}: acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f}")
    return metrics


def save_model_metadata(output_dir: Path, kind: str, backbone: str, class_ids: list[int], class_names: list[str], extra: dict | None = None):
    meta = {"kind": kind, "backbone": backbone, "class_ids": class_ids, "class_names": class_names}
    if extra:
        meta.update(extra)
    (output_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))


def make_loader(ds, batch, shuffle):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=4, pin_memory=torch.cuda.is_available())




def best_weight_path(run_dir: Path) -> Path:
    return run_dir / "weights" / "best.pt"


def load_matching_weights(model: nn.Module, weights_path: Path, device, *, strict: bool = False) -> None:
    """
    Loads weights into a model. With strict=False, layers with incompatible
    shapes are skipped, which is useful when reusing an old classifier backbone
    but changing the number of output classes.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if strict:
        model.load_state_dict(state, strict=True)
        print(f"  Loaded strict weights: {weights_path}")
        return
    current = model.state_dict()
    compatible = {}
    skipped = []
    for k, v in state.items():
        key = k.replace("module.", "")
        if key in current and current[key].shape == v.shape:
            compatible[key] = v
        else:
            skipped.append(k)
    current.update(compatible)
    model.load_state_dict(current, strict=False)
    print(f"  Loaded {len(compatible)} compatible tensors from {weights_path}")
    if skipped:
        print(f"  Skipped {len(skipped)} incompatible tensors, usually classifier head layers")


def copy_weights_file(src: Path, dst: Path) -> bool:
    src = Path(src)
    if not src.exists():
        print(f"  [warn] source weights not found: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  Registered existing weights: {src} -> {dst}")
    return True


def maybe_copy_metadata(src_run_dir: Path, dst_run_dir: Path) -> None:
    for name in ["model_meta.json", "metrics.json", "history.csv", "confusion_matrix.png"]:
        src = src_run_dir / name
        if src.exists():
            dst = dst_run_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def resolve_fine_parent_dir(fine_dir: Path, parent: str) -> Path | None:
    """Accepts either <root>/fine/<parent> or <root>/<parent>."""
    if fine_dir is None:
        return None
    fine_dir = Path(fine_dir)
    candidates = [fine_dir / parent, fine_dir / "fine" / parent]
    for c in candidates:
        if best_weight_path(c).exists():
            return c
    return None


def ensure_coarse_available(args) -> bool:
    """Makes args.output/coarse/weights/best.pt available when skipping/reusing."""
    run_dir = args.output / "coarse"
    dst = best_weight_path(run_dir)
    if dst.exists():
        print(f"  Coarse weights already available: {dst}")
        return True
    if args.coarse_weights:
        return copy_weights_file(args.coarse_weights, dst)
    print(f"  [warn] coarse weights missing: {dst}")
    print("  Provide --coarse_weights or train coarse before evaluation.")
    return False


def ensure_fine_available(args, parent: str) -> bool:
    """Makes args.output/fine/<parent>/weights/best.pt available from --fine_dir."""
    run_dir = args.output / "fine" / parent
    dst = best_weight_path(run_dir)
    if dst.exists():
        print(f"  Fine weights already available for {parent}: {dst}")
        return True
    src_run_dir = resolve_fine_parent_dir(args.fine_dir, parent) if args.fine_dir else None
    if src_run_dir is None:
        print(f"  [warn] fine weights missing for {parent}: {dst}")
        return False
    copied = copy_weights_file(best_weight_path(src_run_dir), dst)
    if copied:
        maybe_copy_metadata(src_run_dir, run_dir)
    return copied

def train_coarse(args, device):
    run_dir = args.output / "coarse"
    existing_best = best_weight_path(run_dir)

    if args.skip_coarse:
        print("\n[skip] coarse stage requested via --skip_coarse")
        ensure_coarse_available(args)
        return {}

    train_ds = CropIndexDataset(args.crops_dir, "train", target="coarse", transform=get_transforms("train"))
    val_ds = CropIndexDataset(args.crops_dir, "val", target="coarse", transform=get_transforms("val"))
    test_ds = CropIndexDataset(args.crops_dir, "test", target="coarse", transform=get_transforms("test"))

    model = build_model(args.backbone, len(COARSE_CLASSES))

    if args.resume and existing_best.exists():
        print(f"\n[resume] coarse already trained; skipping training and loading {existing_best}")
        load_matching_weights(model, existing_best, device, strict=True)
        metrics = evaluate_classifier(model.to(device), make_loader(test_ds, args.batch, False), device, COARSE_CLASSES, run_dir, "Path B cascade — coarse classifier")
        save_model_metadata(run_dir, "coarse", args.backbone, list(range(len(COARSE_CLASSES))), COARSE_CLASSES)
        return metrics

    if args.coarse_weights:
        print(f"\nInitializing coarse model from --coarse_weights={args.coarse_weights}")
        load_matching_weights(model, args.coarse_weights, device, strict=False)

    info = train_one(
        model,
        make_loader(train_ds, args.batch, True),
        make_loader(val_ds, args.batch, False),
        args.epochs,
        args.lr,
        device,
        args.patience,
        run_dir,
        class_weights_from_dataset(train_ds, len(COARSE_CLASSES)),
    )
    model.load_state_dict(torch.load(existing_best, map_location=device))
    metrics = evaluate_classifier(model.to(device), make_loader(test_ds, args.batch, False), device, COARSE_CLASSES, run_dir, "Path B cascade — coarse classifier")
    metrics.update(info)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_model_metadata(run_dir, "coarse", args.backbone, list(range(len(COARSE_CLASSES))), COARSE_CLASSES)
    return metrics

def train_fine(args, device, maps):
    fine_classes = maps["fine_classes"]
    coarse_to_fine = {c: [int(i) for i in ids] for c, ids in maps["coarse_to_fine_indices"].items()}
    all_metrics = {}

    if args.skip_fine:
        print("\n[skip] fine stage requested via --skip_fine")
        for parent in COARSE_CLASSES:
            ensure_fine_available(args, parent)
        return all_metrics

    for parent in COARSE_CLASSES:
        fine_ids = sorted(coarse_to_fine.get(parent, []))
        if len(fine_ids) < 2:
            print(f"\n[skip] fine classifier for {parent}: only {len(fine_ids)} fine class")
            continue

        run_dir = args.output / "fine" / parent
        existing_best = best_weight_path(run_dir)
        src_run_dir = resolve_fine_parent_dir(args.fine_dir, parent) if args.fine_dir else None

        print(f"\n{'─'*60}\nFine classifier parent={parent} classes={len(fine_ids)}\n{'─'*60}")

        train_ds = CropIndexDataset(args.crops_dir, "train", target="fine", parent=parent, fine_subset=fine_ids, transform=get_transforms("train"))
        val_ds = CropIndexDataset(args.crops_dir, "val", target="fine", parent=parent, fine_subset=fine_ids, transform=get_transforms("val"))
        test_ds = CropIndexDataset(args.crops_dir, "test", target="fine", parent=parent, fine_subset=fine_ids, transform=get_transforms("test"))
        if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
            print(f"[skip] {parent}: empty train/val/test after filtering")
            continue

        class_names = [fine_classes[i] for i in train_ds.class_ids]
        model = build_model(args.backbone, len(train_ds.class_ids))

        if args.resume and existing_best.exists():
            print(f"[resume] fine {parent} already trained; skipping training and loading {existing_best}")
            load_matching_weights(model, existing_best, device, strict=True)
            metrics = evaluate_classifier(model.to(device), make_loader(test_ds, args.batch, False), device, class_names, run_dir, f"Path B cascade — fine {parent}")
            save_model_metadata(run_dir, "fine", args.backbone, train_ds.class_ids, class_names, {"parent_coarse": parent, "parent_coarse_idx": COARSE_CLASSES.index(parent)})
            all_metrics[parent] = metrics
            del model
            torch.cuda.empty_cache()
            continue

        if src_run_dir is not None and not existing_best.exists():
            print(f"Initializing fine {parent} from --fine_dir={src_run_dir}")
            load_matching_weights(model, best_weight_path(src_run_dir), device, strict=False)

        info = train_one(
            model,
            make_loader(train_ds, args.batch, True),
            make_loader(val_ds, args.batch, False),
            args.epochs,
            args.lr,
            device,
            args.patience,
            run_dir,
            class_weights_from_dataset(train_ds, len(train_ds.class_ids)),
        )
        model.load_state_dict(torch.load(existing_best, map_location=device))
        metrics = evaluate_classifier(model.to(device), make_loader(test_ds, args.batch, False), device, class_names, run_dir, f"Path B cascade — fine {parent}")
        metrics.update(info)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        save_model_metadata(run_dir, "fine", args.backbone, train_ds.class_ids, class_names, {"parent_coarse": parent, "parent_coarse_idx": COARSE_CLASSES.index(parent)})
        all_metrics[parent] = metrics
        del model
        torch.cuda.empty_cache()
    return all_metrics

def summarize(output: Path):
    records = []
    for p in output.rglob("metrics.json"):
        rel = p.parent.relative_to(output)
        m = json.loads(p.read_text())
        m["model"] = str(rel)
        records.append(m)
    if not records:
        print("No metrics found.")
        return
    df = pd.DataFrame(records).set_index("model").sort_values("f1_macro", ascending=False)
    print("\nPATH B CASCADE SUMMARY")
    print(df.to_string(float_format="{:.4f}"))


def parse_args():
    p = argparse.ArgumentParser(description="Train Path B coarse->fine cascade classifiers")
    p.add_argument("--crops_dir", type=Path, required=True, help="processed root created by preprocess_cascade.py")
    p.add_argument("--output", type=Path, default=Path("runs/path_B_cascade"))
    p.add_argument("--backbone", choices=list(BACKBONES.keys()), default="resnet50")
    p.add_argument("--stages", nargs="+", choices=["coarse", "fine", "all"], default=["all"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--skip_coarse", action="store_true", help="Do not train the coarse classifier. Optionally register --coarse_weights into output/coarse/weights/best.pt.")
    p.add_argument("--skip_fine", action="store_true", help="Do not train fine classifiers. Optionally register weights from --fine_dir.")
    p.add_argument("--coarse_weights", type=Path, default=None, help="Existing coarse best.pt to initialize or register when --skip_coarse is used.")
    p.add_argument("--fine_dir", type=Path, default=None, help="Existing fine classifiers root. Accepts either <root>/fine/<parent>/weights/best.pt or <root>/<parent>/weights/best.pt.")
    p.add_argument("--resume", action="store_true", help="Skip training stages whose output weights already exist, then evaluate/summarize them.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.output)
        raise SystemExit(0)

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    maps = load_maps(args.crops_dir)
    stages = {"coarse", "fine"} if "all" in args.stages else set(args.stages)

    if "coarse" in stages:
        print("\nTraining coarse classifier")
        train_coarse(args, device)
    if "fine" in stages:
        print("\nTraining fine classifiers")
        train_fine(args, device, maps)
    summarize(args.output)