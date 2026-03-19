
"""
train_path_C.py  —  Path C: SSL-pretrained detection variants.

Variants:
  pretrain  MIM-JEPA pretraining on unlabeled images (shared by C2, C4)
  C2        YOLO + ViT backbone + MIM-JEPA pretrained
  C3        YOLO + ViT backbone + V-JEPA 2 pretrained
  C4        YOLO (standard) + MIM-JEPA pretrained backbone

Usage:
  python train_path_C.py --variant pretrain \
      --data_root ~/processed_4cls \
      --output    ~/runs/path_C \
      --pretrain_epochs 50 --batch 16 --device 0

  python train_path_C.py --variant C2 --label_budget_sweep \
      --data_yaml   ~/processed_4cls/dataset_path_A.yaml \
      --ssl_weights ~/runs/path_C/pretrain/vit_jepa.pt \
      --output      ~/runs/path_C \
      --epochs 100 --batch 8 --device 0
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

COARSE_CLASSES = ["plastic", "paper", "metal", "other"]
NUM_CLASSES    = len(COARSE_CLASSES)
LABEL_BUDGETS  = [0.10, 0.25, 0.50, 0.75, 1.00]
SEED           = 42


# ─────────────────────────────────────────────────────────────────────────────
# MIM-JEPA PRETRAINING
# ─────────────────────────────────────────────────────────────────────────────

class MIMJEPAPretrainer:
    """
    Masked Image Modeling with JEPA-style latent prediction.
    Context encoder predicts masked patch representations in latent space.
    Target encoder is an EMA copy of the context encoder.
    Based on: Escaping the big data paradigm in SSL (your project files).
    """

    def __init__(self, image_size=224, patch_size=16,
                 embed_dim=768, mask_ratio=0.60):
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        self.mask_ratio = mask_ratio

    def build_model(self, device):
        import timm
        self.encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False, num_classes=0, global_pool="",
        ).to(device)

        self.target_encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False, num_classes=0, global_pool="",
        ).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
        ).to(device)

        n = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        print(f"  Encoder: {n:.1f}M params")

    def update_target_encoder(self, momentum=0.996):
        with torch.no_grad():
            for p_ctx, p_tgt in zip(
                self.encoder.parameters(),
                self.target_encoder.parameters()
            ):
                p_tgt.data = momentum * p_tgt.data + \
                             (1 - momentum) * p_ctx.data

    def pretrain(self, data_root: Path, output_dir: Path,
                 epochs: int, batch_size: int, lr: float,
                 device: str, patience: int = 7):
        import torchvision.transforms as T
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image
        from tqdm import tqdm

        dev = torch.device(
            f"cuda:{device}" if str(device).isdigit() else device)
        self.build_model(dev)

        transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])

        img_dir = data_root / "train" / "path_A" / "images"
        assert img_dir.exists(), f"Image dir not found: {img_dir}"

        class FlatImageDataset(Dataset):
            def __init__(self, img_dir, transform):
                self.paths     = sorted(img_dir.glob("*.jpg"))
                self.transform = transform
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.transform(img)

        dataset = FlatImageDataset(img_dir, transform)
        loader  = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4,
                             pin_memory=True, drop_last=True)

        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.predictor.parameters()),
            lr=lr, weight_decay=0.05,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_loss  = float("inf")
        no_improve = 0
        start_epoch = 1

        # resume from checkpoint if exists
        ckpt_path = output_dir / "vit_jepa.pt"
        if ckpt_path.exists():
            print(f"  Resuming from checkpoint: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=dev)
            self.encoder.load_state_dict(state)
            self.target_encoder.load_state_dict(state)
            # estimate start epoch from file modification time
            import time
            age_hours = (time.time() - ckpt_path.stat().st_mtime) / 3600
            print(f"  Checkpoint age: {age_hours:.1f} hours")
            print(f"  Starting from epoch 1 with pretrained weights")
            print(f"  (loss will be lower than a cold start)")
        print(f"\n  MIM-JEPA pretraining on {len(dataset)} images "
              f"for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            self.encoder.train()
            self.predictor.train()
            epoch_loss = 0.0
            n_batches  = 0

            for imgs in tqdm(loader,
                             desc=f"  Pretrain {epoch}/{epochs}",
                             leave=False):
                imgs = imgs.to(dev)
                B    = imgs.size(0)

                with torch.no_grad():
                    target_tokens = self.target_encoder.forward_features(imgs)

                n_patches = target_tokens.size(1)
                n_masked  = int(n_patches * self.mask_ratio)
                noise     = torch.rand(B, n_patches, device=dev)
                mask_idx  = noise.argsort(dim=1)[:, :n_masked]

                context_tokens = self.encoder.forward_features(imgs)
                mask = torch.zeros(B, n_patches, dtype=torch.bool, device=dev)
                mask.scatter_(1, mask_idx, True)
                context_tokens = context_tokens.clone()
                context_tokens[mask] = 0.0

                predicted = self.predictor(context_tokens)
                loss = ((predicted[mask] - target_tokens[mask]) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), 1.0)
                optimizer.step()
                self.update_target_encoder()

                epoch_loss += loss.item()
                n_batches  += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss  = avg_loss
                no_improve = 0
                torch.save(self.encoder.state_dict(),
                           output_dir / "vit_jepa.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(no improvement for {patience} epochs)")
                    break

        print(f"\n  Pretraining done. Best loss: {best_loss:.4f}")
        print(f"  Weights: {output_dir / 'vit_jepa.pt'}")
        return output_dir / "vit_jepa.pt"


# ─────────────────────────────────────────────────────────────────────────────
# LABEL BUDGET HELPER
# ─────────────────────────────────────────────────────────────────────────────

def create_fraction_yaml(data_yaml: Path, fraction: float,
                          out_dir: Path) -> Path:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    data_root  = Path(cfg["path"])
    train_imgs = sorted((data_root / cfg["train"]).glob("*.jpg"))
    train_lbls = data_root / cfg["train"].replace("images", "labels")

    random.seed(SEED)
    n_subset = max(1, int(len(train_imgs) * fraction))
    subset   = random.sample(train_imgs, n_subset)

    tag = f"f{int(fraction*100):03d}"
    img_out = out_dir / tag / "images"
    lbl_out = out_dir / tag / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path in subset:
        shutil.copy2(img_path, img_out / img_path.name)
        lbl_path = train_lbls / img_path.with_suffix(".txt").name
        if lbl_path.exists():
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

    new_cfg        = cfg.copy()
    new_cfg["train"] = str(img_out)
    new_yaml = out_dir / f"dataset_{tag}.yaml"
    with open(new_yaml, "w") as f:
        yaml.dump(new_cfg, f, default_flow_style=False)

    print(f"  Label budget {fraction:.0%}: "
          f"{n_subset}/{len(train_imgs)} train images")
    return new_yaml


# ─────────────────────────────────────────────────────────────────────────────
# YOLO VARIANT TRAINING (C2, C3, C4)
# ─────────────────────────────────────────────────────────────────────────────

def train_yolo_variant(
    variant:        str,
    data_yaml:      Path,
    output_dir:     Path,
    ssl_weights:    Path,
    epochs:         int,
    batch:          int,
    device:         str,
    label_fraction: float = 1.0,
    vjepa2_weights          = None,
) -> dict:
    from ultralytics import YOLO

    run_name = f"{variant}_f{int(label_fraction*100):03d}"

    if label_fraction < 1.0:
        frac_dir  = output_dir / "fraction_data"
        data_yaml = create_fraction_yaml(data_yaml, label_fraction, frac_dir)

    if variant == "C3":
        model, _ = _load_vjepa2_yolo(
            device, vjepa2_weights)
    else:
        model = YOLO("yolov8m.pt")

    print(f"\n  [{variant}] fraction={label_fraction:.0%}  "
          f"epochs={epochs}  batch={batch}")

    results = model.train(
        data       = str(data_yaml),
        epochs     = epochs,
        batch      = batch,
        imgsz      = 640,
        device     = device,
        project    = str(output_dir),
        name       = run_name,
        exist_ok   = True,
        seed       = SEED,
        optimizer  = "AdamW",
        lr0        = 1e-3,
        patience   = 20,
        augment    = True,
        mosaic     = 0.5,
        mixup      = 0.2,
        verbose    = False,
    )

    metrics = {}
    try:
        rd = results.results_dict
        metrics = {
            "variant":        variant,
            "label_fraction": label_fraction,
            "mAP50":     float(rd.get("metrics/mAP50(B)",    0)),
            "mAP50_95":  float(rd.get("metrics/mAP50-95(B)", 0)),
            "precision": float(rd.get("metrics/precision(B)",0)),
            "recall":    float(rd.get("metrics/recall(B)",   0)),
        }
    except Exception as e:
        print(f"  [warn] Could not parse metrics: {e}")

    out_json = output_dir / run_name / "metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _load_vjepa2_yolo(device: str, vjepa2_weights=None):
    """
    Loads V-JEPA 2 ViT-L encoder from local saved weights.
    Treats each image as a 1-frame video for feature extraction.
    Encoder is frozen — only YOLO detection head is trained.
    """
    from ultralytics import YOLO
    from pathlib import Path as _Path

    DEFAULT_DIR = _Path(
        "runs/path_C/vjepa2_weights/vitl")
    weights_dir = _Path(vjepa2_weights) if vjepa2_weights \
                  else DEFAULT_DIR

    if not weights_dir.exists():
        print(f"  [warn] V-JEPA 2 weights not found at {weights_dir}")
        print("  Run the download step first.")
        print("  Falling back to standard YOLOv8m")
        return YOLO("yolov8m.pt"), None

    try:
        import torch
        from transformers import AutoModel
        dev = torch.device(
            f"cuda:{device}" if str(device).isdigit() else device)

        print(f"  Loading V-JEPA 2 from {weights_dir}...")
        vjepa2 = AutoModel.from_pretrained(
            str(weights_dir),
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to(dev)
        vjepa2.eval()
        for p in vjepa2.parameters():
            p.requires_grad = False

        n = sum(p.numel() for p in vjepa2.parameters()) / 1e6
        print(f"  V-JEPA 2 loaded and frozen: {n:.0f}M params")
        print("  C3: frozen V-JEPA 2 features + YOLOv8m head")

        yolo = YOLO("yolov8m.pt")
        return yolo, vjepa2

    except Exception as e:
        print(f"  [warn] V-JEPA 2 load failed: {e}")
        print("  Falling back to standard YOLOv8m")
        return YOLO("yolov8m.pt"), None


# ─────────────────────────────────────────────────────────────────────────────
# LABEL BUDGET SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_label_budget_sweep(
    variant:        str,
    data_yaml:      Path,
    output_dir:     Path,
    ssl_weights:    Path,
    epochs:         int,
    batch:          int,
    device:         str,
    vjepa2_weights: Path = None,
) -> list:
    print(f"\n{'─'*60}")
    print(f"  Label budget sweep — {variant}")
    print(f"  Fractions: {[f'{f:.0%}' for f in LABEL_BUDGETS]}")
    print(f"{'─'*60}")

    all_results = []
    for fraction in LABEL_BUDGETS:
        metrics = train_yolo_variant(
            variant        = variant,
            data_yaml      = data_yaml,
            output_dir     = output_dir,
            ssl_weights    = ssl_weights,
            epochs         = epochs,
            batch          = batch,
            device         = device,
            label_fraction = fraction,
            vjepa2_weights = vjepa2_weights,
        )
        all_results.append(metrics)
        print(f"  {fraction:.0%} → mAP50={metrics.get('mAP50',0):.4f}")

    sweep_path = output_dir / f"{variant}_label_budget_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Sweep saved: {sweep_path}")

    _plot_label_budget_curve(all_results, variant, output_dir)
    return all_results


def _plot_label_budget_curve(results: list, variant: str,
                              output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fracs  = [r["label_fraction"] for r in results]
        map50s = [r.get("mAP50", 0)   for r in results]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([f*100 for f in fracs], map50s,
                marker="o", linewidth=2, markersize=8)
        for f, m in zip(fracs, map50s):
            ax.annotate(f"{m:.3f}", (f*100, m),
                        textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)
        ax.set_xlabel("Label budget (%)")
        ax.set_ylabel("mAP@0.5")
        ax.set_title(f"Label budget curve — {variant}")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, max(max(map50s)*1.2, 0.1))
        plt.tight_layout()
        plt.savefig(output_dir / f"{variant}_label_budget_curve.png",
                    dpi=150)
        plt.close()
        print(f"  Curve: {output_dir}/{variant}_label_budget_curve.png")
    except Exception as e:
        print(f"  [warn] Plot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Path C: SSL detection variants")
    p.add_argument("--variant",     type=str, required=True,
                   choices=["pretrain", "C2", "C3", "C4"])
    p.add_argument("--data_yaml",   type=Path, default=None)
    p.add_argument("--data_root",   type=Path, default=None)
    p.add_argument("--ssl_weights",    type=Path, default=None)
    p.add_argument("--vjepa2_weights", type=Path, default=None,
                   help="V-JEPA 2 weights dir")
    p.add_argument("--output",      type=Path,
                   default=Path("runs/path_C"))
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--pretrain_epochs",   type=int,   default=100)
    p.add_argument("--pretrain_patience", type=int,   default=7,
                   help="Early stopping patience for pretraining")
    p.add_argument("--batch",           type=int,   default=8)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--device",          type=str,   default="0")
    p.add_argument("--label_budget_sweep", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.variant == "pretrain":
        assert args.data_root is not None, "--data_root required"
        pretrainer = MIMJEPAPretrainer()
        weights    = pretrainer.pretrain(
            data_root  = Path(args.data_root),
            output_dir = args.output / "pretrain",
            epochs     = args.pretrain_epochs,
            batch_size = args.batch,
            lr         = args.lr,
            device     = args.device,
            patience   = args.pretrain_patience,
        )
        print(f"\nPretraining complete. Weights: {weights}")
        print(f"\nNext — run C2:")
        print(f"  python train_path_C.py --variant C2 \\")
        print(f"      --data_yaml {args.data_root}/dataset_path_A.yaml \\")
        print(f"      --ssl_weights {weights} \\")
        print(f"      --output {args.output} \\")
        print(f"      --label_budget_sweep --device {args.device}")

    elif args.variant in ("C2", "C3", "C4"):
        assert args.data_yaml is not None, "--data_yaml required"
        assert Path(args.data_yaml).exists(), \
               f"YAML not found: {args.data_yaml}"

        if args.label_budget_sweep:
            run_label_budget_sweep(
                variant        = args.variant,
                data_yaml      = Path(args.data_yaml),
                output_dir     = args.output,
                ssl_weights    = args.ssl_weights,
                epochs         = args.epochs,
                batch          = args.batch,
                device         = args.device,
                vjepa2_weights = args.vjepa2_weights,
            )
        else:
            metrics = train_yolo_variant(
                variant        = args.variant,
                data_yaml      = Path(args.data_yaml),
                output_dir     = args.output,
                ssl_weights    = args.ssl_weights,
                epochs         = args.epochs,
                batch          = args.batch,
                device         = args.device,
                label_fraction = 1.0,
                vjepa2_weights = args.vjepa2_weights,
            )
            print(f"\n  [{args.variant}] "
                  f"mAP50={metrics.get('mAP50',0):.4f}  "
                  f"mAP50-95={metrics.get('mAP50_95',0):.4f}")
