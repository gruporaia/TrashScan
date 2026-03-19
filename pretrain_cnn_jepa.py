"""
pretrain_cnn_jepa.py  —  Standalone CNN MIM-JEPA pretraining for Path C4.

Trains a ResNet-50 backbone using JEPA-style masked feature prediction
in latent space. No pixel reconstruction — predictions happen in the
CNN feature map space, matching the JEPA philosophy.

Saves resnet_jepa.pt for use as C4 backbone initialization.

Usage:
  conda activate litter_C
  python pretrain_cnn_jepa.py \
      --data_root ~/V-JEPA-2/mrosado/thrashscan/processed_4cls \
      --output    ~/V-JEPA-2/mrosado/thrashscan/runs/path_C/pretrain_cnn \
      --epochs    100 \
      --patience  7 \
      --batch     16 \
      --device    0
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import timm


def parse_args():
    p = argparse.ArgumentParser(
        description="CNN MIM-JEPA pretraining for Path C4")
    p.add_argument("--data_root",  type=Path, required=True,
                   help="processed_4cls root dir")
    p.add_argument("--output",     type=Path,
                   default=Path("runs/path_C/pretrain_cnn"))
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--patience",   type=int,   default=7)
    p.add_argument("--batch",      type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--mask_ratio", type=float, default=0.60)
    p.add_argument("--momentum",   type=float, default=0.996,
                   help="EMA momentum for target encoder update")
    p.add_argument("--device",     type=str,   default="0")
    return p.parse_args()


class FlatImageDataset(Dataset):
    """Loads all JPG images from a flat directory."""
    def __init__(self, img_dir: Path, transform):
        self.paths     = sorted(img_dir.glob("*.jpg"))
        self.transform = transform
        assert len(self.paths) > 0, f"No images found in {img_dir}"
        print(f"  Dataset: {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def build_models(device: torch.device):
    """
    Context encoder  : ResNet-50 feature extractor (no pooling, no head)
    Target encoder   : EMA copy — provides stable prediction targets
    Predictor        : 1x1 conv network operating on feature maps
    """
    encoder = timm.create_model(
        "resnet50",
        pretrained=False,
        num_classes=0,
        global_pool="",
    ).to(device)

    target_encoder = timm.create_model(
        "resnet50",
        pretrained=False,
        num_classes=0,
        global_pool="",
    ).to(device)
    target_encoder.load_state_dict(encoder.state_dict())
    for p in target_encoder.parameters():
        p.requires_grad = False

    # predictor: 2048 -> 2048 feature space prediction
    predictor = nn.Sequential(
        nn.Conv2d(2048, 2048, 1),
        nn.BatchNorm2d(2048),
        nn.GELU(),
        nn.Conv2d(2048, 2048, 1),
    ).to(device)

    n_enc  = sum(p.numel() for p in encoder.parameters())   / 1e6
    n_pred = sum(p.numel() for p in predictor.parameters()) / 1e6
    print(f"  Context encoder  : ResNet-50  {n_enc:.1f}M params")
    print(f"  Target encoder   : ResNet-50  {n_enc:.1f}M params (EMA, frozen)")
    print(f"  Predictor        : {n_pred:.1f}M params")

    return encoder, target_encoder, predictor


@torch.no_grad()
def update_ema(encoder, target_encoder, momentum: float):
    """Exponential moving average update of target encoder."""
    for p_enc, p_tgt in zip(encoder.parameters(),
                             target_encoder.parameters()):
        p_tgt.data = momentum * p_tgt.data + \
                     (1.0 - momentum) * p_enc.data


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        f"cuda:{args.device}"
        if str(args.device).isdigit() and torch.cuda.is_available()
        else "cpu"
    )
    print(f"\nDevice : {device}")
    if torch.cuda.is_available() and str(args.device).isdigit():
        prop = torch.cuda.get_device_properties(int(args.device))
        print(f"GPU    : {prop.name}  "
              f"VRAM: {prop.total_memory/1e9:.1f}GB")

    # ── dataset ───────────────────────────────────────────────────────────
    transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.1),
        T.ColorJitter(brightness=0.4, contrast=0.4,
                       saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225]),
    ])

    img_dir = args.data_root / "train" / "path_A" / "images"
    assert img_dir.exists(), f"Image dir not found: {img_dir}"

    dataset = FlatImageDataset(img_dir, transform)
    loader  = DataLoader(
        dataset,
        batch_size  = args.batch,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
    )

    # ── models ────────────────────────────────────────────────────────────
    encoder, target_encoder, predictor = build_models(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) +
        list(predictor.parameters()),
        lr           = args.lr,
        weight_decay = 0.05,
        betas        = (0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── training ──────────────────────────────────────────────────────────
    ckpt_path  = args.output / "resnet_jepa.pt"
    best_loss  = float("inf")
    no_improve = 0

    print(f"\nCNN MIM-JEPA Pretraining")
    print(f"  Images     : {len(dataset)}")
    print(f"  Epochs     : {args.epochs}  (patience={args.patience})")
    print(f"  Batch      : {args.batch}")
    print(f"  LR         : {args.lr}")
    print(f"  Mask ratio : {args.mask_ratio:.0%}")
    print(f"  EMA mom    : {args.momentum}")
    print(f"  Checkpoint : {ckpt_path}")
    print()

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        predictor.train()
        epoch_loss = 0.0
        n_batches  = 0

        for imgs in tqdm(loader,
                         desc=f"Epoch {epoch:3d}/{args.epochs}",
                         leave=False):
            imgs = imgs.to(device)
            B    = imgs.size(0)

            # target features — no gradient
            with torch.no_grad():
                target_feat = target_encoder(imgs)

            # context features
            context_feat = encoder(imgs)
            H = context_feat.shape[2]
            W = context_feat.shape[3]

            # create spatial block mask on feature map
            n_regions = H * W
            n_masked  = int(n_regions * args.mask_ratio)
            noise     = torch.rand(B, n_regions, device=device)
            mask_idx  = noise.argsort(dim=1)[:, :n_masked]
            mask      = torch.zeros(
                B, n_regions, dtype=torch.bool, device=device)
            mask.scatter_(1, mask_idx, True)
            mask_2d   = mask.view(B, 1, H, W).float()

            # zero out masked spatial regions
            masked_context = context_feat * (1.0 - mask_2d)

            # predict target features for masked regions
            predicted = predictor(masked_context)

            # JEPA loss: L2 in feature space, masked regions only
            diff = (predicted - target_feat.detach()) * mask_2d
            n_masked_elements = mask_2d.sum() * target_feat.shape[1]
            loss = (diff ** 2).sum() / (n_masked_elements + 1e-7)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) +
                list(predictor.parameters()), 1.0)
            optimizer.step()
            update_ema(encoder, target_encoder, args.momentum)

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now   = scheduler.get_last_lr()[0]

        # early stopping
        if avg_loss < best_loss:
            best_loss  = avg_loss
            no_improve = 0
            torch.save(encoder.state_dict(), ckpt_path)
            status = "✓ saved"
        else:
            no_improve += 1
            status = f"no improve {no_improve}/{args.patience}"

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={avg_loss:.6f}  "
              f"lr={lr_now:.2e}  "
              f"{status}")

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    print(f"\nPretraining complete.")
    print(f"  Best loss  : {best_loss:.6f}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"\nNext — run C4 label budget sweep:")
    print(f"  python train_path_C.py --variant C4 --label_budget_sweep \\")
    print(f"      --data_yaml   {args.data_root}/dataset_path_A.yaml \\")
    print(f"      --ssl_weights {ckpt_path} \\")
    print(f"      --output      runs/path_C \\")
    print(f"      --epochs      150 --batch 8 --device {args.device}")


if __name__ == "__main__":
    main()
