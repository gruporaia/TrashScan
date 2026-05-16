"""
train_path_A.py  —  Path A: YOLO family detection baseline.

Trains each model in the grid under identical conditions and logs
all results to MLflow for direct comparison.

Grid (default):
  yolov8n · yolov8s · yolov9s · yolov11n

Usage:
  python train_path_A.py \
      --data    /path/to/processed/dataset_path_A.yaml \
      --output  /path/to/runs/path_A \
      --models  yolov8n yolov9s \
      --epochs  100 \
      --batch   16 \
      --device  0

Summarize existing runs without retraining:
  python train_path_A.py --summarize --output /path/to/runs/path_A
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    # YOLOv5
    "yolov5su":  "yolov5su.pt",
    # YOLOv8 family
    "yolov8n":   "yolov8n.pt",
    "yolov8s":   "yolov8s.pt",
    "yolov8m":   "yolov8m.pt",
    "yolov8l":   "yolov8l.pt",
    # YOLOv9 family
    "yolov9s":   "yolov9s.pt",
    "yolov9c":   "yolov9c.pt",
    "yolov9e":   "yolov9e.pt",
    # YOLOv10 family (NMS-free)
    "yolov10n":  "yolov10n.pt",
    "yolov10s":  "yolov10s.pt",
    "yolov10m":  "yolov10m.pt",
    # YOLOv11 family
    "yolov11n":  "yolo11n.pt",
    "yolov11s":  "yolo11s.pt",
    "yolov11m":  "yolo11m.pt",
    "yolov11m_o2o":  "yolo11m.pt",  # sem NMS
    "yolov11l":  "yolo11l.pt",
    "yolov11x":  "yolo11x.pt",
    # RT-DETR (transformer-based, no NMS)
    "rtdetr-l":  "rtdetr-l.pt",
    "rtdetr-x":  "rtdetr-x.pt",
}

COARSE_CLASSES = ["plastic", "paper", "metal", "glass",  "other"]

# ─────────────────────────────────────────────────────────────────────────────
# O2O COMPONENTS  (YOLOv11m NMS-free)
# ─────────────────────────────────────────────────────────────────────────────

class HungarianMatcher(nn.Module):
    """Assigns exactly 1 prediction per ground-truth (one-to-one)."""

    def __init__(self, cost_cls=1.0, cost_bbox=5.0):
        super().__init__()
        self.cost_cls  = cost_cls
        self.cost_bbox = cost_bbox

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, gt_labels, gt_boxes):
        M = gt_labels.shape[0]
        if M == 0:
            empty = torch.zeros(0, dtype=torch.long, device=pred_logits.device)
            return empty, empty

        probs     = pred_logits.sigmoid()
        cost_cls  = -probs[:, gt_labels]
        cost_bbox = torch.cdist(pred_boxes.float(), gt_boxes.float(), p=1)

        C = self.cost_cls * cost_cls + self.cost_bbox * cost_bbox
        row, col = linear_sum_assignment(C.cpu().numpy())
        return (
            torch.as_tensor(row, dtype=torch.long, device=pred_logits.device),
            torch.as_tensor(col, dtype=torch.long, device=pred_logits.device),
        )


class DualAssignmentLoss(nn.Module):
    """
    Combines the original TAL loss (rich gradient signal, one-to-many)
    with a Hungarian one-to-one loss (eliminates duplicates at inference).
    """

    def __init__(self, base_loss, matcher, weight_o2o=1.0):
        super().__init__()
        self.base_loss  = base_loss
        self.matcher    = matcher
        self.weight_o2o = weight_o2o
        self.bce        = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, batch):
        # ── TAL loss (one-to-many) ────────────────────────────────────────
        loss_o2m, loss_items = self.base_loss(preds, batch)

        # ── Hungarian loss (one-to-one) per image ────────────────────────
        loss_o2o = torch.tensor(0.0, device=loss_o2m.device)
        try:
            pred_raw = preds[1] if isinstance(preds, (list, tuple)) else preds
            bs = int(batch["batch_idx"].max().item()) + 1

            for i in range(bs):
                mask = batch["batch_idx"] == i
                if mask.sum() == 0:
                    continue

                gt_cls = batch["cls"][mask].long().squeeze(-1)
                gt_box = batch["bboxes"][mask]

                p = pred_raw[1][i] if isinstance(pred_raw, (list, tuple)) else pred_raw[i]
                pred_boxes  = p[:, :4]
                pred_logits = p[:, 4:]

                p_idx, g_idx = self.matcher(pred_logits, pred_boxes, gt_cls, gt_box)
                if p_idx.numel() == 0:
                    continue

                target_cls = torch.zeros_like(pred_logits[p_idx])
                target_cls[range(len(g_idx)), gt_cls[g_idx]] = 1.0
                loss_o2o = loss_o2o + self.bce(pred_logits[p_idx], target_cls).mean()

        except Exception as e:
            print(f"[warn] o2o loss skipped: {e}")

        return loss_o2m + self.weight_o2o * loss_o2o, loss_items


class O2OTrainer(DetectionTrainer):
    def init_criterion(self):
        base    = v8DetectionLoss(self.model)
        matcher = HungarianMatcher(cost_cls=1.0, cost_bbox=5.0)
        return DualAssignmentLoss(base, matcher, weight_o2o=1.0)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_class_weights(data_yaml: Path) -> list:
    """
    Load per-class weights from meta/class_weights.yaml.
    Returns a list ordered by COARSE_CLASSES index.
    Falls back to uniform weights if file not found.
    """
    weights_path = data_yaml.parent / "meta" / "class_weights.yaml"
    if not weights_path.exists():
        print(f"[warn] class_weights.yaml not found at {weights_path}, using uniform weights")
        return [1.0] * len(COARSE_CLASSES)
    with open(weights_path) as f:
        data = yaml.safe_load(f)
    cw = data.get("coarse_weights", {})
    weights = [cw.get(cls, 1.0) for cls in COARSE_CLASSES]
    print(f"Loaded class weights: { {c: round(w,3) for c,w in zip(COARSE_CLASSES, weights)} }")
    return weights


def measure_latency(model: YOLO, imgsz: int, device: str, n_runs: int = 100) -> float:
    """Average inference latency in ms over n_runs passes after 10-run warmup."""
    device_obj = torch.device(f'cuda:{device}' if str(device).isdigit() else device)
    dummy = torch.zeros(1, 3, imgsz, imgsz).to(device_obj)
    for _ in range(10):
        model.predict(source=dummy, verbose=False)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.predict(source=dummy, verbose=False)
    return (time.perf_counter() - t0) / n_runs * 1000


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_single(
    model_key:    str,
    data_yaml:    Path,
    output_dir:   Path,
    epochs:       int,
    batch:        int,
    imgsz:        int,
    device:       str,
    class_weights:list,
    patience:     int  = 30,
) -> dict:
    model_str = MODEL_REGISTRY[model_key]
    print(f"\n{'─'*60}")
    print(f"  Model  : {model_key}  ({model_str})")
    print(f"  Data   : {data_yaml}")
    print(f"  Epochs : {epochs}   Batch : {batch}   imgsz : {imgsz}")
    print(f"{'─'*60}")

    model = YOLO(model_str)

    extra_kwargs = {}
    if model_key == "yolov11m_o2o":
        extra_kwargs["trainer"] = O2OTrainer

    results = model.train(
        data            = str(data_yaml),
        epochs          = epochs,
        batch           = batch,
        imgsz           = imgsz,
        device          = device,
        project         = str(output_dir),
        name            = model_key,
        exist_ok        = True,
        seed            = 42,
        deterministic   = True,

        # ── optimizer ────────────────────────────────────────────────────
        optimizer       = "AdamW",
        lr0             = 1e-3,
        lrf             = 0.01,
        momentum        = 0.937,
        weight_decay    = 5e-4,
        warmup_epochs   = 3,
        warmup_momentum = 0.8,

        # ── loss ─────────────────────────────────────────────────────────
        cls             = 0.5,
        box             = 7.5,
        dfl             = 1.5,

        # ── augmentation (Ultralytics built-in, applied at training time) ─
        # These replace the Albumentations pipeline we removed from preprocess
        augment         = True,
        mosaic          = 0.5,      # 4-image mosaic, p=0.5
        mixup           = 0.2,      # MixUp alpha=0.2
        copy_paste      = 0.0,      # done in preprocessing already
        flipud          = 0.1,      # vertical flip
        fliplr          = 0.5,      # horizontal flip
        degrees         = 30.0,     # rotation ±30°
        translate       = 0.1,
        scale           = 0.5,
        shear           = 2.0,
        perspective     = 0.0,
        hsv_h           = 0.015,    # hue shift
        hsv_s           = 0.7,      # saturation
        hsv_v           = 0.4,      # value / brightness

        # ── other ─────────────────────────────────────────────────────────
        rect            = False,
        cache           = "disk",
        workers         = 8,
        patience        = patience,
        save_period     = 10,
        val             = True,
        plots           = True,
        verbose         = True,
        **extra_kwargs,
    )

    # ── extract metrics ───────────────────────────────────────────────────
    metrics = {}
    try:
        rd = results.results_dict
        metrics = {
            "mAP50":     float(rd.get("metrics/mAP50(B)",    0)),
            "mAP50_95":  float(rd.get("metrics/mAP50-95(B)", 0)),
            "precision": float(rd.get("metrics/precision(B)", 0)),
            "recall":    float(rd.get("metrics/recall(B)",    0)),
            "box_loss":  float(rd.get("val/box_loss",         0)),
            "cls_loss":  float(rd.get("val/cls_loss",         0)),
        }
    except Exception as e:
        print(f"[warn] Could not parse metrics: {e}")

    # ── latency ───────────────────────────────────────────────────────────
    try:
        ms = measure_latency(model, imgsz, device)
        metrics["latency_ms"] = round(ms, 3)
        print(f"  Latency: {ms:.2f} ms/image")
    except Exception as e:
        print(f"[warn] Latency measurement failed: {e}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MLFLOW
# ─────────────────────────────────────────────────────────────────────────────

def log_to_mlflow(experiment: str, model_key: str, metrics: dict,
                  params: dict, output_dir: Path):
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=model_key):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        best_pt = output_dir / model_key / "weights" / "best.pt"
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")
        results_csv = output_dir / model_key / "results.csv"
        if results_csv.exists():
            mlflow.log_artifact(str(results_csv))


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(output_dir: Path):
    records = []
    for p in sorted(output_dir.glob("*/metrics.json")):
        with open(p) as f:
            m = json.load(f)
        m["model"] = p.parent.name
        records.append(m)
    if not records:
        print("No metrics.json files found.")
        return
    df = pd.DataFrame(records).set_index("model").sort_values("mAP50", ascending=False)
    print("\n" + "="*70)
    print("PATH A  —  YOLO BENCHMARK SUMMARY")
    print("="*70)
    print(df.to_string(float_format="{:.4f}".format))
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Path A: YOLO family training grid")
    p.add_argument("--data",       type=Path, required=False)
    p.add_argument("--output",     type=Path, default=Path("runs/path_A"))
    p.add_argument("--models",     nargs="+",
                   default=["yolov8n", "yolov8s", "yolov9s", "yolov11n"],
                   choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--epochs",     type=int,  default=100)
    p.add_argument("--batch",      type=int,  default=16)
    p.add_argument("--imgsz",      type=int,  default=640)
    p.add_argument("--device",     type=str,  default="0")
    p.add_argument("--patience",   type=int,  default=30,
                   help="Early stopping: epochs without improvement before stopping")
    p.add_argument("--mlflow_uri", type=str,  default="./mlruns")
    p.add_argument("--summarize",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.summarize:
        print_summary(args.output)
        raise SystemExit(0)

    assert args.data is not None, "--data is required"
    assert args.data.exists(),    f"Dataset YAML not found: {args.data}"

    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(int(args.device))
        print(f"GPU  : {dev.name}")
        print(f"VRAM : {dev.total_memory/1e9:.1f} GB")
    else:
        print("[WARN] CUDA not available — training on CPU will be very slow")

    mlflow.set_tracking_uri(args.mlflow_uri)
    class_weights = load_class_weights(args.data)
    args.output.mkdir(parents=True, exist_ok=True)

    for model_key in args.models:
        params = {
            "model":   model_key,
            "epochs":  args.epochs,
            "batch":   args.batch,
            "imgsz":   args.imgsz,
            "device":  args.device,
            "seed":    42,
        }

        metrics = train_single(
            model_key     = model_key,
            data_yaml     = args.data,
            output_dir    = args.output,
            epochs        = args.epochs,
            batch         = args.batch,
            imgsz         = args.imgsz,
            device        = args.device,
            class_weights = class_weights,
            patience      = args.patience,
        )

        metrics_path = args.output / model_key / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        log_to_mlflow(
            experiment = "litter_path_A",
            model_key  = model_key,
            metrics    = metrics,
            params     = params,
            output_dir = args.output,
        )

        print(f"\n  [{model_key}] "
              f"mAP50={metrics.get('mAP50',0):.4f}  "
              f"mAP50-95={metrics.get('mAP50_95',0):.4f}  "
              f"latency={metrics.get('latency_ms',0):.2f}ms")

    print_summary(args.output)