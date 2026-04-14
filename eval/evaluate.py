"""
evaluate.py  —  Unified evaluation for all benchmark paths.

Produces:
  results/
    individual/
      path_A_yolov9s.json          per-model metrics
      path_A_yolov9s_history.csv   per-epoch training curve
      path_A_yolov9s_summary.csv   single-row best-epoch snapshot
    global/
      benchmark_summary.csv        all models ranked by mAP50
      benchmark_summary.json
      benchmark_bar_chart.png

Usage — evaluate all Path A models from runs dir:
  python evaluate.py \
      --path      A \
      --runs_dir  ~/runs/path_A \
      --data_yaml ~/processed/dataset_path_A.yaml \
      --output    ~/results \
      --device    0

Evaluate single model:
  python evaluate.py \
      --path      A \
      --model_key yolov9s \
      --weights   ~/runs/path_A/yolov9s/weights/best.pt \
      --data_yaml ~/processed/dataset_path_A.yaml \
      --output    ~/results \
      --device    0

Global summary from all saved results:
  python evaluate.py --summarize --output ~/results
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from ultralytics import YOLO
try:
    from ensemble_boxes import weighted_boxes_fusion
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

COARSE_CLASSES  = ["plastic", "paper", "metal", "glass", "organic", "hazardous", "other"]
IOU_THRESHOLDS  = np.linspace(0.5, 0.95, 10)
TACO_BASELINE   = 0.633   # YOLOv5s single-class 2022 paper

INDIVIDUAL_COLS = [
    # identity
    "path", "model_key", "run_id", "imgsz", "seed",
    # training config
    "epochs_trained", "stopped_early", "best_epoch",
    "batch_size", "optimizer", "lr0", "lrf",
    "momentum", "weight_decay", "warmup_epochs", "patience",
    "mosaic", "mixup", "train_images", "val_images", "test_images",
    # detection metrics
    "mAP50", "mAP50_95", "precision", "recall", "f1",
    # per-class AP
    "AP50_plastic", "AP50_paper", "AP50_metal", "AP50_glass",
    "AP50_organic", "AP50_hazardous", "AP50_other",
    # per-class AR
    "AR50_plastic", "AR50_paper", "AR50_metal", "AR50_glass",
    "AR50_organic", "AR50_hazardous", "AR50_other",
    # per-class gt counts
    "n_gt_plastic", "n_gt_paper", "n_gt_metal", "n_gt_glass",
    "n_gt_organic", "n_gt_hazardous", "n_gt_other",
    # efficiency
    "latency_ms", "fps", "model_params_M", "model_size_MB",
    "gpu_memory_MB",
]

GLOBAL_COLS = [
    "rank", "path", "model_key", "run_id", "imgsz",
    "epochs_trained", "stopped_early",
    "mAP50", "mAP50_95", "precision", "recall", "f1",
    "AP50_plastic", "AP50_paper", "AP50_metal", "AP50_glass",
    "AP50_organic", "AP50_hazardous", "AP50_other",
    "latency_ms", "fps", "model_params_M", "model_size_MB",
    "train_images", "batch_size", "optimizer", "lr0", "seed",
    # detection vs classification separation
    "det_mAP50", "det_mAP50_95",
    "cls_accuracy", "cls_total_boxes", "det_cls_gap",
    "cls_acc_plastic", "cls_acc_paper", "cls_acc_metal", "cls_acc_other",
]


# ─────────────────────────────────────────────────────────────────────────────
# IoU HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    ix1   = np.maximum(box1[:, None, 0], box2[None, :, 0])
    iy1   = np.maximum(box1[:, None, 1], box2[None, :, 1])
    ix2   = np.minimum(box1[:, None, 2], box2[None, :, 2])
    iy2   = np.minimum(box1[:, None, 3], box2[None, :, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-7)


# ─────────────────────────────────────────────────────────────────────────────
# PER-CLASS AP + AR
# ─────────────────────────────────────────────────────────────────────────────

def compute_ap_ar(
    pred_boxes, pred_scores, pred_classes,
    gt_boxes, gt_classes,
    iou_thresh, num_classes,
):
    gt_counts  = np.zeros(num_classes, dtype=int)
    det_scores  = [[] for _ in range(num_classes)]
    det_tp      = [[] for _ in range(num_classes)]

    for img_idx in range(len(pred_boxes)):
        gc = gt_classes[img_idx]
        for c in gc:
            gt_counts[c] += 1

        pb = pred_boxes[img_idx]
        ps = pred_scores[img_idx]
        pc = pred_classes[img_idx]
        gb = gt_boxes[img_idx]

        if len(pb) == 0:
            continue

        order     = np.argsort(-ps)
        matched_gt = set()

        for idx in order:
            c   = pc[idx]
            box = pb[idx:idx+1]
            det_scores[c].append(ps[idx])

            gt_mask = (gc == c)
            if gt_mask.sum() == 0 or len(gb) == 0:
                det_tp[c].append(0)
                continue

            gb_c      = gb[gt_mask]
            gt_idx_c  = np.where(gt_mask)[0]
            ious      = box_iou(box, gb_c)[0]
            best_i    = np.argmax(ious)
            best_iou  = ious[best_i]
            global_gi = gt_idx_c[best_i]

            if best_iou >= iou_thresh and global_gi not in matched_gt:
                det_tp[c].append(1)
                matched_gt.add(global_gi)
            else:
                det_tp[c].append(0)

    per_class_ap = np.zeros(num_classes)
    per_class_ar = np.zeros(num_classes)

    for c in range(num_classes):
        if not det_scores[c] or gt_counts[c] == 0:
            continue
        scores = np.array(det_scores[c])
        tp     = np.array(det_tp[c])
        order  = np.argsort(-scores)
        tp     = tp[order]
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(1 - tp)
        prec   = cum_tp / (cum_tp + cum_fp + 1e-7)
        rec    = cum_tp / (gt_counts[c] + 1e-7)

        ap = sum(
            prec[rec >= t].max() if (rec >= t).any() else 0.0
            for t in np.linspace(0, 1, 101)
        ) / 101
        per_class_ap[c] = ap
        per_class_ar[c] = rec[-1] if len(rec) > 0 else 0.0

    return per_class_ap, per_class_ar, gt_counts


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT YOLO PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def collect_predictions(model, test_dir, device, imgsz=640):
    img_dir   = test_dir / "images"
    lbl_dir   = test_dir / "labels"
    img_paths = sorted(img_dir.glob("*.jpg"))

    pred_boxes, pred_scores, pred_classes = [], [], []
    gt_boxes,   gt_classes                = [], []

    for img_path in tqdm(img_paths, desc="  Inference"):
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        gb, gc   = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id       = int(parts[0])
                cx,cy,nw,nh  = map(float, parts[1:5])
                x1 = (cx - nw/2) * imgsz
                y1 = (cy - nh/2) * imgsz
                x2 = (cx + nw/2) * imgsz
                y2 = (cy + nh/2) * imgsz
                gb.append([x1, y1, x2, y2])
                gc.append(cls_id)

        gt_boxes.append(  np.array(gb, dtype=np.float32) if gb else np.zeros((0,4)))
        gt_classes.append(np.array(gc, dtype=int))

        r  = model.predict(str(img_path), conf=0.001, iou=0.6,
                           imgsz=imgsz, device=device, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            pb = r.boxes.xyxy.cpu().numpy()
            ps = r.boxes.conf.cpu().numpy()
            pc = r.boxes.cls.cpu().numpy().astype(int)
        else:
            pb = np.zeros((0,4))
            ps = np.zeros(0)
            pc = np.zeros(0, dtype=int)

        pred_boxes.append(pb)
        pred_scores.append(ps)
        pred_classes.append(pc)

    return pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes


# ─────────────────────────────────────────────────────────────────────────────
# LATENCY + MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────

def measure_latency(model, imgsz, device, n=100):
    dev = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    dummy = torch.zeros(1, 3, imgsz, imgsz).to(dev)
    for _ in range(10):
        model.predict(source=dummy, verbose=False)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(source=dummy, verbose=False)
    return (time.perf_counter() - t0) / n * 1000


def get_model_info(weights_path: Path):
    size_mb = round(weights_path.stat().st_size / 1e6, 2)
    try:
        ckpt   = torch.load(weights_path, map_location="cpu")
        model  = ckpt.get("model", None)
        params = sum(p.numel() for p in model.parameters()) / 1e6 if model else 0
    except Exception:
        params = 0
    return round(params, 2), size_mb


def get_gpu_memory():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIG READER
# ─────────────────────────────────────────────────────────────────────────────

def read_training_config(run_dir: Path) -> dict:
    """
    Reads args.yaml and results.csv saved by Ultralytics to extract
    training config and epoch info.
    """
    cfg = {}

    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        with open(args_yaml) as f:
            args = yaml.safe_load(f)
        cfg["batch_size"]    = args.get("batch",         "N/A")
        cfg["optimizer"]     = args.get("optimizer",     "N/A")
        cfg["lr0"]           = args.get("lr0",           "N/A")
        cfg["lrf"]           = args.get("lrf",           "N/A")
        cfg["momentum"]      = args.get("momentum",      "N/A")
        cfg["weight_decay"]  = args.get("weight_decay",  "N/A")
        cfg["warmup_epochs"] = args.get("warmup_epochs", "N/A")
        cfg["patience"]      = args.get("patience",      "N/A")
        cfg["mosaic"]        = args.get("mosaic",        "N/A")
        cfg["mixup"]         = args.get("mixup",         "N/A")
        cfg["imgsz"]         = args.get("imgsz",         640)
        cfg["seed"]          = args.get("seed",          42)

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]
            cfg["epochs_trained"] = int(df["epoch"].max())

            map_col = "metrics/mAP50(B)"
            if map_col in df.columns:
                best_idx          = df[map_col].idxmax()
                cfg["best_epoch"] = int(df.loc[best_idx, "epoch"])
            else:
                cfg["best_epoch"] = cfg["epochs_trained"]

            max_epochs = cfg.get("patience", 30)
            cfg["stopped_early"] = cfg["epochs_trained"] < int(
                args.get("epochs", 300) if args_yaml.exists() else 300
            )
        except Exception:
            cfg["epochs_trained"] = "N/A"
            cfg["best_epoch"]     = "N/A"
            cfg["stopped_early"]  = "N/A"

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CURVE
# ─────────────────────────────────────────────────────────────────────────────

def save_training_curve(run_dir: Path, output_path: Path, model_key: str):
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    plots = [
        ("train/box_loss",        "Train box loss"),
        ("train/cls_loss",        "Train cls loss"),
        ("metrics/mAP50(B)",      "mAP@0.5"),
        ("metrics/mAP50-95(B)",   "mAP@0.5:0.95"),
        ("metrics/precision(B)",  "Precision"),
        ("metrics/recall(B)",     "Recall"),
    ]

    for ax, (col, title) in zip(axes, plots):
        col = col.strip()
        if col in df.columns:
            ax.plot(df["epoch"], df[col], linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{title} (no data)")

    fig.suptitle(f"Training curves — {model_key}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS — CONFUSION MATRIX + PR CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    pred_classes, gt_classes, pred_scores,
    save_path, conf_thr=0.25,
):
    y_true, y_pred = [], []
    for i in range(len(gt_classes)):
        gc   = gt_classes[i]
        pc   = pred_classes[i]
        ps   = pred_scores[i]
        mask = ps >= conf_thr
        pc   = pc[mask]
        for g in gc:
            y_true.append(g)
            y_pred.append(int(pc[0]) if len(pc) > 0 else len(COARSE_CLASSES))

    extended = COARSE_CLASSES + ["background"]
    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(len(extended))))
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=extended).plot(
        ax=ax, colorbar=True, xticks_rotation=45
    )
    ax.set_title(save_path.stem.replace("_", " "))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr_curves(
    pred_boxes, pred_scores, pred_classes,
    gt_boxes, gt_classes,
    save_path, iou_thresh=0.5,
):
    num_classes = len(COARSE_CLASSES)
    fig, axes   = plt.subplots(2, 4, figsize=(18, 9),
                                constrained_layout=True)
    axes = axes.flatten()

    for c in range(num_classes):
        scores_c, tp_c, n_gt = [], [], 0

        for i in range(len(pred_boxes)):
            gc    = gt_classes[i]
            n_gt += (gc == c).sum()
            pb    = pred_boxes[i]
            ps    = pred_scores[i]
            pc    = pred_classes[i]
            mask  = (pc == c)
            if not mask.any():
                continue

            pb_c  = pb[mask]
            ps_c  = ps[mask]
            gb    = gt_boxes[i]
            gc_   = gt_classes[i]
            gb_c  = gb[gc_ == c] if (gc_ == c).any() else np.zeros((0,4))
            matched = set()

            for idx in np.argsort(-ps_c):
                scores_c.append(ps_c[idx])
                if len(gb_c) == 0:
                    tp_c.append(0)
                    continue
                ious     = box_iou(pb_c[idx:idx+1], gb_c)[0]
                best_i   = np.argmax(ious)
                best_iou = ious[best_i]
                if best_iou >= iou_thresh and best_i not in matched:
                    tp_c.append(1)
                    matched.add(best_i)
                else:
                    tp_c.append(0)

        ax = axes[c]
        if not scores_c or n_gt == 0:
            ax.set_title(f"{COARSE_CLASSES[c]} (no data)")
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            continue

        scores_c = np.array(scores_c)
        tp_c     = np.array(tp_c)
        order    = np.argsort(-scores_c)
        tp_c     = tp_c[order]
        cum_tp   = np.cumsum(tp_c)
        cum_fp   = np.cumsum(1 - tp_c)
        prec     = cum_tp / (cum_tp + cum_fp + 1e-7)
        rec      = cum_tp / (n_gt + 1e-7)
        ap       = sum(
            prec[rec >= t].max() if (rec >= t).any() else 0.0
            for t in np.linspace(0, 1, 101)
        ) / 101

        ax.plot(rec, prec, linewidth=1.5)
        ax.fill_between(rec, prec, alpha=0.15)
        ax.set_title(f"{COARSE_CLASSES[c]}  AP={ap:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.grid(True, alpha=0.3)

    for i in range(num_classes, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(save_path.stem.replace("_", " "), fontsize=13)
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION — ONE MODEL
# ─────────────────────────────────────────────────────────────────────────────


def compute_detection_only_map(
    pred_boxes, pred_scores, pred_classes,
    gt_boxes, gt_classes,
):
    """
    Computes mAP treating all classes as a single 'litter' class.
    Measures pure localization ability independent of classification.
    """
    # collapse all classes to 0
    pred_classes_collapsed = [np.zeros_like(pc) for pc in pred_classes]
    gt_classes_collapsed   = [np.zeros_like(gc) for gc in gt_classes]

    ap_per_iou = []
    for iou_t in IOU_THRESHOLDS:
        ap, _, _ = compute_ap_ar(
            pred_boxes, pred_scores, pred_classes_collapsed,
            gt_boxes, gt_classes_collapsed,
            iou_thresh=iou_t, num_classes=1,
        )
        ap_per_iou.append(float(ap.mean()))

    return {
        "det_mAP50":    round(ap_per_iou[0], 5),
        "det_mAP50_95": round(float(np.mean(ap_per_iou)), 5),
    }


def compute_classification_accuracy(
    pred_boxes, pred_scores, pred_classes,
    gt_boxes, gt_classes,
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.25,
):
    """
    Computes classification accuracy ONLY on correctly localized boxes.
    A prediction counts if IoU >= iou_thresh with a ground truth box.
    Then measures whether the class label is correct.
    Answers: given the model found an object, how often does it label it right?
    """
    correct = 0
    total   = 0
    per_class_correct = np.zeros(len(COARSE_CLASSES))
    per_class_total   = np.zeros(len(COARSE_CLASSES))

    for img_idx in range(len(pred_boxes)):
        pb = pred_boxes[img_idx]
        ps = pred_scores[img_idx]
        pc = pred_classes[img_idx]
        gb = gt_boxes[img_idx]
        gc = gt_classes[img_idx]

        if len(pb) == 0 or len(gb) == 0:
            continue

        # filter by confidence
        mask = ps >= conf_thresh
        pb   = pb[mask]
        pc   = pc[mask]
        ps   = ps[mask]

        if len(pb) == 0:
            continue

        # for each prediction find best matching gt box
        ious = box_iou(pb, gb)   # (N_pred, N_gt)

        matched_gt = set()
        order = np.argsort(-ps)

        for idx in order:
            best_gt   = np.argmax(ious[idx])
            best_iou  = ious[idx, best_gt]

            if best_iou >= iou_thresh and best_gt not in matched_gt:
                matched_gt.add(best_gt)
                gt_cls   = int(gc[best_gt])
                pred_cls = int(pc[idx])
                total   += 1
                per_class_total[gt_cls] += 1
                if pred_cls == gt_cls:
                    correct += 1
                    per_class_correct[gt_cls] += 1

    cls_accuracy = round(correct / max(total, 1), 5)

    # per-class accuracy
    per_class_acc = {}
    for i, cls in enumerate(COARSE_CLASSES):
        if per_class_total[i] > 0:
            per_class_acc[f"cls_acc_{cls}"] = round(
                float(per_class_correct[i] / per_class_total[i]), 5)
        else:
            per_class_acc[f"cls_acc_{cls}"] = None

    return {
        "cls_accuracy":      cls_accuracy,
        "cls_total_boxes":   total,
        **per_class_acc,
    }

def evaluate_model(
    path_tag, model_key, weights, data_yaml,
    output_dir, run_dir, device="0", imgsz=640,
):
    print(f"\n{'─'*60}")
    print(f"  Path      : {path_tag}")
    print(f"  Model     : {model_key}")
    print(f"  Weights   : {weights}")
    print(f"{'─'*60}")

    # resolve test dir
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    data_root    = Path(cfg["path"])
    test_img_dir = data_root / cfg.get("test", "test/path_A/images")
    test_dir     = test_img_dir.parent
    assert test_dir.exists(), f"Test dir not found: {test_dir}"

    # count splits
    train_images = len(list((data_root / cfg.get("train","train/path_A/images")).glob("*.jpg")))
    val_images   = len(list((data_root / cfg.get("val",  "val/path_A/images"  )).glob("*.jpg")))
    test_images  = len(list((test_dir / "images").glob("*.jpg"))) if (test_dir/"images").exists() else 0

    model = YOLO(str(weights))

    # collect predictions on test split
    torch.cuda.reset_peak_memory_stats()
    pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes = \
        collect_predictions(model, test_dir, device, imgsz)
    gpu_mem = get_gpu_memory()

    # mAP across IoU thresholds
    map_per_iou = []
    for iou_t in IOU_THRESHOLDS:
        ap, _, _ = compute_ap_ar(
            pred_boxes, pred_scores, pred_classes,
            gt_boxes, gt_classes,
            iou_thresh=iou_t, num_classes=len(COARSE_CLASSES),
        )
        map_per_iou.append(ap.mean())

    mAP50    = float(map_per_iou[0])
    mAP50_95 = float(np.mean(map_per_iou))

    # per-class AP + AR at IoU=0.5
    ap50, ar50, gt_counts = compute_ap_ar(
        pred_boxes, pred_scores, pred_classes,
        gt_boxes, gt_classes,
        iou_thresh=0.5, num_classes=len(COARSE_CLASSES),
    )

    # precision / recall / F1 at conf=0.25
    y_true, y_pred = [], []
    for i in range(len(gt_classes)):
        gc   = gt_classes[i]
        pc   = pred_classes[i]
        ps   = pred_scores[i]
        mask = ps >= 0.25
        pc   = pc[mask]
        for g in gc:
            y_true.append(g)
            y_pred.append(int(pc[0]) if len(pc) > 0 else len(COARSE_CLASSES))

    labels = list(range(len(COARSE_CLASSES)))
    prec = float(precision_score(y_true, y_pred, labels=labels,
                                  average="macro", zero_division=0))
    rec  = float(recall_score(   y_true, y_pred, labels=labels,
                                  average="macro", zero_division=0))
    f1   = float(f1_score(       y_true, y_pred, labels=labels,
                                  average="macro", zero_division=0))

    # latency + model info
    latency_ms  = measure_latency(model, imgsz, device)
    params_m, size_mb = get_model_info(Path(weights))

    # training config from run dir
    train_cfg = read_training_config(run_dir) if run_dir else {}

    run_id = Path(run_dir).name if run_dir else model_key

    # ── detection-only and classification-only metrics ──────────────────
    det_only = compute_detection_only_map(
        pred_boxes, pred_scores, pred_classes,
        gt_boxes, gt_classes,
    )
    cls_only = compute_classification_accuracy(
        pred_boxes, pred_scores, pred_classes,
        gt_boxes, gt_classes,
    )

    # ── assemble result dict ──────────────────────────────────────────────
    result = {
        "path":          path_tag,
        "model_key":     model_key,
        "run_id":        run_id,
        "imgsz":         train_cfg.get("imgsz", imgsz),
        "seed":          train_cfg.get("seed", 42),
        "epochs_trained":train_cfg.get("epochs_trained", "N/A"),
        "stopped_early": train_cfg.get("stopped_early",  "N/A"),
        "best_epoch":    train_cfg.get("best_epoch",     "N/A"),
        "batch_size":    train_cfg.get("batch_size",     "N/A"),
        "optimizer":     train_cfg.get("optimizer",      "N/A"),
        "lr0":           train_cfg.get("lr0",            "N/A"),
        "lrf":           train_cfg.get("lrf",            "N/A"),
        "momentum":      train_cfg.get("momentum",       "N/A"),
        "weight_decay":  train_cfg.get("weight_decay",   "N/A"),
        "warmup_epochs": train_cfg.get("warmup_epochs",  "N/A"),
        "patience":      train_cfg.get("patience",       "N/A"),
        "mosaic":        train_cfg.get("mosaic",         "N/A"),
        "mixup":         train_cfg.get("mixup",          "N/A"),
        "train_images":  train_images,
        "val_images":    val_images,
        "test_images":   test_images,
        "mAP50":         round(mAP50,    5),
        "mAP50_95":      round(mAP50_95, 5),
        "precision":     round(prec, 5),
        "recall":        round(rec,  5),
        "f1":            round(f1,   5),
        "latency_ms":    round(latency_ms, 3),
        "fps":           round(1000 / latency_ms if latency_ms > 0 else 0, 2),
        "model_params_M":params_m,
        "model_size_MB": size_mb,
        "gpu_memory_MB": gpu_mem,
        # detection-only (collapsed single class)
        "det_mAP50":     det_only.get("det_mAP50",    None),
        "det_mAP50_95":  det_only.get("det_mAP50_95", None),
        # classification-only (on correctly localized boxes)
        "cls_accuracy":  cls_only.get("cls_accuracy",  None),
        "cls_total_boxes": cls_only.get("cls_total_boxes", None),
        # gap: how much classification error costs vs pure detection
        "det_cls_gap":   round(
            (det_only.get("det_mAP50", 0) or 0) -
            round(mAP50, 5), 5
        ),
    }

    for i, cls in enumerate(COARSE_CLASSES):
        result[f"AP50_{cls}"]  = round(float(ap50[i]),    5)
        result[f"AR50_{cls}"]  = round(float(ar50[i]),    5)
        result[f"n_gt_{cls}"]  = int(gt_counts[i])
        result[f"cls_acc_{cls}"] = cls_only.get(f"cls_acc_{cls}", None)

    # ── save individual JSON ──────────────────────────────────────────────
    ind_dir = output_dir / "individual"
    ind_dir.mkdir(parents=True, exist_ok=True)
    result_id   = f"{path_tag}_{run_id}_{model_key}"
    json_path   = ind_dir / f"{result_id}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── save per-model summary CSV (single row) ───────────────────────────
    summary_cols = [c for c in INDIVIDUAL_COLS if c in result]
    pd.DataFrame([result])[summary_cols].to_csv(
        ind_dir / f"{result_id}_summary.csv", index=False
    )

    # ── save per-epoch history CSV from results.csv ───────────────────────
    if run_dir and (Path(run_dir) / "results.csv").exists():
        hist_src = Path(run_dir) / "results.csv"
        hist_dst = ind_dir / f"{result_id}_history.csv"
        df_hist  = pd.read_csv(hist_src)
        df_hist.columns = [c.strip() for c in df_hist.columns]
        df_hist.to_csv(hist_dst, index=False)

    # ── plots ─────────────────────────────────────────────────────────────
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_confusion_matrix(
        pred_classes, gt_classes, pred_scores,
        save_path=plots_dir / f"confusion_{result_id}.png",
    )
    plot_pr_curves(
        pred_boxes, pred_scores, pred_classes,
        gt_boxes, gt_classes,
        save_path=plots_dir / f"pr_curve_{result_id}.png",
    )
    if run_dir:
        save_training_curve(
            Path(run_dir),
            plots_dir / f"training_curve_{result_id}.png",
            model_key,
        )

    # print summary
    print(f"\n  mAP@0.5    : {mAP50:.4f}")
    print(f"  mAP@0.5:95 : {mAP50_95:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1         : {f1:.4f}")
    print(f"  Latency    : {latency_ms:.2f} ms  ({result['fps']:.1f} FPS)")
    print(f"  Params     : {params_m:.1f}M")
    print(f"  Size       : {size_mb:.1f} MB")
    print(f"  Saved      : {json_path}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_ensemble(
    path_tag:   str,
    model_keys: list,
    weights_list: list,
    data_yaml:  Path,
    output_dir: Path,
    device:     str = "0",
    imgsz:      int = 640,
    iou_thr:    float = 0.55,
    skip_box_thr: float = 0.01,
) -> dict:
    """
    Weighted Box Fusion ensemble across multiple YOLO models.
    Each model contributes equally weighted predictions.
    Falls back to simple NMS if ensemble_boxes not installed.
    """
    import yaml as pyyaml
    if not ENSEMBLE_AVAILABLE:
        print("[warn] ensemble_boxes not installed.")
        print("  Install with: pip install ensemble-boxes")
        print("  Falling back to single best model evaluation")
        return evaluate_model(
            path_tag, model_keys[0], weights_list[0],
            data_yaml, output_dir, None, device, imgsz,
        )

    print(f"\n{'─'*60}")
    print(f"  Ensemble: {model_keys}")
    print(f"  WBF iou_thr={iou_thr}  skip_box_thr={skip_box_thr}")
    print(f"{'─'*60}")

    with open(data_yaml) as f:
        cfg = pyyaml.safe_load(f)
    data_root    = Path(cfg["path"])
    test_img_dir = data_root / cfg.get("test", "test/path_A/images")
    test_dir     = test_img_dir.parent
    img_paths    = sorted((test_dir / "images").glob("*.jpg"))
    lbl_dir      = test_dir / "labels"

    models = [YOLO(str(w)) for w in weights_list]
    n_models = len(models)

    ensemble_pred_boxes   = []
    ensemble_pred_scores  = []
    ensemble_pred_classes = []
    gt_boxes_all   = []
    gt_classes_all = []

    for img_path in tqdm(img_paths, desc="  Ensemble inference"):
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        gb, gc = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id      = int(parts[0])
                cx,cy,nw,nh = map(float, parts[1:5])
                x1=(cx-nw/2); y1=(cy-nh/2)
                x2=(cx+nw/2); y2=(cy+nh/2)
                gb.append([x1,y1,x2,y2])
                gc.append(cls_id)
        gt_boxes_all.append(np.array(gb,dtype=np.float32)
                            if gb else np.zeros((0,4)))
        gt_classes_all.append(np.array(gc, dtype=int))

        # collect predictions from all models (normalized 0-1)
        boxes_list   = []
        scores_list  = []
        labels_list  = []

        for model in models:
            r = model.predict(str(img_path), conf=0.001,
                              iou=0.6, imgsz=imgsz,
                              device=device, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                pb = r.boxes.xyxyn.cpu().numpy()   # normalized
                ps = r.boxes.conf.cpu().numpy()
                pc = r.boxes.cls.cpu().numpy().astype(int)
            else:
                pb = np.zeros((0,4))
                ps = np.zeros(0)
                pc = np.zeros(0, dtype=int)
            boxes_list.append(pb.tolist())
            scores_list.append(ps.tolist())
            labels_list.append(pc.tolist())

        # WBF fusion
        weights  = [1.0] * n_models
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

        # convert back to pixel coords (imgsz x imgsz)
        if len(fused_boxes) > 0:
            fused_boxes_px = (np.array(fused_boxes) * imgsz).astype(np.float32)
        else:
            fused_boxes_px = np.zeros((0,4), dtype=np.float32)

        ensemble_pred_boxes.append(fused_boxes_px)
        ensemble_pred_scores.append(np.array(fused_scores))
        ensemble_pred_classes.append(np.array(fused_labels, dtype=int))

    # compute metrics using same functions as single model
    map_per_iou = []
    for iou_t in IOU_THRESHOLDS:
        ap, _, _ = compute_ap_ar(
            ensemble_pred_boxes, ensemble_pred_scores, ensemble_pred_classes,
            gt_boxes_all, gt_classes_all,
            iou_thresh=iou_t, num_classes=len(COARSE_CLASSES),
        )
        map_per_iou.append(ap.mean())

    mAP50    = float(map_per_iou[0])
    mAP50_95 = float(np.mean(map_per_iou))

    ap50, ar50, gt_counts = compute_ap_ar(
        ensemble_pred_boxes, ensemble_pred_scores, ensemble_pred_classes,
        gt_boxes_all, gt_classes_all,
        iou_thresh=0.5, num_classes=len(COARSE_CLASSES),
    )

    y_true, y_pred = [], []
    for i in range(len(gt_classes_all)):
        gc   = gt_classes_all[i]
        pc   = ensemble_pred_classes[i]
        ps   = ensemble_pred_scores[i]
        mask = ps >= 0.25
        pc   = pc[mask]
        for g in gc:
            y_true.append(g)
            y_pred.append(int(pc[0]) if len(pc) > 0 else len(COARSE_CLASSES))

    labels = list(range(len(COARSE_CLASSES)))
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = float(precision_score(y_true, y_pred, labels=labels,
                                  average="macro", zero_division=0))
    rec  = float(recall_score(   y_true, y_pred, labels=labels,
                                  average="macro", zero_division=0))
    f1   = float(f1_score(       y_true, y_pred, labels=labels,
                                  average="macro", zero_division=0))

    ensemble_key = "ensemble_" + "_".join(model_keys)
    result = {
        "path":       path_tag,
        "model_key":  ensemble_key,
        "run_id":     ensemble_key,
        "imgsz":      imgsz,
        "mAP50":      round(mAP50,    5),
        "mAP50_95":   round(mAP50_95, 5),
        "precision":  round(prec, 5),
        "recall":     round(rec,  5),
        "f1":         round(f1,   5),
        "n_models":   n_models,
        "models":     model_keys,
        "wbf_iou_thr": iou_thr,
    }
    for i, cls in enumerate(COARSE_CLASSES):
        result[f"AP50_{cls}"] = round(float(ap50[i]), 5)
        result[f"AR50_{cls}"] = round(float(ar50[i]), 5)
        result[f"n_gt_{cls}"] = int(gt_counts[i])

    ind_dir = output_dir / "individual"
    ind_dir.mkdir(parents=True, exist_ok=True)
    out_path = ind_dir / f"{path_tag}_{ensemble_key}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Ensemble mAP50    : {mAP50:.4f}")
    print(f"  Ensemble mAP50-95 : {mAP50_95:.4f}")
    print(f"  Ensemble F1       : {f1:.4f}")
    print(f"  Saved: {out_path}")
    return result

def build_global_summary(output_dir: Path):
    ind_dir = output_dir / "individual"
    records = []
    for p in sorted(ind_dir.glob("*.json")):
        with open(p) as f:
            records.append(json.load(f))

    if not records:
        print(f"No individual result files found in {ind_dir}")
        return

    df = pd.DataFrame(records)
    df = df.sort_values("mAP50", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    global_dir = output_dir / "global"
    global_dir.mkdir(exist_ok=True)

    # save full global CSV
    csv_path = global_dir / "benchmark_summary.csv"
    global_cols_present = [c for c in GLOBAL_COLS if c in df.columns]
    df[global_cols_present].to_csv(csv_path, index=False)

    # save JSON
    json_path = global_dir / "benchmark_summary.json"
    with open(json_path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    # print ranked table
    display_cols = ["rank", "path", "model_key", "run_id",
                    "imgsz", "epochs_trained", "stopped_early",
                    "mAP50", "mAP50_95", "precision",
                    "recall", "f1", "latency_ms", "fps",
                    "model_params_M"]
    display_cols = [c for c in display_cols if c in df.columns]

    print("\n" + "="*90)
    print("GLOBAL BENCHMARK SUMMARY  —  all paths, ranked by mAP@0.5")
    print("="*90)
    print(df[display_cols].to_string(index=False,
                                      float_format="{:.4f}".format))
    print("="*90)

    # per-class AP table
    ap_cols = [c for c in df.columns if c.startswith("AP50_")]
    if ap_cols:
        print("\nPer-class AP@0.5:")
        id_cols = ["path", "model_key", "run_id"]
        id_cols = [c for c in id_cols if c in df.columns]
        print(df[id_cols + ap_cols].to_string(
            index=False, float_format="{:.4f}".format))

    _plot_global_bar(df, global_dir)
    print(f"\nGlobal results saved to: {global_dir}")


def _plot_global_bar(df, save_dir):
    metrics = ["mAP50", "mAP50_95", "precision", "recall", "f1"]
    metrics = [m for m in metrics if m in df.columns]
    labels  = [f"{r['path']}_{r.get('run_id', r['model_key'])}"
               for _, r in df.iterrows()]
    x       = np.arange(len(labels))
    width   = 0.15

    fig, ax = plt.subplots(figsize=(max(14, len(labels)*2), 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, df[metric].values, width, label=metric)

    ax.set_xticks(x + width*2)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Benchmark comparison — all paths")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "benchmark_bar_chart.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Unified benchmark evaluation")
    p.add_argument("--path",      type=str,  default="A",
                   choices=["A", "B", "C"])
    p.add_argument("--model_key", type=str,  default=None)
    p.add_argument("--weights",   type=Path, default=None)
    p.add_argument("--runs_dir",  type=Path, default=None,
                   help="Directory with multiple model run subfolders")
    p.add_argument("--data_yaml", type=Path, default=None)
    p.add_argument("--output",    type=Path, default=Path("results"))
    p.add_argument("--device",    type=str,  default="0")
    p.add_argument("--imgsz",     type=int,  default=640)
    p.add_argument("--summarize",   action="store_true")
    p.add_argument("--ensemble",    action="store_true",
                   help="Run WBF ensemble across all models in --runs_dir")
    p.add_argument("--ensemble_iou", type=float, default=0.55,
                   help="WBF IoU threshold (default 0.55)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.summarize:
        build_global_summary(args.output)
        raise SystemExit(0)

    if args.path == "B":
        if args.runs_dir:
            ind_dir = args.output / "individual"
            ind_dir.mkdir(parents=True, exist_ok=True)
            for metrics_json in sorted(args.runs_dir.glob("*/metrics.json")):
                with open(metrics_json) as f:
                    result = json.load(f)
                result["path"]      = "B"
                result["model_key"] = result.get("classifier",
                                                  metrics_json.parent.name)
                result["run_id"]    = metrics_json.parent.name
                result.setdefault("mAP50",    None)
                result.setdefault("mAP50_95", None)
                out = ind_dir / f"B_{result['run_id']}_{result['model_key']}.json"
                with open(out, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  Loaded Path B result: {out.name}")
        build_global_summary(args.output)
        raise SystemExit(0)

    assert args.data_yaml is not None, "--data_yaml is required"
    assert args.data_yaml.exists(),    f"YAML not found: {args.data_yaml}"

    # ── single model mode ─────────────────────────────────────────────────
    if args.model_key and args.weights:
        run_dir = args.weights.parent.parent  # weights/best.pt -> model_dir
        evaluate_model(
            path_tag  = args.path,
            model_key = args.model_key,
            weights   = args.weights,
            data_yaml = args.data_yaml,
            output_dir= args.output,
            run_dir   = run_dir,
            device    = args.device,
            imgsz     = args.imgsz,
        )

    # ── ensemble mode ────────────────────────────────────────────────────
    if args.ensemble and args.runs_dir:
        weight_paths = sorted(args.runs_dir.glob("*/weights/best.pt"))
        if not weight_paths:
            print(f"No best.pt found under {args.runs_dir}")
            raise SystemExit(1)
        model_keys   = [wp.parent.parent.name for wp in weight_paths]
        print(f"  Ensembling {len(model_keys)} models: {model_keys}")
        evaluate_ensemble(
            path_tag     = args.path,
            model_keys   = model_keys,
            weights_list = weight_paths,
            data_yaml    = args.data_yaml,
            output_dir   = args.output,
            device       = args.device,
            imgsz        = args.imgsz,
            iou_thr      = args.ensemble_iou,
        )
        build_global_summary(args.output)
        raise SystemExit(0)

    # ── batch mode: all best.pt under runs_dir ────────────────────────────
    elif args.runs_dir:
        assert args.runs_dir.exists(), f"runs_dir not found: {args.runs_dir}"
        weight_paths = sorted(args.runs_dir.glob("*/weights/best.pt"))
        if not weight_paths:
            print(f"No best.pt found under {args.runs_dir}")
            raise SystemExit(1)
        for wp in weight_paths:
            model_key = wp.parent.parent.name
            run_dir   = wp.parent.parent
            evaluate_model(
                path_tag  = args.path,
                model_key = model_key,
                weights   = wp,
                data_yaml = args.data_yaml,
                output_dir= args.output,
                run_dir   = run_dir,
                device    = args.device,
                imgsz     = args.imgsz,
            )
    else:
        print("Provide either (--model_key + --weights) or --runs_dir")
        raise SystemExit(1)

    build_global_summary(args.output)
