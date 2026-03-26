"""
evaluate_path_B_combined.py

True apples-to-apples Path B evaluation:
  1. Frozen YOLOv8m detector  → bounding boxes (class-agnostic or 4-class)
  2. Path B classifier        → reassigns class label per crop
  3. Compute mAP50 / mAP50-95 on frozen test set with reassigned labels

This produces detection metrics comparable to Path A.

Usage:
  conda activate litter_AB
  python evaluate_path_B_combined.py \
      --detector_weights  ~/runs/path_A_4cls/yolov8m/weights/best.pt \
      --classifier_dir    ~/runs/path_B \
      --classifiers       resnet50 vit_b16_scratch vit_b16_imagenet \
      --data_yaml         ~/processed_4cls/dataset_path_A.yaml \
      --output            ~/results_4cls \
      --device            0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

COARSE_CLASSES = ["plastic", "paper", "metal", "other"]
NUM_CLASSES    = len(COARSE_CLASSES)
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)

CLASSIFIER_REGISTRY = {
    "resnet50":         "resnet50.a1_in1k",
    "vit_b16_scratch":  "vit_base_patch16_224",
    "vit_b16_imagenet": "vit_base_patch16_224.augreg_in21k_ft_in1k",
}

CROP_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# LOAD CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier(clf_name: str, weights_path: Path,
                    device: torch.device) -> nn.Module:
    import timm
    model_str  = CLASSIFIER_REGISTRY[clf_name]
    pretrained = clf_name != "vit_b16_scratch"
    model = timm.create_model(model_str, pretrained=False,
                               num_classes=NUM_CLASSES)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {clf_name}: {n:.1f}M params from {weights_path.name}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CROP + CLASSIFY
# ─────────────────────────────────────────────────────────────────────────────

def classify_crops(img_pil: Image.Image,
                   boxes_xyxy: np.ndarray,
                   classifier: nn.Module,
                   device: torch.device) -> np.ndarray:
    """
    Given a PIL image and Nx4 boxes in xyxy pixel coords,
    return an array of N predicted class indices.
    """
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=int)

    crops = []
    W, H  = img_pil.size
    for x1, y1, x2, y2 in boxes_xyxy:
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(W, int(x2)); y2 = min(H, int(y2))
        if x2 <= x1 or y2 <= y1:
            crops.append(torch.zeros(3, 224, 224))
            continue
        crop = img_pil.crop((x1, y1, x2, y2)).convert("RGB")
        crops.append(CROP_TRANSFORM(crop))

    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        logits  = classifier(batch)
        preds   = logits.argmax(dim=1).cpu().numpy()
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# mAP COMPUTATION  (same as evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def box_iou_np(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """box1: (N,4)  box2: (M,4)  returns (N,M) IoU matrix, xyxy format."""
    ax1, ay1, ax2, ay2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    bx1, by1, bx2, by2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    ix1 = np.maximum(ax1[:,None], bx1[None,:])
    iy1 = np.maximum(ay1[:,None], by1[None,:])
    ix2 = np.minimum(ax2[:,None], bx2[None,:])
    iy2 = np.minimum(ay2[:,None], by2[None,:])

    inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
    a_area = (ax2-ax1)*(ay2-ay1)
    b_area = (bx2-bx1)*(by2-by1)
    union  = a_area[:,None] + b_area[None,:] - inter
    return inter / (union + 1e-7)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """COCO-style 101-point AP."""
    r_pts = np.linspace(0, 1, 101)
    ap = 0.0
    for r in r_pts:
        mask = recalls >= r
        ap  += precisions[mask].max() if mask.any() else 0.0
    return ap / 101


def compute_map(
    pred_boxes:   list,   # per-image list of (N,4) xyxy arrays
    pred_scores:  list,   # per-image list of (N,)  float arrays
    pred_classes: list,   # per-image list of (N,)  int   arrays
    gt_boxes:     list,   # per-image list of (M,4) xyxy arrays
    gt_classes:   list,   # per-image list of (M,)  int   arrays
    iou_thresh:   float,
) -> tuple:
    """Returns (AP per class array, AR per class array, gt_counts array)."""
    # collect all predictions sorted by score
    all_preds = []   # (img_idx, score, pred_cls, box)
    for i, (boxes, scores, classes) in enumerate(
            zip(pred_boxes, pred_scores, pred_classes)):
        for b, s, c in zip(boxes, scores, classes):
            all_preds.append((i, float(s), int(c), b))

    all_preds.sort(key=lambda x: -x[1])

    ap_per_class = np.zeros(NUM_CLASSES)
    ar_per_class = np.zeros(NUM_CLASSES)
    gt_counts    = np.zeros(NUM_CLASSES, dtype=int)

    for cls in range(NUM_CLASSES):
        gt_counts[cls] = sum(
            (gt_classes[i] == cls).sum()
            for i in range(len(gt_classes))
        )
        if gt_counts[cls] == 0:
            continue

        cls_preds = [(i, s, b) for i, s, c, b in all_preds if c == cls]
        if not cls_preds:
            continue

        matched = [set() for _ in range(len(gt_boxes))]
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        for k, (img_i, score, pred_b) in enumerate(cls_preds):
            gt_b = gt_boxes[img_i]
            gt_c = gt_classes[img_i]
            cls_mask = gt_c == cls
            cls_gt   = gt_b[cls_mask]
            cls_idx  = np.where(cls_mask)[0]

            if len(cls_gt) == 0:
                fp[k] = 1
                continue

            ious    = box_iou_np(pred_b[None], cls_gt)[0]
            best_j  = ious.argmax()
            best_iou= ious[best_j]

            if best_iou >= iou_thresh and \
               cls_idx[best_j] not in matched[img_i]:
                tp[k] = 1
                matched[img_i].add(cls_idx[best_j])
            else:
                fp[k] = 1

        cum_tp  = np.cumsum(tp)
        cum_fp  = np.cumsum(fp)
        recalls    = cum_tp / (gt_counts[cls] + 1e-7)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-7)

        ap_per_class[cls] = compute_ap(recalls, precisions)
        ar_per_class[cls] = recalls.max() if len(recalls) else 0.0

    return ap_per_class, ar_per_class, gt_counts


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_combined(
    detector:    YOLO,
    classifier:  nn.Module,
    clf_name:    str,
    test_img_dir: Path,
    test_lbl_dir: Path,
    device:      torch.device,
    imgsz:       int = 640,
    det_conf:    float = 0.001,
    det_iou:     float = 0.6,
) -> dict:
    img_paths = sorted(test_img_dir.glob("*.jpg"))
    print(f"\n  Running combined inference on {len(img_paths)} test images...")

    pred_boxes_all   = []
    pred_scores_all  = []
    pred_classes_all = []
    gt_boxes_all     = []
    gt_classes_all   = []

    t0 = time.perf_counter()

    for img_path in tqdm(img_paths, desc=f"  {clf_name}"):
        # ── ground truth ──────────────────────────────────────────────────
        lbl_path = test_lbl_dir / img_path.with_suffix(".txt").name
        gt_b, gt_c = [], []
        if lbl_path.exists():
            img_pil  = Image.open(img_path)
            W, H     = img_pil.size
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id      = int(parts[0])
                cx,cy,nw,nh = map(float, parts[1:5])
                x1=(cx-nw/2)*W; y1=(cy-nh/2)*H
                x2=(cx+nw/2)*W; y2=(cy+nh/2)*H
                gt_b.append([x1,y1,x2,y2])
                gt_c.append(cls_id)
        gt_boxes_all.append(
            np.array(gt_b, dtype=np.float32)
            if gt_b else np.zeros((0,4), dtype=np.float32))
        gt_classes_all.append(np.array(gt_c, dtype=int))

        # ── stage 1: detect ───────────────────────────────────────────────
        result = detector.predict(
            str(img_path), conf=det_conf, iou=det_iou,
            imgsz=imgsz, device=str(device.index
                         if device.type=="cuda" else "cpu"),
            verbose=False)[0]

        if result.boxes is None or len(result.boxes) == 0:
            pred_boxes_all.append(np.zeros((0,4), dtype=np.float32))
            pred_scores_all.append(np.zeros(0, dtype=np.float32))
            pred_classes_all.append(np.zeros(0, dtype=int))
            continue

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()   # (N,4) pixel coords
        scores     = result.boxes.conf.cpu().numpy()   # (N,)

        # ── stage 2: classify each crop ───────────────────────────────────
        if not img_pil:
            img_pil = Image.open(img_path)
        new_classes = classify_crops(
            img_pil, boxes_xyxy, classifier, device)

        pred_boxes_all.append(boxes_xyxy)
        pred_scores_all.append(scores)
        pred_classes_all.append(new_classes)

    elapsed = time.perf_counter() - t0
    fps     = len(img_paths) / elapsed

    # ── compute mAP across IoU thresholds ─────────────────────────────────
    map_per_iou = []
    for iou_t in IOU_THRESHOLDS:
        ap, _, _ = compute_map(
            pred_boxes_all, pred_scores_all, pred_classes_all,
            gt_boxes_all, gt_classes_all, iou_thresh=iou_t)
        map_per_iou.append(ap.mean())

    mAP50    = float(map_per_iou[0])
    mAP50_95 = float(np.mean(map_per_iou))

    ap50, ar50, gt_counts = compute_map(
        pred_boxes_all, pred_scores_all, pred_classes_all,
        gt_boxes_all, gt_classes_all, iou_thresh=0.5)

    result_dict = {
        "path":       "B",
        "model_key":  f"B_{clf_name}_combined",
        "classifier": clf_name,
        "mAP50":      round(mAP50,    5),
        "mAP50_95":   round(mAP50_95, 5),
        "precision":  None,   # crop-level metric, not comparable here
        "recall":     None,
        "fps":        round(fps, 2),
        "n_images":   len(img_paths),
        "det_conf":   det_conf,
        "det_iou":    det_iou,
    }
    for i, cls in enumerate(COARSE_CLASSES):
        result_dict[f"AP50_{cls}"] = round(float(ap50[i]), 5)
        result_dict[f"AR50_{cls}"] = round(float(ar50[i]), 5)
        result_dict[f"n_gt_{cls}"] = int(gt_counts[i])

    print(f"\n  [{clf_name}]  mAP50={mAP50:.4f}  "
          f"mAP50-95={mAP50_95:.4f}  fps={fps:.1f}")
    for i, cls in enumerate(COARSE_CLASSES):
        print(f"    AP50 {cls:<8}: {ap50[i]:.4f}  "
              f"(n_gt={gt_counts[i]})")

    return result_dict


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Path B combined (detector + classifier) mAP evaluation")
    p.add_argument("--detector_weights", type=Path, required=True,
                   help="Path A best.pt (frozen detector)")
    p.add_argument("--classifier_dir",   type=Path, required=True,
                   help="runs/path_B root (contains resnet50/, vit_b16_*/)")
    p.add_argument("--classifiers",      nargs="+",
                   default=list(CLASSIFIER_REGISTRY.keys()),
                   choices=list(CLASSIFIER_REGISTRY.keys()))
    p.add_argument("--data_yaml",        type=Path, required=True)
    p.add_argument("--output",           type=Path,
                   default=Path("results_4cls"))
    p.add_argument("--device",           type=str, default="0")
    p.add_argument("--imgsz",            type=int, default=640)
    p.add_argument("--det_conf",         type=float, default=0.001)
    p.add_argument("--det_iou",          type=float, default=0.6)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        f"cuda:{args.device}"
        if args.device.isdigit() and torch.cuda.is_available()
        else "cpu")
    print(f"Device: {device}")

    # resolve test image + label dirs from YAML
    with open(args.data_yaml) as f:
        cfg = yaml.safe_load(f)
    data_root    = Path(cfg["path"])
    test_img_dir = data_root / cfg.get("test", "test/path_A/images")
    test_lbl_dir = Path(str(test_img_dir).replace("images", "labels"))
    assert test_img_dir.exists(), f"Test images not found: {test_img_dir}"
    assert test_lbl_dir.exists(), f"Test labels not found: {test_lbl_dir}"
    print(f"Test set: {test_img_dir}")

    # load frozen detector
    print(f"\nLoading detector: {args.detector_weights}")
    detector = YOLO(str(args.detector_weights))

    # output dirs
    ind_dir = args.output / "individual"
    ind_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for clf_name in args.classifiers:
        weights_path = args.classifier_dir / clf_name / "weights" / "best.pt"
        if not weights_path.exists():
            print(f"\n  [skip] {clf_name}: weights not found at {weights_path}")
            continue

        print(f"\n{'─'*60}")
        print(f"  Evaluating: {clf_name}")
        print(f"{'─'*60}")

        classifier = load_classifier(clf_name, weights_path, device)

        result = evaluate_combined(
            detector     = detector,
            classifier   = classifier,
            clf_name     = clf_name,
            test_img_dir = test_img_dir,
            test_lbl_dir = test_lbl_dir,
            device       = device,
            imgsz        = args.imgsz,
            det_conf     = args.det_conf,
            det_iou      = args.det_iou,
        )
        all_results.append(result)

        # save individual result
        out_path = ind_dir / f"B_{clf_name}_combined.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

        # free classifier memory before next one
        del classifier
        torch.cuda.empty_cache()

    # print comparison table
    print(f"\n{'='*60}")
    print("PATH B COMBINED — FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Classifier':<25} {'mAP50':>7} {'mAP50-95':>9} {'FPS':>6}")
    print(f"{'─'*50}")
    for r in sorted(all_results, key=lambda x: -x["mAP50"]):
        print(f"  {r['classifier']:<23} "
              f"{r['mAP50']:>7.4f} "
              f"{r['mAP50_95']:>9.4f} "
              f"{r['fps']:>6.1f}")
    print(f"{'='*60}")

    # save combined summary
    summary_path = args.output / "path_B_combined_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
