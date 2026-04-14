"""
validate_all.py

Validates all benchmark scripts for syntax errors, logic issues,
and correct state. Also validates external dataset downloads.

Usage:
  python validate_all.py
  python validate_all.py --check_downloads \
      --external_root ~/V-JEPA-2/mrosado/thrashscan/external_datasets
  python validate_all.py --all \
      --base_dir      ~/V-JEPA-2/mrosado/thrashscan \
      --external_root ~/V-JEPA-2/mrosado/thrashscan/external_datasets \
      --processed_dir ~/V-JEPA-2/mrosado/thrashscan/processed_4cls \
      --results_dir   ~/V-JEPA-2/mrosado/thrashscan/results
"""

import argparse
import ast
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

COARSE_CLASSES_4 = ["plastic", "paper", "metal", "other"]
COARSE_CLASSES_7 = ["plastic", "paper", "metal", "glass",
                     "organic", "hazardous", "other"]

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"

results = []


def log(status, check, detail=""):
    symbol = {"PASS": "✓", "FAIL": "✗", "WARN": "!", "INFO": "i"}.get(status, "?")
    line = f"  {symbol}  {check}"
    if detail:
        line += f"\n       {detail}"
    print(line)
    results.append((status, check))


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTAX CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_syntax(base_dir: Path):
    print("\n── Syntax checks ──────────────────────────────────────────")
    scripts = [
        "preprocess.py",
        "train_path_A.py",
        "train_path_B.py",
        "evaluate.py",
        "merge_datasets.py",
        "download_external_datasets.py",
        "validate_all.py",
    ]
    for script in scripts:
        path = base_dir / script
        if not path.exists():
            log("WARN", f"{script} not found")
            continue
        try:
            with open(path) as f:
                source = f.read()
            ast.parse(source)
            log("PASS", f"{script} syntax OK")
        except SyntaxError as e:
            log("FAIL", f"{script} syntax error",
                f"Line {e.lineno}: {e.msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESS.PY LOGIC CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_preprocess(base_dir: Path):
    print("\n── preprocess.py logic ────────────────────────────────────")
    path = base_dir / "preprocess.py"
    if not path.exists():
        log("WARN", "preprocess.py not found")
        return

    with open(path) as f:
        content = f.read()

    # 4-class taxonomy
    if 'COARSE_CLASSES = ["plastic", "paper", "metal", "other"]' in content:
        log("PASS", "4-class taxonomy correctly set")
    elif 'COARSE_CLASSES = ["plastic", "paper", "metal", "glass"' in content:
        log("FAIL", "Still using 7-class taxonomy — run the taxonomy patch")
    else:
        log("WARN", "COARSE_CLASSES not found in expected format")

    # 4-class mapping
    checks = [
        ('"Battery":               "other"',  "Battery → other"),
        ('"Broken glass":          "other"',  "Broken glass → other"),
        ('"Food waste":            "other"',  "Food waste → other"),
        ('"Glass jar":             "other"',  "Glass jar → other"),
    ]
    for pattern, label in checks:
        if pattern in content:
            log("PASS", f"Mapping: {label}")
        else:
            log("FAIL", f"Missing mapping: {label}")

    # key functions present
    for fn in ["letterbox", "copy_paste_oversample", "tile_image",
               "compute_class_weights", "run_preprocessing",
               "write_yolo_yaml"]:
        if f"def {fn}" in content:
            log("PASS", f"Function {fn} present")
        else:
            log("FAIL", f"Function {fn} missing")

    # fixed seed
    if "random_state=42" in content and "seed=42" in content:
        log("PASS", "Fixed seed=42 used in splits")
    else:
        log("WARN", "Check seed consistency in StratifiedShuffleSplit")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN_PATH_A.PY LOGIC CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_train_A(base_dir: Path):
    print("\n── train_path_A.py logic ──────────────────────────────────")
    path = base_dir / "train_path_A.py"
    if not path.exists():
        log("WARN", "train_path_A.py not found")
        return

    with open(path) as f:
        content = f.read()

    checks = [
        ("yolov5su",             "yolov5su in registry"),
        ("yolov9s",              "yolov9s in registry"),
        ("yolov9e",              "yolov9e in registry"),
        ("yolov8m",              "yolov8m in registry"),
        ("yolov11m",             "yolov11m in registry"),
        ("patience",             "patience parameter present"),
        ("AdamW",                "AdamW optimizer"),
        ("seed",                 "fixed seed"),
        ("augment",              "augmentation enabled"),
        ("mosaic",               "mosaic augmentation"),
        ("mlflow",               "MLflow logging"),
        ("metrics.json",         "metrics.json saved"),
        ("def train_single",     "train_single function"),
        ("def print_summary",    "print_summary function"),
        ("def measure_latency",  "latency measurement"),
    ]
    for pattern, label in checks:
        if pattern in content:
            log("PASS", label)
        else:
            log("FAIL", f"Missing: {label}")

    # check device fix is applied
    if "device_obj = torch.device" in content:
        log("PASS", "Latency device fix applied")
    else:
        log("WARN", "Latency device fix may be missing")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN_PATH_B.PY LOGIC CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_train_B(base_dir: Path):
    print("\n── train_path_B.py logic ──────────────────────────────────")
    path = base_dir / "train_path_B.py"
    if not path.exists():
        log("WARN", "train_path_B.py not found")
        return

    with open(path) as f:
        content = f.read()

    checks = [
        ('COARSE_CLASSES = ["plastic", "paper", "metal", "other"]',
         "4-class taxonomy"),
        ("NUM_CLASSES    = len(COARSE_CLASSES)",
         "NUM_CLASSES derived from COARSE_CLASSES"),
        ("resnet50",              "ResNet-50 in classifier registry"),
        ("vit_b16",               "ViT-B/16 in classifier registry"),
        ("CropDataset",           "CropDataset class"),
        ("def train_classifier",  "train_classifier function"),
        ("def evaluate_on_test",  "evaluate_on_test function"),
        ("CrossEntropyLoss",      "weighted cross entropy loss"),
        ("CosineAnnealingLR",     "cosine LR scheduler"),
        ("confusion_matrix",      "confusion matrix"),
        ("average_precision_score","per-class AP"),
        ("mlflow",                "MLflow logging"),
        ("metrics.json",          "metrics.json saved"),
        ("history.csv",           "training history saved"),
        ("best.pt",               "best weights saved"),
        ("early stopping",        "early stopping logic"),
    ]
    for pattern, label in checks:
        status = "PASS" if pattern in content else "FAIL"
        log(status, label)


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATE.PY LOGIC CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_evaluate(base_dir: Path):
    print("\n── evaluate.py logic ──────────────────────────────────────")
    path = base_dir / "evaluate.py"
    if not path.exists():
        log("WARN", "evaluate.py not found")
        return

    with open(path) as f:
        content = f.read()

    checks = [
        ("def evaluate_model",       "evaluate_model function"),
        ("def build_global_summary", "build_global_summary function"),
        ("def collect_predictions",  "collect_predictions function"),
        ("def compute_ap_ar",        "compute_ap_ar function"),
        ("def measure_latency",      "measure_latency function"),
        ("def plot_confusion_matrix","confusion matrix plot"),
        ("def plot_pr_curves",       "PR curve plot"),
        ("def save_training_curve",  "training curve plot"),
        ("def read_training_config", "training config reader"),
        ("args.path == \"B\"",      "Path B handling"),
        ("benchmark_summary.csv",    "global CSV output"),
        ("benchmark_summary.json",   "global JSON output"),
        ("benchmark_bar_chart.png",  "bar chart output"),
        ("individual",               "individual results folder"),
        ("_summary.csv",             "per-model summary CSV"),
        ("_history.csv",             "per-model history CSV"),
        ("IOU_THRESHOLDS",           "IoU threshold range"),
        ("mAP50_95",                 "mAP50:0.95 metric"),
        ("latency_ms",               "latency metric"),
        ("model_params_M",           "model params metric"),
        ("gpu_memory_MB",            "GPU memory metric"),
    ]
    for pattern, label in checks:
        status = "PASS" if pattern in content else "FAIL"
        log(status, label)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MERGE_DATASETS.PY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_merge(base_dir: Path):
    print("\n── merge_datasets.py logic ────────────────────────────────")
    path = base_dir / "merge_datasets.py"
    if not path.exists():
        log("WARN", "merge_datasets.py not found")
        return

    with open(path) as f:
        content = f.read()

    checks = [
        ("def merge_coco_jsons",          "merge_coco_jsons function"),
        ("def write_merged_taco_structure","taco structure writer"),
        ("global_img_id",                 "global image ID counter"),
        ("global_ann_id",                 "global annotation ID counter"),
        ("remap_cat",                     "category remapping"),
        ("subprocess",                    "preprocess.py subprocess call"),
        ("coco_format",                   "coco_format dir scanning"),
    ]
    for pattern, label in checks:
        status = "PASS" if pattern in content else "FAIL"
        log(status, label)


# ─────────────────────────────────────────────────────────────────────────────
# 7. DOWNLOAD SCRIPT CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_download_script(base_dir: Path):
    print("\n── download_external_datasets.py logic ────────────────────")
    path = base_dir / "download_external_datasets.py"
    if not path.exists():
        log("WARN", "download_external_datasets.py not found")
        return

    with open(path) as f:
        content = f.read()

    checks = [
        ("def download_open_litter_map",    "OLM download function"),
        ("def download_drinking_waste",     "Drinking Waste download function"),
        ("def download_mju_waste",          "MJU-Waste download function"),
        ("def process_mju_waste",           "MJU-Waste processing function"),
        ("def _voc_to_coco_mju",            "VOC→COCO converter for MJU"),
        ("def convert_roboflow_dataset",    "Roboflow converter"),
        ("def yolo_to_coco",                "YOLO→COCO converter"),
        ("gdown",                           "gdown for Google Drive"),
        ("roboflow",                        "Roboflow API"),
        ("mju_only",                        "--mju_only flag"),
        ("roboflow_key",                    "--roboflow_key flag"),
        ("skip_olm",                        "--skip_olm flag"),
        ("def print_download_summary",      "download summary function"),
    ]
    for pattern, label in checks:
        status = "PASS" if pattern in content else "FAIL"
        log(status, label)


# ─────────────────────────────────────────────────────────────────────────────
# 8. PROCESSED DATA CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_processed_data(processed_dir: Path):
    print("\n── Processed data validation ──────────────────────────────")
    if not processed_dir.exists():
        log("WARN", f"Processed dir not found: {processed_dir}")
        return

    # check YAML
    for path_tag in ["path_A", "path_B"]:
        yaml_path = processed_dir / f"dataset_{path_tag}.yaml"
        if yaml_path.exists():
            import yaml as pyyaml
            with open(yaml_path) as f:
                cfg = pyyaml.safe_load(f)
            nc = cfg.get("nc", 0)
            names = cfg.get("names", [])
            if nc == 4 and set(names) == set(COARSE_CLASSES_4):
                log("PASS", f"{path_tag} YAML: 4 classes correctly set")
            elif nc == 7:
                log("FAIL", f"{path_tag} YAML: still 7 classes — needs regeneration")
            else:
                log("WARN", f"{path_tag} YAML: nc={nc} names={names}")
        else:
            log("WARN", f"dataset_{path_tag}.yaml not found")

    # check split sizes
    for split in ["train", "val", "test"]:
        for path_tag in ["path_A", "path_B"]:
            img_dir = processed_dir / split / path_tag / "images"
            lbl_dir = processed_dir / split / path_tag / "labels"
            if not img_dir.exists():
                continue
            n_imgs = len(list(img_dir.glob("*.jpg")))
            n_lbls = len(list(lbl_dir.glob("*.txt"))) \
                     if lbl_dir.exists() else 0
            if n_imgs == n_lbls and n_imgs > 0:
                log("PASS", f"{split}/{path_tag}: {n_imgs} images, "
                             f"{n_lbls} labels match")
            else:
                log("FAIL", f"{split}/{path_tag}: {n_imgs} images vs "
                             f"{n_lbls} labels — mismatch")

    # check meta files
    for meta_file in ["split_indices.npz", "class_weights.yaml",
                      "class_map.json"]:
        path = processed_dir / "meta" / meta_file
        if path.exists():
            log("PASS", f"meta/{meta_file} exists")
        else:
            log("WARN", f"meta/{meta_file} not found")

    # check class weights are 4-class
    cw_path = processed_dir / "meta" / "class_weights.yaml"
    if cw_path.exists():
        import yaml as pyyaml
        with open(cw_path) as f:
            cw = pyyaml.safe_load(f)
        weights = cw.get("coarse_weights", {})
        if set(weights.keys()) == set(COARSE_CLASSES_4):
            log("PASS", f"class_weights.yaml has correct 4 classes")
        else:
            log("FAIL", f"class_weights.yaml has wrong classes: "
                         f"{list(weights.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. EXTERNAL DATASET DOWNLOAD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def check_downloads(external_root: Path):
    print("\n── External dataset download validation ───────────────────")
    if not external_root.exists():
        log("WARN", f"External root not found: {external_root}")
        log("INFO", "Run: python download_external_datasets.py --mju_only")
        return

    coco_root = external_root / "coco_format"

    # check each expected dataset
    datasets = {
        "open_litter_map": {
            "min_images": 500,
            "description": "Open Litter Map",
        },
        "drinking_waste": {
            "min_images": 500,
            "description": "Drinking Waste Classification",
        },
        "mju_waste": {
            "min_images": 200,
            "description": "MJU-Waste",
        },
    }

    total_external = 0

    for ds_name, ds_info in datasets.items():
        ds_dir = coco_root / ds_name if coco_root.exists() else None
        if ds_dir is None or not ds_dir.exists():
            log("WARN", f"{ds_info['description']}: not downloaded yet",
                "Run download_external_datasets.py to download")
            continue

        # count images and annotations
        json_files = list(ds_dir.rglob("*.json"))
        n_images   = 0
        n_anns     = 0
        cls_counts = Counter()

        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                n_images += len(data.get("images", []))
                for ann in data.get("annotations", []):
                    cid = ann.get("category_id", 3)
                    cls_name = COARSE_CLASSES_4[cid] \
                               if cid < len(COARSE_CLASSES_4) else "unknown"
                    cls_counts[cls_name] += 1
                    n_anns += 1
            except Exception:
                pass

        # check actual image files
        img_files = list(ds_dir.rglob("*.jpg")) + \
                    list(ds_dir.rglob("*.jpeg")) + \
                    list(ds_dir.rglob("*.png"))

        if len(img_files) >= ds_info["min_images"]:
            log("PASS", f"{ds_info['description']}: "
                         f"{len(img_files)} images, {n_anns} annotations")
        else:
            log("FAIL", f"{ds_info['description']}: only {len(img_files)} images "
                         f"(expected ≥{ds_info['min_images']})")

        if cls_counts:
            log("INFO", f"  Class distribution: "
                         + ", ".join(f"{k}={v}"
                                     for k,v in sorted(cls_counts.items())))

        total_external += len(img_files)

    print(f"\n  Total external images available: {total_external}")
    if total_external > 2000:
        log("PASS", f"External data sufficient for meaningful training boost")
    elif total_external > 0:
        log("WARN", f"Only {total_external} external images — "
                     "try downloading more datasets")
    else:
        log("WARN", "No external images found — using TACO only")


# ─────────────────────────────────────────────────────────────────────────────
# 10. RESULTS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def check_results(results_dir: Path):
    print("\n── Results validation ─────────────────────────────────────")
    if not results_dir.exists():
        log("WARN", f"Results dir not found: {results_dir}")
        return

    ind_dir    = results_dir / "individual"
    global_dir = results_dir / "global"

    if not ind_dir.exists():
        log("WARN", "No individual results yet — run evaluate.py")
        return

    json_files = list(ind_dir.glob("*.json"))
    log("INFO", f"{len(json_files)} individual result files found")

    paths_seen = set()
    for jf in json_files:
        try:
            with open(jf) as f:
                r = json.load(f)
            path_tag  = r.get("path", "?")
            model_key = r.get("model_key", "?")
            map50     = r.get("mAP50", None)
            acc       = r.get("accuracy", None)
            paths_seen.add(path_tag)

            if map50 is not None:
                log("PASS", f"Path {path_tag} / {model_key}: "
                             f"mAP50={map50:.4f}")
            elif acc is not None:
                log("PASS", f"Path {path_tag} / {model_key}: "
                             f"accuracy={acc:.4f}")
            else:
                log("WARN", f"Path {path_tag} / {model_key}: "
                             "no primary metric found")
        except Exception as e:
            log("FAIL", f"Could not read {jf.name}: {e}")

    if global_dir.exists():
        csv_path = global_dir / "benchmark_summary.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            log("PASS", f"Global summary CSV: {len(df)} rows")
            if "rank" in df.columns:
                best = df.iloc[0]
                log("INFO", f"Current leader: "
                             f"{best.get('model_key','?')} "
                             f"mAP50={best.get('mAP50',0):.4f}")
        else:
            log("WARN", "Global summary CSV not generated yet — "
                         "run: python evaluate.py --summarize")
    else:
        log("WARN", "No global results yet")


# ─────────────────────────────────────────────────────────────────────────────
# 11. IMPORT CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_imports():
    print("\n── Import / dependency checks ─────────────────────────────")
    deps = [
        ("torch",             "PyTorch"),
        ("torchvision",       "TorchVision"),
        ("ultralytics",       "Ultralytics"),
        ("timm",              "timm"),
        ("albumentations",    "Albumentations"),
        ("cv2",               "OpenCV"),
        ("numpy",             "NumPy"),
        ("sklearn",           "scikit-learn"),
        ("mlflow",            "MLflow"),
        ("pycocotools",       "pycocotools"),
        ("pandas",            "pandas"),
        ("matplotlib",        "matplotlib"),
        ("yaml",              "PyYAML"),
        ("tqdm",              "tqdm"),
        ("PIL",               "Pillow"),
        ("einops",            "einops"),
    ]
    for module, name in deps:
        try:
            importlib.import_module(module)
            log("PASS", f"{name} importable")
        except ImportError:
            log("FAIL", f"{name} not installed — pip install {module}")

    # version checks
    try:
        import numpy as np
        if np.__version__.startswith("1."):
            log("PASS", f"NumPy {np.__version__} (compatible with ultralytics)")
        else:
            log("WARN", f"NumPy {np.__version__} — ultralytics may conflict")
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            log("PASS", f"CUDA available: {dev.name} "
                         f"({dev.total_memory/1e9:.1f}GB)")
        else:
            log("WARN", "CUDA not available")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary():
    pass_count = sum(1 for s,_ in results if s == "PASS")
    fail_count = sum(1 for s,_ in results if s == "FAIL")
    warn_count = sum(1 for s,_ in results if s == "WARN")

    print("\n" + "="*60)
    print(f"VALIDATION SUMMARY")
    print("="*60)
    print(f"  PASS : {pass_count}")
    print(f"  FAIL : {fail_count}")
    print(f"  WARN : {warn_count}")
    print("="*60)

    if fail_count > 0:
        print("\nFailed checks:")
        for status, check in results:
            if status == "FAIL":
                print(f"  ✗  {check}")

    if warn_count > 0:
        print("\nWarnings:")
        for status, check in results:
            if status == "WARN":
                print(f"  !  {check}")

    if fail_count == 0:
        print("\nAll critical checks passed.")
    else:
        print(f"\n{fail_count} critical issues need fixing before training.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Validate all benchmark scripts")
    p.add_argument("--base_dir",      type=Path,
                   default=Path("/home/ubuntu/V-JEPA-2/mrosado/thrashscan"))
    p.add_argument("--external_root", type=Path, default=None)
    p.add_argument("--processed_dir", type=Path, default=None)
    p.add_argument("--results_dir",   type=Path, default=None)
    p.add_argument("--check_downloads", action="store_true",
                   help="Check external dataset downloads")
    p.add_argument("--check_processed", action="store_true",
                   help="Check processed data directory")
    p.add_argument("--check_results",   action="store_true",
                   help="Check results directory")
    p.add_argument("--all",           action="store_true",
                   help="Run all checks")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    check_syntax(args.base_dir)
    check_imports()
    check_preprocess(args.base_dir)
    check_train_A(args.base_dir)
    check_train_B(args.base_dir)
    check_evaluate(args.base_dir)
    check_merge(args.base_dir)
    check_download_script(args.base_dir)

    if args.check_downloads or args.all:
        ext = args.external_root or args.base_dir / "external_datasets"
        check_downloads(ext)

    if args.check_processed or args.all:
        proc = args.processed_dir or args.base_dir / "processed_4cls"
        check_processed_data(proc)

    if args.check_results or args.all:
        res = args.results_dir or args.base_dir / "results"
        check_results(res)

    print_final_summary()
