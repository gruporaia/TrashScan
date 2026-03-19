"""
preprocess.py  —  Standardized TACO preprocessing for all 3 benchmark paths.

Produces:
  data/
    meta/
      class_map.json          # TACO cat_id -> coarse/fine label mappings
      split_indices.npz       # fixed-seed train/val/test image indices
      class_weights.yaml      # 1/sqrt(freq_c) normalized weights per taxonomy
      aug_config.yaml         # Albumentations config snapshot
    {train,val,test}/
      path_A/
        images/               # 640x640 letterboxed .jpg
        labels/               # YOLO .txt (cx cy w h cls, normalized)
      path_B/
        images/               # 640x640 letterboxed .jpg  (detect stage)
        labels/               # YOLO .txt
        crops/                # 224x224 instance crops     (classify stage)
      path_C/
        tiles/                # 224x224 sliding-window tiles
        tile_meta.json        # tile -> source image + bbox mapping

Usage:
  python preprocess.py --taco_root /data/TACO --output_root /data/processed --path all
  python preprocess.py --taco_root /data/TACO --output_root /data/processed --path A
"""

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml
from pycocotools.coco import COCO
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CLASS TAXONOMY
#     Three levels derived from TACO's 60 native categories.
#     Edit COARSE_MAP to adjust merging logic; FINE_MIN_SAMPLES controls
#     the minimum annotations threshold for a class to stay in fine taxonomy.
# ─────────────────────────────────────────────────────────────────────────────

FINE_MIN_SAMPLES = 20   # classes with fewer train instances -> merged to "other"

# TACO supercategory -> coarse class label
# Based on TACO paper taxonomy (arXiv:2003.06975)
SUPERCATEGORY_TO_COARSE = {
    "Aluminium foil":        "metal",
    "Battery":               "other",
    "Blister pack":          "plastic",
    "Bottle":                "plastic",
    "Bottle cap":            "metal",
    "Broken glass":          "other",
    "Can":                   "metal",
    "Carton":                "paper",
    "Cigarette":             "other",
    "Cup":                   "plastic",
    "Drink can":             "metal",
    "Drink carton":          "paper",
    "Food Can":              "metal",
    "Food waste":            "other",
    "Garbage bag":           "plastic",
    "Glass jar":             "other",
    "Lid":                   "plastic",
    "Magazine paper":        "paper",
    "Meal carton":           "paper",
    "Metal bottle cap":      "metal",
    "Metal lid":             "metal",
    "Other carton":          "paper",
    "Other glass":           "other",
    "Other metal":           "metal",
    "Other plastic":         "plastic",
    "Other plastic bag":     "plastic",
    "Other plastic bottle":  "plastic",
    "Other plastic container":"plastic",
    "Other plastic cup":     "plastic",
    "Other plastic wrapper": "plastic",
    "Plastic bag & wrapper": "plastic",
    "Plastic bottle cap":    "plastic",
    "Plastic film":          "plastic",
    "Plastic straw":         "plastic",
    "Plastic utensils":      "plastic",
    "Polystyrene item":      "plastic",
    "Pop tab":               "metal",
    "Rope & strings":        "other",
    "Scrap metal":           "metal",
    "Shoe":                  "other",
    "Single-use carrier bag":"plastic",
    "Six pack rings":        "plastic",
    "Spread tub":            "plastic",
    "Squeezable tube":       "plastic",
    "Styrofoam piece":       "plastic",
    "Tissues":               "paper",
    "Toilet tube":           "paper",
    "Trash bag":             "plastic",
    "Unlabeled litter":      "other",
    "Wrapping paper":        "paper",
}

COARSE_CLASSES = ["plastic", "paper", "metal", "other"]
BINARY_CLASSES  = ["litter"]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LETTERBOX  (shared by Path A and B detect stage)
# ─────────────────────────────────────────────────────────────────────────────

def letterbox(img: np.ndarray, target: int = 640, fill: int = 114):
    """
    Resize image preserving aspect ratio; pad to square with gray fill=114.
    Returns (padded_img, scale, (pad_w_left, pad_h_top)).
    """
    h, w = img.shape[:2]
    scale = min(target / h, target / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = target - new_h
    pad_w = target - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right  = pad_w // 2, pad_w - pad_w // 2

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(fill, fill, fill)
    )
    return padded, scale, (left, top)


def adapt_bboxes_to_letterbox(bboxes_xywh, orig_hw, scale, pad_lt, target=640):
    """
    Transform COCO [x,y,w,h] pixel bboxes into YOLO [cx,cy,w,h] normalized
    after letterbox transform.  Returns list of (cx, cy, w, h) in [0,1].
    """
    oh, ow = orig_hw
    pl, pt = pad_lt
    adapted = []
    for x, y, bw, bh in bboxes_xywh:
        # clip to image boundary
        x  = max(0, x);  y  = max(0, y)
        bw = min(bw, ow - x);  bh = min(bh, oh - y)
        if bw <= 0 or bh <= 0:
            continue
        # scale + pad
        x1 = x  * scale + pl
        y1 = y  * scale + pt
        x2 = (x + bw) * scale + pl
        y2 = (y + bh) * scale + pt
        cx = (x1 + x2) / 2 / target
        cy = (y1 + y2) / 2 / target
        nw = (x2 - x1) / target
        nh = (y2 - y1) / target
        if nw > 0 and nh > 0:
            adapted.append((
                round(cx, 6), round(cy, 6),
                round(nw, 6), round(nh, 6)
            ))
    return adapted


# ─────────────────────────────────────────────────────────────────────────────
# 3.  AUGMENTATION PIPELINE
#     Applied to train split only.  Val / test: resize + normalize only.
# ─────────────────────────────────────────────────────────────────────────────

def build_train_augmentation(path_tag: str) -> A.Compose:
    """
    Returns an Albumentations pipeline with bbox-safe transforms.
    path_tag: 'A' | 'B' | 'C'  — MIM masking only added for C.
    """
    bbox_params = A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
        clip=True,
    )

    base_transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT,
                 value=(114, 114, 114), p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=0, p=0.3),
        # Photometric — critical for outdoor lighting variation
        A.RandomBrightnessContrast(brightness_limit=0.3,
                                   contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15,
                             sat_shift_limit=30,
                             val_shift_limit=20, p=0.4),
        A.CLAHE(clip_limit=2.0, p=0.2),
        # Noise / blur — simulate real-world capture conditions
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.ImageCompression(quality_lower=75, p=0.2),
        # MixUp / CutMix approximation via CoarseDropout (single-image safe)
        A.CoarseDropout(
            max_holes=4, max_height=40, max_width=40,
            min_holes=1, fill_value=114, p=0.3
        ),
    ]

    return A.Compose(base_transforms, bbox_params=bbox_params)


def build_val_transform(target: int = 640) -> A.Compose:
    """Val / test: only normalize.  Letterbox is applied separately."""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 4.  COPY-PASTE OVERSAMPLING
#     Paste rare-class instances onto random background images.
#     Cap: total synthetic images per class <= 3x majority class count.
# ─────────────────────────────────────────────────────────────────────────────

def copy_paste_oversample(
    image_ids: list,
    img_to_annotations: dict,
    img_to_path: dict,
    coarse_label_per_img: dict,
    class_counts: dict,
    target_dir_img: Path,
    target_dir_lbl: Path,
    id_to_coarse: dict,
    rng: random.Random,
) -> list:
    """
    Brings each minority class up to median_count * 2.
    Images are pre-loaded into memory per class to avoid repeated disk reads.
    A tqdm bar per class shows live progress.
    """
    counts_sorted  = sorted(class_counts.values())
    median_count   = counts_sorted[len(counts_sorted) // 2]
    majority_count = max(class_counts.values())
    # cap: bring minority classes up to 75% of majority
    # avoids OOM on large merged datasets while still balancing
    cap = int(majority_count * 0.75)

    print(f"  Oversampling cap: {cap}  (2x median={median_count})")
    for cls, cnt in sorted(class_counts.items()):
        will_gen = max(0, cap - cnt)
        print(f"    {cls:<12} have={cnt:<5} generate={will_gen:<5} final={cnt+will_gen}")

    synthetic_records = []

    for cls_name, count in class_counts.items():
        need = cap - count
        if need <= 0:
            print(f"  [{cls_name}] already at/above cap, skipping")
            continue

        # donor ids: images that contain this class
        donor_ids = [
            iid for iid in image_ids
            if any(
                id_to_coarse.get(a["category_id"]) == cls_name
                for a in img_to_annotations.get(iid, [])
            )
        ]
        if not donor_ids:
            print(f"  [{cls_name}] no donor images found, skipping")
            continue

        # pre-load donor images into memory once for this class
        donor_cache = {}
        for did in donor_ids:
            img = cv2.imread(str(img_to_path[did]))
            if img is not None:
                donor_cache[did] = img
        if not donor_cache:
            continue
        valid_donor_ids = list(donor_cache.keys())

        # pre-load a capped background pool to limit RAM usage
        bg_pool_ids = rng.sample(image_ids, min(200, len(image_ids)))
        bg_cache = {}
        for bid in bg_pool_ids:
            img = cv2.imread(str(img_to_path[bid]))
            if img is not None:
                bg_cache[bid] = img
        valid_bg_ids = list(bg_cache.keys())
        if not valid_bg_ids:
            continue

        for i in tqdm(range(need), desc=f"  oversampling [{cls_name}]", leave=True):
            src_id  = rng.choice(valid_donor_ids)
            bg_id   = rng.choice(valid_bg_ids)
            src_img = donor_cache[src_id]
            bg_img  = bg_cache[bg_id]

            donor_anns = [
                a for a in img_to_annotations.get(src_id, [])
                if id_to_coarse.get(a["category_id"]) == cls_name
            ]
            if not donor_anns:
                continue
            ann = rng.choice(donor_anns)

            x, y, bw, bh = [int(v) for v in ann["bbox"]]
            sh, sw = src_img.shape[:2]
            x  = max(0, x);  y  = max(0, y)
            bw = min(bw, sw - x);  bh = min(bh, sh - y)
            if bw <= 0 or bh <= 0:
                continue

            patch = src_img[y:y+bh, x:x+bw]
            bgh, bgw = bg_img.shape[:2]

            patch = patch[:min(bh, bgh), :min(bw, bgw)]
            bh, bw = patch.shape[:2]
            if bh <= 0 or bw <= 0:
                continue

            px  = rng.randint(0, max(0, bgw - bw - 1))
            py  = rng.randint(0, max(0, bgh - bh - 1))
            py2 = min(py + bh, bgh)
            px2 = min(px + bw, bgw)
            bh  = py2 - py
            bw  = px2 - px
            patch = patch[:bh, :bw]

            composite = bg_img.copy()
            composite[py:py2, px:px2] = patch

            lb_img, scale, pad_lt = letterbox(composite)

            syn_stem = f"syn_{cls_name}_{bg_id}_{src_id}_{i}"
            img_out  = target_dir_img / f"{syn_stem}.jpg"
            lbl_out  = target_dir_lbl / f"{syn_stem}.txt"

            cv2.imwrite(str(img_out), lb_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            bboxes_adapted = adapt_bboxes_to_letterbox(
                [(px, py, bw, bh)],
                (bgh, bgw), scale, pad_lt
            )
            coarse_idx = COARSE_CLASSES.index(cls_name)
            with open(lbl_out, "w") as f:
                for (cx, cy, nw, nh) in bboxes_adapted:
                    f.write(f"{coarse_idx} {cx} {cy} {nw} {nh}\n")

            synthetic_records.append((img_out, lbl_out))

    return synthetic_records

def tile_image(img: np.ndarray, tile_size: int = 224, overlap: float = 0.5):
    """
    Yields (tile_img, x_offset, y_offset) for each sliding window position.
    """
    h, w = img.shape[:2]
    stride = int(tile_size * (1 - overlap))
    for y in range(0, max(1, h - tile_size + 1), stride):
        for x in range(0, max(1, w - tile_size + 1), stride):
            tile = img[y:y+tile_size, x:x+tile_size]
            # pad if tile is smaller than tile_size (edge case)
            th, tw = tile.shape[:2]
            if th < tile_size or tw < tile_size:
                tile = cv2.copyMakeBorder(
                    tile, 0, tile_size-th, 0, tile_size-tw,
                    cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )
            yield tile, x, y


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLASS WEIGHT COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(class_counts: dict) -> dict:
    """
    w_c = 1 / sqrt(freq_c),  normalized so sum(w) == len(classes).
    """
    total = sum(class_counts.values())
    raw   = {c: 1.0 / math.sqrt(count / total)
             for c, count in class_counts.items() if count > 0}
    mean_w = sum(raw.values()) / len(raw)
    return {c: round(w / mean_w, 6) for c, w in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(taco_root: Path, output_root: Path, paths: list[str]):
    rng = random.Random(42)
    np.random.seed(42)

    ann_file = taco_root / "data" / "annotations.json"
    assert ann_file.exists(), f"Annotation file not found: {ann_file}"

    coco   = COCO(str(ann_file))
    img_ids = sorted(coco.getImgIds())

    # ── build category mappings ───────────────────────────────────────────
    categories = {c["id"]: c for c in coco.loadCats(coco.getCatIds())}

    # detect if already a 4-class merged JSON — skip TACO remapping
    cat_names     = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    already_mapped = set(cat_names.values()) <= set(COARSE_CLASSES)

    if already_mapped:
        print("  Detected pre-mapped 4-class annotations — skipping TACO remapping")
        id_to_coarse = {cid: name for cid, name in cat_names.items()
                        if name in COARSE_CLASSES}
        id_to_fine   = cat_names
    else:
        # native cat_id -> coarse label (original TACO 60-class remapping)
        id_to_coarse = {}
        for cid, cat in categories.items():
            supcat = cat.get("supercategory", cat["name"])
            id_to_coarse[cid] = SUPERCATEGORY_TO_COARSE.get(supcat, "other")
        # native cat_id -> fine label
        id_to_fine = {cid: cat["name"] for cid, cat in categories.items()}

    # ── per-image data ─────────────────────────────────────────────────────
    img_meta          = {im["id"]: im for im in coco.loadImgs(img_ids)}
    img_to_annotations: dict[int, list] = defaultdict(list)
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_to_annotations[ann["image_id"]].append(ann)

    # map image_id -> file path
    img_to_path = {}
    for iid, im in img_meta.items():
        fname = im["file_name"]
        # try flat images/ folder first (merged dataset)
        p = taco_root / "data" / "images" / Path(fname).name
        if not p.exists():
            # fallback: TACO batch structure e.g. batch_1/000001.jpg
            p = taco_root / "data" / fname
        img_to_path[iid] = p

    # drop images with no annotations
    img_ids = [iid for iid in img_ids if img_to_annotations.get(iid)]

    # ── coarse label per image (majority class for stratification) ────────
    coarse_label_per_img = {}
    for iid in img_ids:
        counts = defaultdict(int)
        for ann in img_to_annotations[iid]:
            counts[id_to_coarse[ann["category_id"]]] += 1
        coarse_label_per_img[iid] = max(counts, key=counts.get)

    # ── merge singleton classes into "other" for stratification ──────────
    from collections import Counter
    label_counts = Counter(coarse_label_per_img.values())
    print("Class counts before merge:", dict(label_counts))
    for iid in img_ids:
        if label_counts[coarse_label_per_img[iid]] < 2:
            coarse_label_per_img[iid] = "other"
    label_counts_after = Counter(coarse_label_per_img.values())
    print("Class counts after merge:", dict(label_counts_after))

    # ── stratified split ──────────────────────────────────────────────────
    img_ids_arr = np.array(img_ids)
    labels_arr  = np.array([coarse_label_per_img[i] for i in img_ids])

    # 70 / 15 / 15
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(img_ids_arr, labels_arr))
    temp_ids   = img_ids_arr[temp_idx]
    temp_labels = labels_arr[temp_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_rel_idx, test_rel_idx = next(sss2.split(temp_ids, temp_labels))

    train_ids = img_ids_arr[train_idx].tolist()
    val_ids   = temp_ids[val_rel_idx].tolist()
    test_ids  = temp_ids[test_rel_idx].tolist()

    print(f"Split → train: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}")

    # ── persist meta ──────────────────────────────────────────────────────
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        meta_dir / "split_indices.npz",
        train=np.array(train_ids),
        val=np.array(val_ids),
        test=np.array(test_ids),
    )

    # class_map.json
    class_map = {
        "coarse": {str(cid): id_to_coarse[cid] for cid in id_to_coarse},
        "fine":   {str(cid): id_to_fine[cid]   for cid in id_to_fine},
        "coarse_classes": COARSE_CLASSES,
        "binary_classes": BINARY_CLASSES,
    }
    (meta_dir / "class_map.json").write_text(json.dumps(class_map, indent=2))

    # ── class counts on train split ───────────────────────────────────────
    train_coarse_counts: dict[str, int] = defaultdict(int)
    for iid in train_ids:
        for ann in img_to_annotations[iid]:
            train_coarse_counts[id_to_coarse[ann["category_id"]]] += 1

    for cls in COARSE_CLASSES:
        train_coarse_counts.setdefault(cls, 0)

    class_weights = compute_class_weights(
        {c: v for c, v in train_coarse_counts.items() if v > 0}
    )
    (meta_dir / "class_weights.yaml").write_text(
        yaml.dump({"coarse_weights": class_weights}, default_flow_style=False)
    )
    print("Class weights:", class_weights)

    # ── build per-path data ───────────────────────────────────────────────
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    if "A" in paths or "B" in paths:
        _process_detect_paths(
            splits, img_to_path, img_to_annotations, img_to_path,
            coarse_label_per_img, train_coarse_counts,
            id_to_coarse, output_root, rng,
            run_A="A" in paths, run_B="B" in paths
        )

    if "C" in paths:
        _process_path_C(
            splits, img_to_path, img_to_annotations,
            id_to_coarse, output_root
        )

    print("\nPreprocessing complete.")
    print(f"Outputs written to: {output_root}")


# ─────────────────────────────────────────────────────────────────────────────
# 7a.  PATH A + B  (detection stage — 640×640 letterbox, YOLO labels)
# ─────────────────────────────────────────────────────────────────────────────

def _process_detect_paths(
    splits, img_to_path, img_to_annotations, img_path_map,
    coarse_label_per_img, train_coarse_counts,
    id_to_coarse, output_root, rng,
    run_A=True, run_B=True,
):
    path_tags = []
    if run_A: path_tags.append("path_A")
    if run_B: path_tags.append("path_B")

    for split_name, img_ids in splits.items():
        for path_tag in path_tags:
            img_dir = output_root / split_name / path_tag / "images"
            lbl_dir = output_root / split_name / path_tag / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{split_name}] Processing {len(img_ids)} images …")
        for iid in tqdm(img_ids, desc=f"  Letterbox {split_name}"):
            src_path = img_to_path.get(iid)
            if src_path is None or not src_path.exists():
                continue

            img = cv2.imread(str(src_path))
            if img is None:
                continue
            orig_hw = img.shape[:2]

            # collect bboxes and coarse labels for this image
            raw_bboxes  = []
            raw_classes = []
            for ann in img_to_annotations.get(iid, []):
                cls_name = id_to_coarse[ann["category_id"]]
                if cls_name not in COARSE_CLASSES:
                    continue
                raw_bboxes.append(ann["bbox"])
                raw_classes.append(COARSE_CLASSES.index(cls_name))

            if not raw_bboxes:
                continue

            # augmentation is handled at training time by Ultralytics
            # preprocessing is kept fully deterministic
            lb_img, scale, pad_lt = letterbox(img)
            adapted_bboxes = adapt_bboxes_to_letterbox(
                raw_bboxes, orig_hw, scale, pad_lt
            )

            stem = f"{iid:08d}"

            for path_tag in path_tags:
                img_dir = output_root / split_name / path_tag / "images"
                lbl_dir = output_root / split_name / path_tag / "labels"

                cv2.imwrite(
                    str(img_dir / f"{stem}.jpg"), lb_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                with open(lbl_dir / f"{stem}.txt", "w") as f:
                    for (cx, cy, nw, nh), cls_idx in zip(
                        adapted_bboxes, raw_classes
                    ):
                        f.write(f"{cls_idx} {cx} {cy} {nw} {nh}\n")

                # Path B: also write 224x224 crops for classify stage
                if path_tag == "path_B":
                    crops_dir = output_root / split_name / path_tag / "crops"
                    crops_dir.mkdir(parents=True, exist_ok=True)
                    _write_crops(lb_img, adapted_bboxes, raw_classes,
                                 crops_dir, stem)

        # copy-paste oversampling on train split only
        if split_name == "train":
            print("  Running copy-paste oversampling …")
            for path_tag in path_tags:
                img_dir = output_root / split_name / path_tag / "images"
                lbl_dir = output_root / split_name / path_tag / "labels"
                synthetics = copy_paste_oversample(
                    img_ids, img_to_annotations, img_to_path,
                    coarse_label_per_img, train_coarse_counts,
                    img_dir, lbl_dir, id_to_coarse, rng,
                )
                print(f"  [{path_tag}] Generated {len(synthetics)} synthetic images")


def _write_crops(
    lb_img: np.ndarray,
    bboxes: list,          # [(cx, cy, nw, nh)] normalized in 640x640
    class_indices: list,
    crops_dir: Path,
    stem: str,
    crop_size: int = 224,
):
    """Extract and resize instance crops to 224x224 for Path B classify stage."""
    h, w = lb_img.shape[:2]   # always 640x640 after letterbox
    for i, ((cx, cy, nw, nh), cls_idx) in enumerate(zip(bboxes, class_indices)):
        x1 = int((cx - nw / 2) * w)
        y1 = int((cy - nh / 2) * h)
        x2 = int((cx + nw / 2) * w)
        y2 = int((cy + nh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = lb_img[y1:y2, x1:x2]
        crop = cv2.resize(crop, (crop_size, crop_size),
                          interpolation=cv2.INTER_LINEAR)
        cls_dir = crops_dir / str(cls_idx)
        cls_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(cls_dir / f"{stem}_{i}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])


# ─────────────────────────────────────────────────────────────────────────────
# 7b.  PATH C  (224×224 sliding-window tiles for SSL/ViT)
# ─────────────────────────────────────────────────────────────────────────────

def _process_path_C(splits, img_to_path, img_to_annotations,
                    id_to_coarse, output_root):
    tile_meta_all = {}

    for split_name, img_ids in splits.items():
        tiles_dir = output_root / split_name / "path_C" / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{split_name}] Tiling for Path C …")

        for iid in tqdm(img_ids, desc=f"  Tile {split_name}"):
            src_path = img_to_path.get(iid)
            if src_path is None or not src_path.exists():
                continue
            img = cv2.imread(str(src_path))
            if img is None:
                continue

            for tile_idx, (tile, tx, ty) in enumerate(tile_image(img)):
                tile_stem = f"{iid:08d}_t{tile_idx:04d}"
                out_path  = tiles_dir / f"{tile_stem}.jpg"
                cv2.imwrite(str(out_path), tile,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])

                tile_meta_all[tile_stem] = {
                    "source_image_id": iid,
                    "tile_x": tx,
                    "tile_y": ty,
                    "split": split_name,
                }

    (output_root / "meta" / "tile_meta.json").write_text(
        json.dumps(tile_meta_all, indent=2)
    )
    print(f"  Total tiles written: {len(tile_meta_all)}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  YOLO DATASET YAML WRITER
#     Generates the .yaml file Ultralytics expects for training.
# ─────────────────────────────────────────────────────────────────────────────

def write_yolo_yaml(output_root: Path, path_tag: str = "path_A"):
    """Writes a dataset.yaml consumable directly by Ultralytics train CLI."""
    cfg = {
        "path":  str(output_root.resolve()),
        "train": f"train/{path_tag}/images",
        "val":   f"val/{path_tag}/images",
        "test":  f"test/{path_tag}/images",
        "nc":    len(COARSE_CLASSES),
        "names": COARSE_CLASSES,
    }
    out = output_root / f"dataset_{path_tag}.yaml"
    out.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    print(f"YOLO dataset config written: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TACO standardized preprocessing")
    p.add_argument("--taco_root",   type=Path, required=True,
                   help="Root dir of cloned TACO repo (contains data/)")
    p.add_argument("--output_root", type=Path, default=Path("data/processed"),
                   help="Where to write processed outputs")
    p.add_argument("--path", type=str, default="all",
                   choices=["A", "B", "C", "all"],
                   help="Which benchmark path(s) to generate data for")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = ["A", "B", "C"] if args.path == "all" else [args.path]
    run_preprocessing(
        taco_root=args.taco_root,
        output_root=args.output_root,
        paths=paths,
    )
    # write YOLO yaml for path A (and B if requested)
    if "A" in paths:
        write_yolo_yaml(args.output_root, "path_A")
    if "B" in paths:
        write_yolo_yaml(args.output_root, "path_B")
