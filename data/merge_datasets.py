
"""
merge_datasets.py  —  Merges TACO + external datasets into single
COCO JSON + image folder, then runs preprocess.py for all 3 paths.

Usage:
  python merge_datasets.py \
      --taco_root      ~/V-JEPA-2/mrosado/thrashscan/TACO \
      --external_root  ~/V-JEPA-2/mrosado/thrashscan/external_datasets \
      --output_root    ~/V-JEPA-2/mrosado/thrashscan/processed_4cls
"""

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from tqdm import tqdm

COARSE_CLASSES = ["plastic", "paper", "metal", "other"]


def remap_cat(cid: int, cats: dict) -> int:
    """category remapping from source cat_id to 4-class index."""
    name = cats.get(cid, "other").lower()
    for cls in COARSE_CLASSES:
        if cls in name:
            return COARSE_CLASSES.index(cls)
    return 3


def merge_coco_jsons(
    taco_json:      Path,
    external_root:  Path,
    output_json:    Path,
    output_img_dir: Path,
):
    output_img_dir.mkdir(parents=True, exist_ok=True)

    merged = {
        "info":        {"description": "TACO+OLM+DrinkingWaste+MJUWaste 4-class"},
        "images":      [],
        "annotations": [],
        "categories":  [{"id": i, "name": n, "supercategory": n}
                        for i, n in enumerate(COARSE_CLASSES)],
    }

    global_img_id = 1
    global_ann_id = 1

    def load_and_merge(src_json: Path, src_name: str, img_base: Path):
        nonlocal global_img_id, global_ann_id

        if not src_json.exists():
            print(f"  [skip] {src_name}: not found at {src_json}")
            return 0, 0

        with open(src_json) as f:
            data = json.load(f)

        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        old_to_new = {}
        n_imgs = 0
        n_anns = 0

        for img in tqdm(data.get("images", []),
                        desc=f"  {src_name}", leave=False):
            fname    = img["file_name"]
            src_path = img_base / fname
            if not src_path.exists():
                src_path = img_base / Path(fname).name
            if not src_path.exists():
                continue

            new_id    = global_img_id
            new_fname = f"{src_name}_{new_id:08d}{src_path.suffix}"
            try:
                shutil.copy2(src_path, output_img_dir / new_fname)
            except Exception:
                continue

            old_to_new[img["id"]] = new_id
            merged["images"].append({
                "id": new_id, "file_name": new_fname,
                "width": img.get("width", 640),
                "height": img.get("height", 480),
            })
            global_img_id += 1
            n_imgs += 1

        for ann in data.get("annotations", []):
            old_id = ann.get("image_id")
            if old_id not in old_to_new:
                continue
            bbox = ann.get("bbox", [0,0,1,1])
            if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
                continue
            merged["annotations"].append({
                "id":           global_ann_id,
                "image_id":     old_to_new[old_id],
                "category_id":  remap_cat(ann["category_id"], cats),
                "bbox":         bbox,
                "area":         ann.get("area", bbox[2]*bbox[3]),
                "iscrowd":      ann.get("iscrowd", 0),
                "segmentation": ann.get("segmentation", []),
            })
            global_ann_id += 1
            n_anns += 1

        return n_imgs, n_anns

    # 1. TACO
    n_i, n_a = load_and_merge(taco_json, "TACO", taco_json.parent)
    print(f"  TACO           : {n_i} images, {n_a} annotations")

    # 2. External — scan coco_format subfolders
    coco_format_dir = external_root / "coco_format"
    if coco_format_dir.exists():
        for ds_dir in sorted(coco_format_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            json_candidates = (
                list(ds_dir.rglob("*annotations*.json")) +
                list(ds_dir.rglob("*.json"))
            )
            json_candidates = [j for j in json_candidates
                               if j.stat().st_size > 1000]
            seen = set()
            for jp in json_candidates:
                if jp in seen:
                    continue
                seen.add(jp)
                img_base = jp.parent / "images"
                if not img_base.exists():
                    img_base = jp.parent
                n_i, n_a = load_and_merge(
                    jp, f"{ds_dir.name}_{jp.stem}", img_base)
                if n_i > 0:
                    print(f"  {ds_dir.name}/{jp.stem}: "
                          f"{n_i} imgs, {n_a} anns")
    else:
        print(f"  [warn] No coco_format dir at {coco_format_dir} — TACO only")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(merged, f)

    cls_counts = Counter(
        COARSE_CLASSES[a["category_id"]] for a in merged["annotations"]
    )
    print(f"\nMerged → {output_json}")
    print(f"  Total images     : {len(merged['images'])}")
    print(f"  Total annotations: {len(merged['annotations'])}")
    for cls in COARSE_CLASSES:
        print(f"  {cls:<12}: {cls_counts.get(cls, 0)}")

    return output_json


def write_merged_taco_structure(
    merged_root: Path,
    output_json: Path,
    output_img_dir: Path,
):
    """
    Creates TACO-compatible structure so preprocess.py works unchanged:
      merged_root/data/annotations.json
      merged_root/data/images  -> output_img_dir (symlink)
    """
    data_dir = merged_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    taco_ann = data_dir / "annotations.json"
    if not taco_ann.exists():
        shutil.copy2(output_json, taco_ann)
        print(f"  Copied annotations → {taco_ann}")

    taco_img = data_dir / "images"
    if not taco_img.exists():
        taco_img.symlink_to(output_img_dir.resolve())
        print(f"  Symlinked images → {taco_img}")

    print(f"TACO-compatible structure ready at: {merged_root}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--taco_root",       type=Path, required=True)
    p.add_argument("--external_root",   type=Path, required=True)
    p.add_argument("--output_root",     type=Path, required=True)
    p.add_argument("--skip_preprocess", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    taco_json      = args.taco_root / "data" / "annotations.json"
    merged_root    = args.output_root / "merged_data"
    output_json    = merged_root / "annotations.json"
    output_img_dir = merged_root / "images"

    assert taco_json.exists(), f"TACO annotations not found: {taco_json}"

    merge_coco_jsons(
        taco_json      = taco_json,
        external_root  = args.external_root,
        output_json    = output_json,
        output_img_dir = output_img_dir,
    )

    write_merged_taco_structure(merged_root, output_json, output_img_dir)

    if not args.skip_preprocess:
        print("\nRunning preprocess.py on merged dataset...")
        result = subprocess.run([
            "python", "preprocess.py",
            "--taco_root",   str(merged_root),
            "--output_root", str(args.output_root),
            "--path",        "all",
        ], check=True)
        if result.returncode == 0:
            print(f"\nDone. Processed data at: {args.output_root}")
            print(f"\nRetrain with:")
            print(f"  python train_path_A.py \\")
            print(f"      --data     {args.output_root}/dataset_path_A.yaml \\")
            print(f"      --output   ~/runs/path_A_4cls \\")
            print(f"      --models   yolov8m yolov9s \\")
            print(f"      --epochs   300 --batch 8 --patience 30 --device 0")
    else:
        print(f"\nRun manually:")
        print(f"  python preprocess.py \\")
        print(f"      --taco_root   {merged_root} \\")
        print(f"      --output_root {args.output_root} \\")
        print(f"      --path all")
