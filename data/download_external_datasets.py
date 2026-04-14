
"""
download_external_datasets.py

Downloads OLM + Drinking Waste (Roboflow) and MJU-Waste (Google Drive).
MJU-Waste requires no API key and is always attempted as fallback.

Usage:
  # MJU-Waste only (no API key needed):
  python download_external_datasets.py --mju_only \
      --output ~/V-JEPA-2/mrosado/thrashscan/external_datasets

  # All datasets with Roboflow key:
  python download_external_datasets.py \
      --roboflow_key YOUR_KEY \
      --output ~/V-JEPA-2/mrosado/thrashscan/external_datasets
"""

import argparse
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

COARSE_CLASSES = ["plastic", "paper", "metal", "other"]

EXTERNAL_LABEL_TO_COARSE = {
    "plastic": "plastic", "plastic_bag": "plastic",
    "plastic_bottle": "plastic", "bottle": "plastic",
    "cup": "plastic", "straw": "plastic", "wrapper": "plastic",
    "film": "plastic", "styrofoam": "plastic", "polystyrene": "plastic",
    "lid": "plastic", "container": "plastic",
    "metal": "metal", "can": "metal", "tin": "metal",
    "aluminium": "metal", "aluminum": "metal", "foil": "metal",
    "bottle_cap": "metal", "pop_tab": "metal",
    "paper": "paper", "cardboard": "paper", "carton": "paper",
    "newspaper": "paper", "tissue": "paper",
    "glass": "other", "organic": "other", "battery": "other",
    "hazardous": "other", "rope": "other", "shoe": "other",
    "trash": "other", "litter": "other", "garbage": "other",
    "unknown": "other", "waste": "other", "cigarette": "other",
}


def map_label(label: str) -> str:
    clean = label.lower().replace(" ", "_").replace("-", "_")
    for key, coarse in EXTERNAL_LABEL_TO_COARSE.items():
        if key in clean:
            return coarse
    return "other"


def yolo_to_coco(images_dir, labels_dir, class_names,
                 output_json, dataset_name):
    from PIL import Image as PILImage
    coco = {
        "info":        {"description": dataset_name},
        "images":      [],
        "annotations": [],
        "categories":  [{"id": i, "name": n}
                        for i, n in enumerate(COARSE_CLASSES)],
    }
    ann_id = img_id = 1
    img_paths = (sorted(images_dir.glob("*.jpg")) +
                 sorted(images_dir.glob("*.jpeg")) +
                 sorted(images_dir.glob("*.png")))
    for img_path in tqdm(img_paths, desc=f"  {dataset_name}", leave=False):
        lbl_path = labels_dir / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            continue
        try:
            with PILImage.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue
        coco["images"].append({"id": img_id, "file_name": img_path.name,
                                "width": w, "height": h})
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, nw, nh = map(float, parts[1:5])
            orig = class_names[cid] if cid < len(class_names) else "unknown"
            coarse_id = COARSE_CLASSES.index(map_label(orig))
            bx=(cx-nw/2)*w; by=(cy-nh/2)*h; bw=nw*w; bh=nh*h
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id,
                "category_id": coarse_id,
                "bbox": [round(bx,2), round(by,2), round(bw,2), round(bh,2)],
                "area": round(bw*bh,2), "iscrowd": 0,
            })
            ann_id += 1
        img_id += 1
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f)
    print(f"  {dataset_name}: {img_id-1} images, {ann_id-1} annotations")
    return output_json


def convert_roboflow_dataset(dataset_dir, dataset_name, output_root):
    if dataset_dir is None or not Path(dataset_dir).exists():
        return None
    import yaml
    dataset_dir = Path(dataset_dir)
    coco_dir = output_root / "coco_format" / dataset_name
    coco_dir.mkdir(parents=True, exist_ok=True)
    class_names = []
    yaml_paths = list(dataset_dir.rglob("data.yaml"))
    if yaml_paths:
        with open(yaml_paths[0]) as f:
            cfg = yaml.safe_load(f)
        class_names = cfg.get("names", [])
        print(f"  Classes: {class_names}")
    for split in ["train", "valid", "test"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists():
            continue
        out_json = coco_dir / split / f"{split}_annotations.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        yolo_to_coco(img_dir, lbl_dir, class_names, out_json, dataset_name)
        img_out = coco_dir / split / "images"
        img_out.mkdir(parents=True, exist_ok=True)
        for img in img_dir.glob("*.*"):
            shutil.copy2(img, img_out / img.name)
    return coco_dir


def download_open_litter_map(output_root, api_key=""):
    print("\n[1/3] Open Litter Map...")
    dest_dir = Path(output_root) / "open_litter_map"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not api_key:
        print("  No API key — skipping OLM")
        print("  Get free key at https://app.roboflow.com")
        print(f"  Manual: https://universe.roboflow.com/open-litter-map/open-litter-map")
        return None
    try:
        os.system("pip install roboflow -q")
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        dataset = rf.workspace("open-litter-map").project(
            "open-litter-map").version(1).download(
            "yolov8", location=str(dest_dir))
        print(f"  Downloaded: {dest_dir}")
        return dest_dir
    except Exception as e:
        print(f"  [warn] {e}")
        return None


def download_drinking_waste(output_root, api_key=""):
    print("\n[2/3] Drinking Waste...")
    dest_dir = Path(output_root) / "drinking_waste"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not api_key:
        print("  No API key — skipping Drinking Waste")
        print(f"  Manual: https://universe.roboflow.com/alex-hyams-cosqx/drinking-waste-classification")
        return None
    try:
        os.system("pip install roboflow -q")
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        dataset = rf.workspace("alex-hyams-cosqx").project(
            "drinking-waste-classification").version(1).download(
            "yolov8", location=str(dest_dir))
        print(f"  Downloaded: {dest_dir}")
        return dest_dir
    except Exception as e:
        print(f"  [warn] {e}")
        return None


def download_mju_waste(output_root):
    """
    MJU-Waste: 2475 RGBD images, single class waste, no API key needed.
    Downloads from Google Drive via gdown.
    """
    print("\n[3/3] MJU-Waste (no API key needed)...")
    dest_dir = Path(output_root) / "mju_waste"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if (dest_dir / "annotations.json").exists():
        print(f"  Already downloaded: {dest_dir}")
        return dest_dir
    print("  Installing gdown...")
    os.system("pip install gdown -q")
    gdrive_id = "1o101UBJGeeMPpI-DSY6oh-tLk9AHXMny"
    zip_path  = dest_dir / "mju_waste.zip"
    try:
        import gdown
        print("  Downloading from Google Drive (~1.3GB)...")
        gdown.download(
            f"https://drive.google.com/uc?id={gdrive_id}",
            str(zip_path), quiet=False)
        if not zip_path.exists() or zip_path.stat().st_size < 1e6:
            raise ValueError("Download incomplete")
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest_dir)
        zip_path.unlink()
        print(f"  Extracted to: {dest_dir}")
        return dest_dir
    except Exception as e:
        print(f"  [warn] gdown failed: {e}")
        print("  Manual download:")
        print("  https://drive.google.com/file/d/1o101UBJGeeMPpI-DSY6oh-tLk9AHXMny")
        print(f"  Extract to: {dest_dir}")
        return None


def process_mju_waste(mju_dir, output_root):
    """Process MJU-Waste after download. All annotations → 'other'."""
    if mju_dir is None or not Path(mju_dir).exists():
        return None
    mju_dir   = Path(mju_dir)
    output_root = Path(output_root)
    print("\n  Processing MJU-Waste...")
    out_dir = output_root / "coco_format" / "mju_waste"
    out_dir.mkdir(parents=True, exist_ok=True)

    coco_jsons = [j for j in mju_dir.rglob("*.json")
                  if any(x in j.name.lower()
                         for x in ["annotation", "instance"]) and
                  j.stat().st_size > 1000]

    if coco_jsons:
        for jp in coco_jsons:
            img_base = jp.parent / "images"
            if not img_base.exists():
                img_base = jp.parent
            split_name = jp.stem.replace("annotations_", "").replace(
                         "instances_", "")
            split_out = out_dir / split_name
            split_out.mkdir(parents=True, exist_ok=True)
            _coco_remap_to_other(jp, img_base, split_out,
                                  f"mju_{split_name}")
    else:
        xml_files = list(mju_dir.rglob("*.xml"))
        if xml_files:
            print(f"  Found {len(xml_files)} VOC XML files — converting...")
            _voc_to_coco_mju(mju_dir, out_dir)
        else:
            print(f"  [warn] No annotation files found in {mju_dir}")
            return None

    print(f"  MJU-Waste processed → {out_dir}")
    return out_dir


def _coco_remap_to_other(src_json, images_src_dir, output_dir, name):
    """Remap all categories to 'other' (index 3) for single-class datasets."""
    with open(src_json) as f:
        data = json.load(f)
    coco = {
        "info":        {"description": name},
        "images":      data.get("images", []),
        "annotations": [],
        "categories":  [{"id": i, "name": n}
                        for i, n in enumerate(COARSE_CLASSES)],
    }
    for ann in data.get("annotations", []):
        new = ann.copy()
        new["category_id"] = 3
        coco["annotations"].append(new)
    out_json = output_dir / f"{name}_annotations.json"
    with open(out_json, "w") as f:
        json.dump(coco, f)
    img_out = output_dir / "images"
    img_out.mkdir(exist_ok=True)
    for img_meta in tqdm(data.get("images", []),
                         desc=f"  Copying {name}", leave=False):
        src = Path(images_src_dir) / img_meta["file_name"]
        if src.exists():
            shutil.copy2(src, img_out / Path(img_meta["file_name"]).name)
    print(f"  {name}: {len(coco['images'])} images → all 'other'")


def _voc_to_coco_mju(voc_root, output_dir):
    """Convert PASCAL VOC XML to COCO for MJU-Waste. All → 'other'."""
    import xml.etree.ElementTree as ET
    from PIL import Image as PILImage

    output_dir.mkdir(parents=True, exist_ok=True)
    img_out = output_dir / "images"
    img_out.mkdir(exist_ok=True)

    coco = {
        "info": {"description": "mju_waste"},
        "images": [], "annotations": [],
        "categories": [{"id": i, "name": n}
                       for i, n in enumerate(COARSE_CLASSES)],
    }
    xml_files = sorted(voc_root.rglob("*.xml"))
    img_id = ann_id = 1

    for xml_path in tqdm(xml_files, desc="  VOC→COCO"):
        if any(x in xml_path.stem.lower()
               for x in ["_d", "depth", "_dep"]):
            continue
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            continue
        filename = root.findtext("filename", "")
        img_path = xml_path.parent / filename
        if not img_path.exists():
            for ext in [".jpg",".jpeg",".png"]:
                c = xml_path.with_suffix(ext)
                if c.exists():
                    img_path = c
                    break
        if not img_path.exists():
            continue
        if any(x in img_path.stem.lower()
               for x in ["_d","depth","_dep"]):
            continue
        try:
            with PILImage.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue
        new_fname = f"mju_{img_id:06d}{img_path.suffix}"
        shutil.copy2(img_path, img_out / new_fname)
        coco["images"].append({"id": img_id, "file_name": new_fname,
                                "width": w, "height": h})
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin=float(bb.findtext("xmin",0)); ymin=float(bb.findtext("ymin",0))
            xmax=float(bb.findtext("xmax",0)); ymax=float(bb.findtext("ymax",0))
            bw=xmax-xmin; bh=ymax-ymin
            if bw<=0 or bh<=0:
                continue
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id,
                "category_id": 3,
                "bbox": [round(xmin,2),round(ymin,2),round(bw,2),round(bh,2)],
                "area": round(bw*bh,2), "iscrowd": 0, "segmentation": [],
            })
            ann_id += 1
        img_id += 1

    out_json = output_dir / "mju_waste_annotations.json"
    with open(out_json, "w") as f:
        json.dump(coco, f)
    print(f"  VOC done: {img_id-1} images, {ann_id-1} annotations")


def print_download_summary(output_root):
    """download summary of all collected datasets."""
    output_root = Path(output_root)
    coco_root   = output_root / "coco_format"
    if not coco_root.exists():
        print("No coco_format dir found yet.")
        return
    print("\n" + "="*50)
    print("Download summary")
    print("="*50)
    total = 0
    for ds_dir in sorted(coco_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        n = 0
        cls_counts = Counter()
        for j in ds_dir.rglob("*.json"):
            try:
                with open(j) as f:
                    data = json.load(f)
                n += len(data.get("images", []))
                for ann in data.get("annotations", []):
                    cid = ann.get("category_id", 3)
                    cls_counts[COARSE_CLASSES[cid]
                               if cid < 4 else "unknown"] += 1
            except Exception:
                pass
        print(f"  {ds_dir.name:<25} {n} images  "
              + "  ".join(f"{k}={v}" for k,v in sorted(cls_counts.items())))
        total += n
    print(f"  {'TOTAL':<25} {total} images")
    print("="*50)
    print("\nNext:")
    print(f"  python merge_datasets.py \\")
    print(f"      --taco_root     ~/V-JEPA-2/mrosado/thrashscan/TACO \\")
    print(f"      --external_root {output_root} \\")
    print(f"      --output_root   ~/V-JEPA-2/mrosado/thrashscan/processed_4cls")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output",        type=Path,
                   default=Path("external_datasets"))
    p.add_argument("--roboflow_key",  type=str,  default="",
                   help="Free key from app.roboflow.com")
    p.add_argument("--skip_olm",      action="store_true")
    p.add_argument("--skip_drinking", action="store_true")
    p.add_argument("--skip_mju",      action="store_true")
    p.add_argument("--mju_only",      action="store_true",
                   help="Download MJU-Waste only (no API key needed)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mju_only:
        mju_dir = download_mju_waste(args.output)
        process_mju_waste(mju_dir, args.output)
        print_download_summary(args.output)
        raise SystemExit(0)

    olm_dir = None if args.skip_olm else \
              download_open_litter_map(args.output, args.roboflow_key)
    dw_dir  = None if args.skip_drinking else \
              download_drinking_waste(args.output, args.roboflow_key)
    mju_dir = None if args.skip_mju else \
              download_mju_waste(args.output)

    if olm_dir:
        convert_roboflow_dataset(Path(olm_dir), "open_litter_map", args.output)
    if dw_dir:
        convert_roboflow_dataset(Path(dw_dir), "drinking_waste", args.output)
    if mju_dir:
        process_mju_waste(mju_dir, args.output)

    print_download_summary(args.output)
