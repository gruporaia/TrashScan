"""
download_vjepa2.py

Downloads V-JEPA 2 / V-JEPA 2.1 pretrained weights from HuggingFace.
NO video dataset download needed — Meta provides pretrained checkpoints.

V-JEPA 2.1 (released 2026-03-16) is the latest and recommended version.
It uses Dense Predictive Loss + Deep Self-Supervision for better features.

Available models:
  vitl16    V-JEPA 2   ViT-L/16  307M  ~1.2GB  recommended baseline
  vitl16_21 V-JEPA 2.1 ViT-L/16  307M  ~1.2GB  recommended (latest)
  vitg16    V-JEPA 2   ViT-g/16  1.1B  ~4.5GB  largest, best perf
  vitg16_21 V-JEPA 2.1 ViT-g/16  1.1B  ~4.5GB  largest + latest

Usage:
  python download_vjepa2.py --model vitl16_21 \
      --output ~/V-JEPA-2/mrosado/thrashscan/runs/path_C/vjepa2_weights
"""

import argparse
import json
from pathlib import Path

VJEPA2_MODELS = {
    "vitl16": {
        "hf_repo":  "facebook/vjepa2-vitl-fpc64-256",
        "version":  "V-JEPA 2",
        "params_M": 307,
        "vram_GB":  6,
        "notes":    "Original V-JEPA 2 ViT-L/16",
    },
    "vitl16_21": {
        "hf_repo":  "facebook/vjepa2.1-vitl-256",
        "version":  "V-JEPA 2.1",
        "params_M": 307,
        "vram_GB":  6,
        "notes":    "RECOMMENDED — latest V-JEPA 2.1 ViT-L/16, "
                    "better dense features, released 2026-03-16",
    },
    "vitg16": {
        "hf_repo":  "facebook/vjepa2-vitg-fpc64-256",
        "version":  "V-JEPA 2",
        "params_M": 1100,
        "vram_GB":  18,
        "notes":    "Largest V-JEPA 2 — may OOM at batch>4 on A10G",
    },
    "vitg16_21": {
        "hf_repo":  "facebook/vjepa2.1-vitg-256",
        "version":  "V-JEPA 2.1",
        "params_M": 1100,
        "vram_GB":  18,
        "notes":    "Largest + latest — requires batch=2 on A10G",
    },
}


def download_vjepa2(model_key: str, output_dir: Path):
    info = VJEPA2_MODELS[model_key]
    local_dir = output_dir / model_key
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {info['version']} — {model_key}")
    print(f"  Repo    : {info['hf_repo']}")
    print(f"  Params  : {info['params_M']}M")
    print(f"  VRAM    : ~{info['vram_GB']}GB")
    print(f"  Notes   : {info['notes']}")
    print(f"  Output  : {local_dir}")

    try:
        import os
        os.system("pip install huggingface_hub -q")
        from huggingface_hub import snapshot_download

        print("\n  Downloading from HuggingFace...")
        snapshot_download(
            repo_id         = info["hf_repo"],
            local_dir       = str(local_dir),
            ignore_patterns = ["*.msgpack", "flax_model*",
                               "tf_model*", "*.ot"],
        )
        print(f"\n  Download complete: {local_dir}")

        # find weights file
        weights_file = None
        for pattern in ["*.bin", "*.safetensors", "*.pth", "*.pt"]:
            candidates = list(local_dir.glob(pattern))
            if candidates:
                weights_file = max(candidates,
                                   key=lambda p: p.stat().st_size)
                break

        if weights_file:
            size_gb = weights_file.stat().st_size / 1e9
            print(f"  Weights : {weights_file.name} ({size_gb:.2f}GB)")
        else:
            print("  [warn] No weight file found — "
                  "check HuggingFace repo structure")

        # save manifest
        manifest = {
            "model_key":    model_key,
            "version":      info["version"],
            "hf_repo":      info["hf_repo"],
            "weights_path": str(weights_file) if weights_file else "",
            "params_M":     info["params_M"],
        }
        with open(local_dir / "vjepa2_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n  Manifest: {local_dir}/vjepa2_manifest.json")
        print(f"\n  Use in train_path_C.py:")
        print(f"    python train_path_C.py --variant C3 \\")
        print(f"        --vjepa2_weights {local_dir} \\")
        print(f"        --data_yaml "
              f"~/processed_4cls/dataset_path_A.yaml \\")
        print(f"        --output ~/runs/path_C \\")
        print(f"        --label_budget_sweep --device 0")

        return local_dir

    except Exception as e:
        print(f"\n  [error] {e}")
        print(f"\n  Manual download:")
        print(f"    pip install huggingface_hub")
        print(f"    python -c \"from huggingface_hub import "
              f"snapshot_download; "
              f"snapshot_download('{info['hf_repo']}', "
              f"local_dir='{local_dir}')\"")
        return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  type=str, default="vitl16_21",
                   choices=list(VJEPA2_MODELS.keys()))
    p.add_argument("--output", type=Path,
                   default=Path(
                       "runs/path_C/vjepa2_weights"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = download_vjepa2(args.model, args.output)
    if result:
        print(f"\nReady. Next step:")
        print(f"  python train_path_C.py --variant C3 \\")
        print(f"      --vjepa2_weights {result} \\")
        print(f"      --data_yaml "
              f"~/V-JEPA-2/mrosado/thrashscan/"
              f"processed_4cls/dataset_path_A.yaml \\")
        print(f"      --output "
              f"~/V-JEPA-2/mrosado/thrashscan/runs/path_C \\")
        print(f"      --label_budget_sweep --device 0")
