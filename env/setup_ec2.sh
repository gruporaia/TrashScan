#!/usr/bin/env bash
# setup_ec2.sh  —  Full EC2 bootstrap for TrashScan litter benchmark
#
# Tested on: Ubuntu 22.04 Deep Learning AMI, g5.2xlarge (NVIDIA A10G, CUDA 12.4)
#
# Run once after SSH into a fresh instance:
#   chmod +x setup_ec2.sh && ./setup_ec2.sh
#
# After completion:
#   conda activate litter_AB   →  Paths A + B
#   conda activate litter_C    →  Path C (SSL pretraining + V-JEPA 2)

set -e   # exit on first error
set -u   # treat unset variables as errors

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$HOME/thrashscan"
DATA_ROOT="$HOME/data"
TACO_ROOT="$DATA_ROOT/TACO"
PROCESSED_ROOT="$PROJECT_ROOT/processed_4cls"
RUNS_ROOT="$PROJECT_ROOT/runs"
VJEPA2_DIR="$RUNS_ROOT/path_C/vjepa2_weights"

echo "========================================================"
echo "  TrashScan — EC2 Bootstrap"
echo "  Project : $PROJECT_ROOT"
echo "  Data    : $DATA_ROOT"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "========================================================"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  VERIFY CUDA
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[1/8] Checking CUDA..."
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader
nvcc --version | grep "release"

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CONDA
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[2/8] Setting up Conda..."
if ! command -v conda &>/dev/null; then
    echo "  Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    source ~/.bashrc
else
    echo "  Conda found: $(conda --version)"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLONE / PULL PROJECT REPO
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[3/8] Project repository..."
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "  Cloning thrashscan repo..."
    # Replace with your actual GitHub URL:
    git clone https://github.com/YOUR_USERNAME/thrashscan.git "$PROJECT_ROOT"
else
    echo "  Repo exists — pulling latest..."
    git -C "$PROJECT_ROOT" pull
fi
cd "$PROJECT_ROOT"

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CREATE CONDA ENVIRONMENTS
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[4/8] Creating Conda environments..."

# litter_AB (Paths A and B)
if conda env list | grep -q "^litter_AB "; then
    echo "  litter_AB exists — updating..."
    conda env update -n litter_AB -f environment_AB.yml --prune
else
    echo "  Creating litter_AB..."
    conda env create -f environment_AB.yml
fi

# litter_C (Path C)
if conda env list | grep -q "^litter_C "; then
    echo "  litter_C exists — updating..."
    conda env update -n litter_C -f environment_C.yml --prune
else
    echo "  Creating litter_C..."
    conda env create -f environment_C.yml
fi

# verify GPU visible to PyTorch in litter_AB
echo ""
echo "  Verifying PyTorch CUDA in litter_AB..."
conda run -n litter_AB python - << 'PYCHECK'
import torch
assert torch.cuda.is_available(), "CUDA not available!"
dev = torch.cuda.get_device_properties(0)
print(f"  GPU  : {dev.name}")
print(f"  VRAM : {dev.total_memory/1e9:.1f} GB")
print(f"  CUDA : {torch.version.cuda}")
x = torch.randn(2, 3, 640, 640, device="cuda")
print(f"  Tensor: {x.shape}  dtype={x.dtype}  ✓")
PYCHECK

# ─────────────────────────────────────────────────────────────────────────────
# 5.  DOWNLOAD TACO
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[5/8] TACO dataset..."
mkdir -p "$DATA_ROOT"

if [ ! -d "$TACO_ROOT/.git" ]; then
    git clone https://github.com/pedropro/TACO.git "$TACO_ROOT"
else
    echo "  TACO already cloned — pulling..."
    git -C "$TACO_ROOT" pull
fi

if [ ! -f "$TACO_ROOT/data/annotations.json" ]; then
    echo "  Downloading TACO images..."
    cd "$TACO_ROOT"
    conda run -n litter_AB pip install requests -q
    conda run -n litter_AB python download.py
    cd "$PROJECT_ROOT"
else
    echo "  TACO images already present"
fi

IMG_COUNT=$(find "$TACO_ROOT/data" -name "*.jpg" | wc -l)
echo "  Images found: $IMG_COUNT"

# ─────────────────────────────────────────────────────────────────────────────
# 6.  DOWNLOAD EXTERNAL DATASETS (MJU-Waste, no API key needed)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[6/8] External datasets..."
EXT_ROOT="$PROJECT_ROOT/external_datasets"
mkdir -p "$EXT_ROOT"

conda run -n litter_AB python data/download_external_datasets.py \
    --mju_only \
    --output "$EXT_ROOT"

# (Optional) add Roboflow datasets if API key available:
# conda run -n litter_AB python data/download_external_datasets.py \
#     --roboflow_key YOUR_KEY_HERE \
#     --output "$EXT_ROOT"

# ─────────────────────────────────────────────────────────────────────────────
# 7.  MERGE + PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[7/8] Merging datasets and preprocessing..."
mkdir -p "$PROCESSED_ROOT"

# merge TACO + external into unified 4-class COCO JSON
conda run -n litter_AB python data/merge_datasets.py \
    --taco_root     "$TACO_ROOT" \
    --external_root "$EXT_ROOT" \
    --output_root   "$PROCESSED_ROOT"

# preprocess.py is called automatically by merge_datasets.py
# but can also be run standalone:
# conda run -n litter_AB python preprocess.py \
#     --taco_root   "$PROCESSED_ROOT/merged_data" \
#     --output_root "$PROCESSED_ROOT" \
#     --path all

echo ""
echo "  Processed output:"
find "$PROCESSED_ROOT" -maxdepth 3 -type d | sort | head -30

# ─────────────────────────────────────────────────────────────────────────────
# 8.  DOWNLOAD V-JEPA 2 WEIGHTS  (Path C / C3 variant)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[8/8] V-JEPA 2 weights (C3 variant)..."
mkdir -p "$VJEPA2_DIR"

conda run -n litter_C python utils/download_vjepa2.py \
    --model  vitl16 \
    --output "$VJEPA2_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Setup complete!"
echo ""
echo "  Path A (YOLO baseline):"
echo "    conda activate litter_AB"
echo "    python paths/train_path_A.py \\"
echo "        --data    $PROCESSED_ROOT/dataset_path_A.yaml \\"
echo "        --output  $RUNS_ROOT/path_A \\"
echo "        --models  yolov8m yolov9s \\"
echo "        --epochs  300 --batch 8 --patience 30 --device 0"
echo ""
echo "  Path B (detect + classify):"
echo "    conda activate litter_AB"
echo "    python paths/train_path_B.py \\"
echo "        --detector_weights $RUNS_ROOT/path_A/yolov8m/weights/best.pt \\"
echo "        --crops_dir        $PROCESSED_ROOT \\"
echo "        --output           $RUNS_ROOT/path_B \\"
echo "        --use_yolo_crops --det_conf 0.25 \\"
echo "        --epochs 50 --batch 32 --device 0"
echo ""
echo "  Path C — MIM-JEPA pretraining (C2 / C4 variants):"
echo "    conda activate litter_C"
echo "    python paths/pretrain_cnn_jepa.py \\"
echo "        --data_root $PROCESSED_ROOT --output $RUNS_ROOT/path_C/pretrain_cnn \\"
echo "        --epochs 100 --patience 7 --batch 16 --device 0"
echo ""
echo "  Path C — C3 (V-JEPA 2, frozen backbone):"
echo "    conda activate litter_C"
echo "    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\"
echo "    python paths/train_C2_C3_ultralytics.py \\"
echo "        --variant C3 \\"
echo "        --data_yaml       $PROCESSED_ROOT/dataset_path_A.yaml \\"
echo "        --vjepa2_weights  $VJEPA2_DIR/vitl16 \\"
echo "        --output          $RUNS_ROOT/path_C \\"
echo "        --epochs 100 --patience 25 --batch 4 --device 0 \\"
echo "        --label_budget_sweep"
echo ""
echo "  Evaluate all paths:"
echo "    conda activate litter_AB"
echo "    python eval/evaluate.py --path A \\"
echo "        --runs_dir $RUNS_ROOT/path_A \\"
echo "        --data_yaml $PROCESSED_ROOT/dataset_path_A.yaml \\"
echo "        --output ~/results --device 0"
echo "    python eval/evaluate.py --summarize --output ~/results"
echo ""
echo "  MLflow UI:"
echo "    mlflow ui --host 0.0.0.0 --port 5000"
echo "    (SSH tunnel: ssh -L 5000:localhost:5000 <instance>)"
echo "========================================================"
