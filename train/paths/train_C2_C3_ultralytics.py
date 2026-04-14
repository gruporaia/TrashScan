"""
train_C2_C3_ultralytics.py  —  Path C SSL-pretrained detection via YOLO hook injection.

Injects SSL-pretrained backbone features into YOLOv8m at layers 4, 6, 9
(P3/P4/P5 positions) using picklable forward hooks.  The YOLO neck and
detection head are trained normally; only the backbone is swapped.

Paper naming:
  C1  ViT-B/16 + MIM-JEPA domain-specific pretraining      (disk: C2_*)
  C2  ViT-L/16 + V-JEPA 2 frozen general video pretraining (disk: C3_*)
  C3  ResNet-50 + MIM-JEPA domain-specific pretraining      (disk: C4_*)

Script variants match disk naming:
  --variant C2  →  ViT-B/16 + MIM-JEPA (paper C1)
  --variant C3  →  ViT-L/16 + V-JEPA 2 (paper C2)  — frozen backbone
  --variant C4  →  ResNet-50 + MIM-JEPA (paper C3)

Usage:

  # C3 (V-JEPA 2, frozen backbone) — requires ~12GB VRAM, batch=4:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  python train_C2_C3_ultralytics.py \\
      --variant         C3 \\
      --data_yaml       ~/processed_4cls/dataset_path_A.yaml \\
      --vjepa2_weights  ~/runs/path_C/vjepa2_weights/vitl \\
      --output          ~/runs/path_C \\
      --epochs          100 --patience 25 --batch 4 --device 0 \\
      --label_budget_sweep

  # C2 (ViT-B + MIM-JEPA):
  python train_C2_C3_ultralytics.py \\
      --variant      C2 \\
      --data_yaml    ~/processed_4cls/dataset_path_A.yaml \\
      --ssl_weights  ~/runs/path_C/pretrain/vit_jepa.pt \\
      --output       ~/runs/path_C \\
      --epochs       100 --patience 25 --batch 8 --device 0 \\
      --label_budget_sweep

  # C4 (ResNet-50 + MIM-JEPA, benchmark leader):
  python train_C2_C3_ultralytics.py \\
      --variant      C4 \\
      --data_yaml    ~/processed_4cls/dataset_path_A.yaml \\
      --ssl_weights  ~/runs/path_C/pretrain_cnn/resnet_jepa.pt \\
      --output       ~/runs/path_C \\
      --epochs       100 --patience 25 --batch 8 --device 0 \\
      --label_budget_sweep
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

COARSE_CLASSES = ["plastic", "paper", "metal", "other"]
LABEL_BUDGETS  = [0.10, 0.25, 0.50, 0.75, 1.00]
SEED           = 42

# YOLOv8m backbone hook layer indices (P3 / P4 / P5)
HOOK_LAYERS    = [4, 6, 9]


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE — passed to all hook instances so they share ViT tokens
# ─────────────────────────────────────────────────────────────────────────────

class _ViTState:
    """Holds the current-batch ViT token tensor produced by the pre-hook."""
    def __init__(self):
        self.tokens: torch.Tensor | None = None


# ─────────────────────────────────────────────────────────────────────────────
# PRE-HOOK — registered on backbone layer 0
# Runs the SSL encoder on the input batch, stores tokens in shared state.
# Must be a named class (not lambda) to be picklable by DataLoader workers.
# ─────────────────────────────────────────────────────────────────────────────

class _ViTPreHook:
    """
    Pre-hook on YOLO backbone layer 0.
    Runs the SSL encoder and stores tokens in shared _ViTState.
    Does NOT modify the layer's input (returns None).
    """
    def __init__(self, extract_fn, state: _ViTState):
        self.extract_fn = extract_fn   # callable: x → tokens [B,N,D]
        self.state      = state

    def __call__(self, module, args):
        x = args[0]                    # [B, 3, H, W]
        self.state.tokens = self.extract_fn(x)
        # return None → do not modify args


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD HOOKS — registered on backbone layers 4, 6, 9
# Replace each layer's output with projected + spatially adapted ViT features.
# ─────────────────────────────────────────────────────────────────────────────

class _HookP3:
    def __init__(self, state: _ViTState, proj: nn.Module, h: int, w: int):
        self.state = state; self.proj = proj; self.h = h; self.w = w

    def __call__(self, module, args, output):
        return _replace_with_vit(self.state.tokens, self.proj,
                                  self.h, self.w, output)


class _HookP4:
    def __init__(self, state: _ViTState, proj: nn.Module, h: int, w: int):
        self.state = state; self.proj = proj; self.h = h; self.w = w

    def __call__(self, module, args, output):
        return _replace_with_vit(self.state.tokens, self.proj,
                                  self.h, self.w, output)


class _HookP5:
    def __init__(self, state: _ViTState, proj: nn.Module, h: int, w: int):
        self.state = state; self.proj = proj; self.h = h; self.w = w

    def __call__(self, module, args, output):
        return _replace_with_vit(self.state.tokens, self.proj,
                                  self.h, self.w, output)


def _replace_with_vit(tokens: torch.Tensor, proj: nn.Module,
                       out_h: int, out_w: int,
                       reference_output: torch.Tensor) -> torch.Tensor:
    """
    Core hook logic:
      1. Receives ViT tokens [B, N, D] from shared state
      2. Strips CLS token if present (197 → 196 for ViT-B/L)
      3. Reshapes to [B, D, 14, 14] spatial map
      4. Bilinear-interpolates to target spatial size (out_h × out_w)
      5. Projects channels with 1×1 conv to match YOLO's expected channel count
      6. Converts to same dtype as original output
    """
    if tokens is None:
        return reference_output  # fallback: don't replace

    B       = tokens.shape[0]
    n_tok   = tokens.shape[1]

    # strip CLS token  (ViT-B: 197→196, ViT-L: 197→196)
    if n_tok in (197,):
        tokens = tokens[:, 1:, :]
        n_tok  = tokens.shape[1]

    # reshape to spatial map
    h_tok = w_tok = int(n_tok ** 0.5)
    feat  = tokens.permute(0, 2, 1).reshape(B, -1, h_tok, w_tok)  # [B,D,14,14]

    # interpolate to target spatial size
    feat = F.interpolate(feat.float(),
                         size=(out_h, out_w),
                         mode="bilinear",
                         align_corners=False)

    # project channels
    feat = proj(feat)

    # match output dtype (AMP safety)
    return feat.to(dtype=reference_output.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# PROBE BACKBONE SHAPES
# Run a dummy forward to get the exact C / H / W at each hook layer.
# ─────────────────────────────────────────────────────────────────────────────

def _probe_backbone_shapes(yolo_model, device, imgsz=640) -> dict:
    """
    Returns {layer_idx: (C, H, W)} for backbone layers in HOOK_LAYERS.
    """
    shapes = {}
    handles = []

    for idx in HOOK_LAYERS:
        def make_hook(i):
            def hook(module, args, output):
                shapes[i] = tuple(output.shape[1:])   # (C, H, W)
            return hook
        h = yolo_model.model.model[idx].register_forward_hook(make_hook(idx))
        handles.append(h)

    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        yolo_model.model(dummy)

    for h in handles:
        h.remove()

    return shapes


# ─────────────────────────────────────────────────────────────────────────────
# PROJECTION BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_projections(vit_dim: int, backbone_shapes: dict,
                        device) -> nn.ModuleDict:
    """
    Build one 1×1 conv projection per backbone hook point.
    Maps SSL feature dim → YOLO's expected channel count at that level.
    """
    projs = nn.ModuleDict()
    for layer_idx, (out_c, _, _) in backbone_shapes.items():
        projs[str(layer_idx)] = nn.Sequential(
            nn.Conv2d(vit_dim, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        ).to(device)
    return projs


# ─────────────────────────────────────────────────────────────────────────────
# SSL ENCODER LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_vit_encoder(ssl_weights: Path, device) -> tuple[nn.Module, int]:
    """
    Load ViT-B/16 with MIM-JEPA pretrained weights.
    Returns (encoder, embed_dim).
    """
    import timm
    encoder = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,
        global_pool="",
    ).to(device)

    state = torch.load(ssl_weights, map_location=device)
    # state may be a plain state_dict or wrapped
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] Missing keys ({len(missing)}): {missing[:5]} …")
    print(f"  ViT-B/16 loaded from {ssl_weights.name}  "
          f"(768-dim, {sum(p.numel() for p in encoder.parameters())/1e6:.0f}M params)")
    return encoder, 768


def _load_resnet_encoder(ssl_weights: Path, device) -> tuple[nn.Module, int]:
    """
    Load ResNet-50 with MIM-JEPA pretrained weights.
    For CNN variants we extract multi-scale features using layer hooks
    rather than patch tokens, so we return a wrapper.
    Returns (encoder_wrapper, embed_dim).
    """
    import timm

    class ResNetTokeniser(nn.Module):
        """
        Wraps ResNet-50 feature extractor to produce ViT-style token output.
        Uses layer4 output (2048, 7×7 for 224 input → 20×20 for 640 input).
        Tokens: [B, 49, 2048] or [B, H*W, 2048].
        """
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            # resize to 224 for ResNet
            x_224 = F.interpolate(x, size=(224, 224),
                                   mode="bilinear", align_corners=False)
            # backbone with global_pool="" returns [B, 2048, 7, 7]
            feat = self.backbone(x_224)           # [B, 2048, 7, 7]
            B, C, H, W = feat.shape
            tokens = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            return tokens

    backbone = timm.create_model(
        "resnet50",
        pretrained=False,
        num_classes=0,
        global_pool="",
    ).to(device)

    state = torch.load(ssl_weights, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, _ = backbone.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] Missing keys ({len(missing)}): {missing[:5]} …")

    encoder = ResNetTokeniser(backbone).to(device)
    n = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"  ResNet-50 tokeniser loaded from {ssl_weights.name}  "
          f"(2048-dim, {n:.0f}M params)")
    return encoder, 2048


def _load_vjepa2_encoder(weights_dir: Path, device) -> tuple[nn.Module, int]:
    """
    Load V-JEPA 2 ViT-L/16 from a local HuggingFace snapshot.
    Strips the predictor immediately after loading to free ~4GB VRAM.
    The encoder is frozen — only the projection adapters are trained.
    Returns (encoder_wrapper, embed_dim).
    """
    from transformers import AutoModel

    print(f"  Loading V-JEPA 2 from {weights_dir} …")
    model = AutoModel.from_pretrained(
        str(weights_dir),
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    # Strip predictor to reclaim VRAM before moving encoder to GPU
    for attr in ("predictor", "teacher", "teacher_encoder",
                 "ema_encoder", "decoder"):
        if hasattr(model, attr):
            delattr(model, attr)
            print(f"  Stripped: {attr}")

    # Identify the encoder sub-module
    if hasattr(model, "encoder"):
        encoder_raw = model.encoder.to(device)
    elif hasattr(model, "vision_model"):
        encoder_raw = model.vision_model.to(device)
    else:
        encoder_raw = model.to(device)

    encoder_raw.eval()
    for p in encoder_raw.parameters():
        p.requires_grad = False

    n = sum(p.numel() for p in encoder_raw.parameters()) / 1e6
    print(f"  V-JEPA 2 encoder on device  ({n:.0f}M params, frozen)")

    class VJEPA2Tokeniser(nn.Module):
        """
        Wraps V-JEPA 2 encoder to extract spatial patch tokens.
        Treats each image as a 1-frame video [B, C, 1, H, W].
        Returns [B, N, D] patch tokens (CLS stripped if present).
        """
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        @torch.no_grad()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            # Resize to 224 (V-JEPA 2 pretraining resolution)
            x_224 = F.interpolate(
                x.float(), size=(224, 224),
                mode="bilinear", align_corners=False
            ).half()   # encoder expects fp16

            # V-JEPA 2 is a video encoder: wrap as 1-frame video
            # shape [B, C, T, H, W]
            x_vid = x_224.unsqueeze(2)   # [B, C, 1, 224, 224]

            try:
                out = self.enc(x_vid)
            except TypeError:
                # Some HF versions expect pixel_values keyword
                out = self.enc(pixel_values=x_vid)

            # Unpack output: may be BaseModelOutput or raw tensor
            if hasattr(out, "last_hidden_state"):
                tokens = out.last_hidden_state    # [B, N, D]
            elif isinstance(out, (tuple, list)):
                tokens = out[0]
            else:
                tokens = out

            # If temporal dimension remains, collapse T=1
            if tokens.dim() == 4:
                B2, T, N, D = tokens.shape
                tokens = tokens.reshape(B2, T * N, D)

            return tokens.float()           # back to fp32 for projection

    tokeniser = VJEPA2Tokeniser(encoder_raw).to(device)
    embed_dim = 1024   # ViT-L embed dim
    return tokeniser, embed_dim


# ─────────────────────────────────────────────────────────────────────────────
# HOOK INJECTION
# ─────────────────────────────────────────────────────────────────────────────

def inject_hooks(yolo_model, encoder, projections: nn.ModuleDict,
                  backbone_shapes: dict, device) -> list:
    """
    Registers pre-hook + 3 forward hooks on the YOLOv8m backbone.

    Pre-hook on layer 0: runs encoder, stores tokens in shared _ViTState.
    Forward hooks on layers 4, 6, 9: replace backbone outputs with
    projected ViT features.

    Returns list of hook handles (call h.remove() to deregister).
    """
    state   = _ViTState()
    handles = []

    # Pre-hook: extract tokens before layer 0 processes anything
    def make_extract_fn(enc, dev):
        def extract(x):
            # resize input to 224 for ViT (encoder handles this internally
            # in the tokeniser wrappers, but calling it here is clean)
            return enc(x)
        return extract

    extract_fn = make_extract_fn(encoder, device)
    pre_hook   = _ViTPreHook(extract_fn, state)
    h = yolo_model.model.model[0].register_forward_pre_hook(pre_hook)
    handles.append(h)

    # Forward hooks on P3, P4, P5 layers
    hook_classes = [_HookP3, _HookP4, _HookP5]
    for (layer_idx, (out_c, out_h, out_w)), HookClass in zip(
        backbone_shapes.items(), hook_classes
    ):
        proj  = projections[str(layer_idx)]
        fhook = HookClass(state, proj, out_h, out_w)
        h     = yolo_model.model.model[layer_idx].register_forward_hook(fhook)
        handles.append(h)

    print(f"  Hooks registered: pre-hook@0 + forward@{list(backbone_shapes.keys())}")
    return handles


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM TRAINER
# Extends DetectionTrainer to:
#   1. Add projection parameters to the optimizer
#   2. Optionally add encoder parameters (C2 / C4 trainable variants)
#   3. Override final_eval to avoid weight-reload crash caused by hook injection
# ─────────────────────────────────────────────────────────────────────────────

def make_custom_trainer(projections: nn.ModuleDict,
                         encoder: nn.Module | None = None,
                         train_encoder: bool = False):
    """
    Factory that returns a ViTDetectionTrainer class capturing projections
    and (optionally) the SSL encoder in its closure.

    Args:
        projections:   nn.ModuleDict of P3/P4/P5 projection Conv2d modules
        encoder:       SSL encoder (None for frozen C3 variant)
        train_encoder: True for C2/C4 (trainable ViT/ResNet backbone)
    """
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import LOGGER
    from ultralytics.utils.torch_utils import strip_optimizer

    extra_params = list(projections.parameters())
    if train_encoder and encoder is not None:
        extra_params += list(encoder.parameters())

    class ViTDetectionTrainer(DetectionTrainer):

        def build_optimizer(self, model, name="AdamW", lr=0.001,
                            momentum=0.9, decay=1e-5, iterations=1e4):
            """Augment the standard optimizer with projection (and encoder) params."""
            optimizer = super().build_optimizer(
                model, name=name, lr=lr,
                momentum=momentum, decay=decay, iterations=iterations)
            if extra_params:
                optimizer.add_param_group({
                    "params":       extra_params,
                    "lr":           lr,
                    "weight_decay": decay,
                })
                n_extra = sum(p.numel() for p in extra_params) / 1e6
                LOGGER.info(f"  Added {n_extra:.2f}M extra params to optimizer")
            return optimizer

        def final_eval(self):
            """
            Override to prevent the crash that occurs when Ultralytics tries
            to reload best.pt and run validation — the serialised YOLO weights
            don't include our hook state, so the reload produces wrong results.
            We only strip the optimizer (reduces file size) and skip re-validation.
            """
            for f in [self.last, self.best]:
                if f.is_file():
                    strip_optimizer(f)
            LOGGER.info(f"Training complete. Best weights: {self.best}")

    return ViTDetectionTrainer


# ─────────────────────────────────────────────────────────────────────────────
# LABEL BUDGET SPLIT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def create_fraction_yaml(data_yaml: Path, fraction: float,
                          scratch_dir: Path) -> Path:
    """
    Creates a subset of the training set at `fraction` of original size.
    Val and test splits are unchanged.
    Returns path to a new data YAML pointing at the subset.
    """
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    data_root  = Path(cfg["path"])
    train_imgs = sorted((data_root / cfg["train"]).glob("*.jpg"))
    train_lbls = data_root / cfg["train"].replace("images", "labels")

    random.seed(SEED)
    n_subset = max(1, int(len(train_imgs) * fraction))
    subset   = random.sample(train_imgs, n_subset)

    tag     = f"f{int(fraction * 100):03d}"
    img_out = scratch_dir / tag / "images"
    lbl_out = scratch_dir / tag / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path in subset:
        shutil.copy2(img_path, img_out / img_path.name)
        lbl_path = train_lbls / img_path.with_suffix(".txt").name
        if lbl_path.exists():
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

    new_cfg         = cfg.copy()
    new_cfg["train"] = str(img_out)
    new_yaml = scratch_dir / f"dataset_{tag}.yaml"
    with open(new_yaml, "w") as f:
        yaml.dump(new_cfg, f, default_flow_style=False)

    print(f"  [{tag}] {n_subset}/{len(train_imgs)} train images")
    return new_yaml


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TRAINING RUN
# ─────────────────────────────────────────────────────────────────────────────

def train_single(
    variant:        str,
    data_yaml:      Path,
    output_dir:     Path,
    ssl_weights:    Path | None,
    vjepa2_weights: Path | None,
    epochs:         int,
    patience:       int,
    batch:          int,
    device:         str,
    label_fraction: float = 1.0,
) -> dict:
    from ultralytics import YOLO

    dev = torch.device(
        f"cuda:{device}" if str(device).isdigit() and torch.cuda.is_available()
        else "cpu"
    )

    tag      = f"f{int(label_fraction*100):03d}"
    run_name = f"{variant}_{tag}"

    # ── create label-budget subset YAML ───────────────────────────────────
    if label_fraction < 1.0:
        scratch = output_dir / "fraction_data"
        data_yaml = create_fraction_yaml(data_yaml, label_fraction, scratch)

    # ── load SSL encoder + build projections ──────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Variant        : {variant}  ({label_fraction:.0%} labels)")
    print(f"  Run name       : {run_name}")
    print(f"{'─'*60}")

    if variant == "C2":
        assert ssl_weights and ssl_weights.exists(), \
               f"--ssl_weights not found: {ssl_weights}"
        encoder, embed_dim = _load_vit_encoder(ssl_weights, dev)
        train_encoder      = True
        frozen_backbone    = False

    elif variant == "C3":
        assert vjepa2_weights and vjepa2_weights.exists(), \
               f"--vjepa2_weights dir not found: {vjepa2_weights}"
        encoder, embed_dim = _load_vjepa2_encoder(vjepa2_weights, dev)
        train_encoder      = False    # V-JEPA 2 fully frozen
        frozen_backbone    = True

    elif variant == "C4":
        assert ssl_weights and ssl_weights.exists(), \
               f"--ssl_weights not found: {ssl_weights}"
        encoder, embed_dim = _load_resnet_encoder(ssl_weights, dev)
        train_encoder      = True
        frozen_backbone    = False

    else:
        raise ValueError(f"Unknown variant: {variant}. Choose C2, C3, or C4.")

    # ── build base YOLO model ─────────────────────────────────────────────
    yolo = YOLO("yolov8m.pt")

    # freeze backbone for frozen-SSL variants (C3)
    if frozen_backbone:
        for i in range(10):   # backbone = layers 0-9
            for p in yolo.model.model[i].parameters():
                p.requires_grad = False
        print("  Backbone layers 0–9 frozen")

    # ── probe backbone shapes ─────────────────────────────────────────────
    backbone_shapes = _probe_backbone_shapes(yolo, dev)
    print("  Backbone shapes at hook points:")
    for idx, shape in backbone_shapes.items():
        print(f"    layer {idx}: C={shape[0]}  H={shape[1]}  W={shape[2]}")

    # ── build projections ─────────────────────────────────────────────────
    projections = _build_projections(embed_dim, backbone_shapes, dev)
    n_proj = sum(p.numel() for p in projections.parameters()) / 1e6
    print(f"  Projection adapters: {n_proj:.2f}M params")

    # ── inject hooks ──────────────────────────────────────────────────────
    hook_handles = inject_hooks(yolo, encoder, projections,
                                 backbone_shapes, dev)

    # ── custom trainer ────────────────────────────────────────────────────
    TrainerClass = make_custom_trainer(
        projections=projections,
        encoder=encoder,
        train_encoder=train_encoder,
    )

    # ── training ──────────────────────────────────────────────────────────
    results = yolo.train(
        data        = str(data_yaml),
        epochs      = epochs,
        batch       = batch,
        imgsz       = 640,
        device      = device,
        project     = str(output_dir),
        name        = run_name,
        exist_ok    = True,
        seed        = SEED,
        optimizer   = "AdamW",
        lr0         = 1e-3,
        lrf         = 0.01,
        momentum    = 0.937,
        weight_decay= 5e-4,
        warmup_epochs=3,
        patience    = patience,
        augment     = True,
        mosaic      = 0.5,
        mixup       = 0.2,
        flipud      = 0.1,
        fliplr      = 0.5,
        degrees     = 30.0,
        translate   = 0.1,
        scale       = 0.5,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        cache       = "disk",
        workers     = 4,
        amp         = True,
        verbose     = False,
        trainer     = TrainerClass,
    )

    # remove hooks after training
    for h in hook_handles:
        h.remove()

    # ── extract metrics ───────────────────────────────────────────────────
    metrics = {
        "variant":        variant,
        "label_fraction": label_fraction,
        "run_name":       run_name,
    }
    try:
        rd = results.results_dict
        metrics.update({
            "mAP50":     round(float(rd.get("metrics/mAP50(B)",    0)), 5),
            "mAP50_95":  round(float(rd.get("metrics/mAP50-95(B)", 0)), 5),
            "precision": round(float(rd.get("metrics/precision(B)",0)), 5),
            "recall":    round(float(rd.get("metrics/recall(B)",   0)), 5),
        })
    except Exception as e:
        print(f"  [warn] Could not parse results_dict: {e}")

    # save metrics JSON
    out_json = output_dir / run_name / "metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  [{run_name}] "
          f"mAP50={metrics.get('mAP50', 'N/A')}  "
          f"mAP50-95={metrics.get('mAP50_95', 'N/A')}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# LABEL BUDGET SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_label_budget_sweep(
    variant, data_yaml, output_dir, ssl_weights,
    vjepa2_weights, epochs, patience, batch, device,
) -> list:
    print(f"\n{'='*60}")
    print(f"  Label budget sweep — {variant}")
    print(f"  Budgets: {[f'{b:.0%}' for b in LABEL_BUDGETS]}")
    print(f"{'='*60}")

    all_results = []
    for fraction in LABEL_BUDGETS:
        metrics = train_single(
            variant        = variant,
            data_yaml      = data_yaml,
            output_dir     = output_dir,
            ssl_weights    = ssl_weights,
            vjepa2_weights = vjepa2_weights,
            epochs         = epochs,
            patience       = patience,
            batch          = batch,
            device         = device,
            label_fraction = fraction,
        )
        all_results.append(metrics)

    # save sweep summary
    sweep_path = output_dir / f"{variant}_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # print table
    print(f"\n{'─'*40}")
    print(f"  {variant} sweep results:")
    print(f"  {'Budget':>8}  {'mAP50':>8}  {'mAP50-95':>10}")
    print(f"  {'─'*30}")
    for r in all_results:
        print(f"  {r['label_fraction']:>7.0%}  "
              f"{r.get('mAP50', 0):>8.4f}  "
              f"{r.get('mAP50_95', 0):>10.4f}")
    print(f"  Sweep saved: {sweep_path}")

    _plot_sweep(all_results, variant, output_dir)
    return all_results


def _plot_sweep(results: list, variant: str, output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fracs  = [r["label_fraction"] for r in results]
        map50s = [r.get("mAP50", 0)   for r in results]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([f*100 for f in fracs], map50s,
                marker="o", linewidth=2, markersize=8)
        for f, m in zip(fracs, map50s):
            ax.annotate(f"{m:.3f}", (f*100, m),
                        textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)
        ax.set_xlabel("Label budget (%)")
        ax.set_ylabel("mAP@0.5")
        ax.set_title(f"Label budget curve — {variant}")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, max(max(map50s) * 1.2, 0.1))
        plt.tight_layout()
        save_path = output_dir / f"{variant}_label_budget_curve.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Curve saved: {save_path}")
    except Exception as e:
        print(f"  [warn] Plot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Path C: SSL backbone injection into YOLOv8m")
    p.add_argument("--variant",          type=str, required=True,
                   choices=["C2", "C3", "C4"],
                   help="C2=ViT-B+JEPA  C3=ViT-L+V-JEPA2  C4=ResNet+JEPA")
    p.add_argument("--data_yaml",        type=Path, required=True,
                   help="processed_4cls/dataset_path_A.yaml")
    p.add_argument("--ssl_weights",      type=Path, default=None,
                   help="vit_jepa.pt or resnet_jepa.pt (C2 / C4)")
    p.add_argument("--vjepa2_weights",   type=Path, default=None,
                   help="V-JEPA 2 HuggingFace snapshot dir (C3)")
    p.add_argument("--output",           type=Path,
                   default=Path("runs/path_C"))
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--patience",         type=int,   default=25)
    p.add_argument("--batch",            type=int,   default=8)
    p.add_argument("--device",           type=str,   default="0")
    p.add_argument("--label_budget_sweep", action="store_true",
                   help="Run sweep over all label fractions")
    p.add_argument("--label_fraction",   type=float, default=1.0,
                   help="Single label fraction when not using --label_budget_sweep")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    assert args.data_yaml.exists(), f"YAML not found: {args.data_yaml}"

    if args.variant == "C3":
        assert args.vjepa2_weights is not None and args.vjepa2_weights.exists(), \
               "--vjepa2_weights required and must exist for C3"
    else:
        assert args.ssl_weights is not None and args.ssl_weights.exists(), \
               "--ssl_weights required and must exist for C2 / C4"

    if args.label_budget_sweep:
        run_label_budget_sweep(
            variant        = args.variant,
            data_yaml      = args.data_yaml,
            output_dir     = args.output,
            ssl_weights    = args.ssl_weights,
            vjepa2_weights = args.vjepa2_weights,
            epochs         = args.epochs,
            patience       = args.patience,
            batch          = args.batch,
            device         = args.device,
        )
    else:
        train_single(
            variant        = args.variant,
            data_yaml      = args.data_yaml,
            output_dir     = args.output,
            ssl_weights    = args.ssl_weights,
            vjepa2_weights = args.vjepa2_weights,
            epochs         = args.epochs,
            patience       = args.patience,
            batch          = args.batch,
            device         = args.device,
            label_fraction = args.label_fraction,
        )
