# TrashScan (branch: `4cls`)

**TrashScan** is a benchmark pipeline for automated litter detection and classification, built on an augmented version of the [TACO dataset](https://arxiv.org/abs/2003.06975).

This is the **`4cls` branch**, which uses a 4-class taxonomy (`plastic`, `paper`, `metal`, `other` — **no glass**) and focuses on a different question from the main branch: **can self-supervised (SSL) pretraining reduce dependency on labeled bounding-box data for litter detection, and how does it compare to full supervision?**

> The [`main`](https://github.com/gruporaia/TrashScan) branch runs a separate experiment with 5 classes (glass included), TTA/WBF post-processing, and a different focus (single-stage YOLO vs. two-stage YOLO+ViT). **These two branches are different versions of the experiment, not directly comparable** — the class configuration, dataset split, and post-processing differ between them.

## About Us

### Team:

- Gabriel Fagundes Mesquita Sousa (Federal University of Lavras — UFLA) — Project Director
- Matheus Henrique Rosado Vicente (Eldorado Research Institute) — Project Coordinator
- Lucas de Oliveira Ferreira (University of São Paulo — USP)
- Caroline Akimi Kurosaki Ueda (University of São Paulo — USP)
- Pedro Henrique de Holanda (University of São Paulo — USP)
- Gabriel de Andrade Abreu (University of São Paulo — USP)
- Andre Luis Debiaso Rossi (São Paulo State University — UNESP) — Advisor

Our thanks to RAIA and to everyone involved for their contributions to this project.

### Who we are

This project was developed by members of RAIA — Rede de Avanço de Inteligência Artificial, a student-led initiative at the Institute of Mathematics and Computer Sciences (ICMC) of the University of São Paulo (USP), São Carlos. RAIA's goal is to build innovative artificial intelligence solutions that generate a positive impact on society.

### Learn more

- Website: https://grupo-raia.org/
- Instagram: https://instagram.com/grupo.raia

---

## Data & Pre-processing

Starting from the original TACO dataset (1,500 images, 60 fine-grained classes), this branch applies:

1. **Class consolidation** into 4 categories: plastic, paper, metal, other. Unlike the `main` branch, **glass is not treated as a separate class here**.
2. **Augmentation** with the community-extended [Roboflow TACO dataset](https://universe.roboflow.com/sadis-workspace/taco-dataset-ql1ng-atu1k), growing the base collection to ~5,000 images, further augmented to 9,519 images / 22,362 annotations. Validation and test sets (1,392 images each) are kept fixed and untouched by augmentation.
3. **Copy-paste oversampling** on the training split only, correcting for class imbalance (e.g. plastic vs. metal instance counts), resulting in a final training partition of 12,091 images.
4. No TTA/WBF inference-time refinement is used in this branch (unlike `main`); evaluation is done directly on model outputs.

All models share the same fixed test set and evaluation protocol (mAP@50, mAP@[50:95], precision, recall), with mAP@50 as the main ranking metric.

---

## Path A — Single-Stage Detection (YOLO)

Fully supervised YOLO-family detectors (YOLOv8, YOLOv9, YOLOv11, RT-DETR) trained with AdamW and standard mosaic/mixup augmentation, used both as a baseline and as the frozen detector for Path B.

| Model | mAP@50 | mAP@[50:95] | Prec. | Rec. |
|---|---|---|---|---|
| **YOLOv8m** | **0.7580** | **0.5694** | 0.8003 | 0.7037 |
| YOLOv9s | 0.7331 | 0.5330 | 0.8050 | 0.6661 |

For reference, models evaluated on the raw, non-consolidated TACO dataset ("original" configuration) perform far worse (e.g. YOLOv8m original: mAP@50 = 0.2159), confirming the value of the consolidation/augmentation pipeline.

---

## Path B — Two-Stage Detect-and-Classify (YOLO + ViT)

The frozen YOLOv8m detector from Path A proposes boxes; ground-truth crops (224×224) are used to train second-stage classifiers: ResNet-50 with ImageNet pretraining (B1), ViT-B/16 from scratch (B2), and ViT-B/16 with ImageNet pretraining (B3).

| Model | mAP@50 | mAP@[50:95] | Prec. | Rec. |
|---|---|---|---|---|
| **B3 ViT-B/16 + ImageNet** | **0.7727** | 0.5433 | 0.8328 | 0.7457 |
| B1 ResNet-50 + ImageNet | 0.7516 | 0.5277 | 0.8162 | 0.7376 |
| B2 ViT-B/16 + Scratch | 0.4994 | 0.3516 | 0.6508 | 0.5977 |

Best result: **B3 ViT-B/16 + ImageNet, mAP@50 = 0.7727**, ahead of the best Path A baseline (0.7580). The gap between B2 (trained from scratch) and B3 (ImageNet-pretrained) shows that pretraining, not the transformer architecture itself, drives most of this gain. As in the `main` branch, the two-stage design still costs roughly double the inference latency of a single detector.

> Note: unlike the `main` branch, here **YOLO + ViT (Path B) outperforms plain YOLO (Path A)**. Given the differences in class count, post-processing, and Path A/B implementation details across branches, this is discussed as an interesting contrast rather than a direct disagreement — see the `main` branch README for that discussion.

---

## Path C — Self-Supervised Pretrained Detection

Path C is **exclusive to this branch** (4 classes only) and evaluates whether self-supervised pretraining improves label efficiency. Pretrained encoders are injected as multi-scale features into a YOLOv8m detection framework, replacing the CSPDarknet backbone while keeping the FPN+PAN neck and detection head. Three variants are tested under label budgets from 10% to 100%:

- **C1** — ViT-B/16 + MIM-JEPA (domain-specific, pretrained on TACO images, fine-tuned)
- **C2** — ViT-L/16 + V-JEPA 2 (general-purpose, pretrained on large-scale internet video, frozen)
- **C3** — ResNet-50 + MIM-JEPA (domain-specific, pretrained on TACO images, fine-tuned)

| Label budget | Method | mAP@50 | mAP@[50:95] | Prec. | Rec. |
|---|---|---|---|---|---|
| 10% | C3 ResNet-50 + MIM-JEPA | 0.4557 | 0.3025 | 0.5634 | 0.4311 |
| 10% | C1 ViT-B/16 + MIM-JEPA | 0.4374 | 0.2911 | 0.5233 | 0.4222 |
| 10% | C2 ViT-L/16 + V-JEPA 2 | 0.0277 | 0.0041 | 0.0946 | 0.0477 |
| 25% | C3 ResNet-50 + MIM-JEPA | 0.5804 | 0.4188 | 0.6881 | 0.5527 |
| 25% | C1 ViT-B/16 + MIM-JEPA | 0.5577 | 0.3890 | 0.6483 | 0.5182 |
| 25% | C2 ViT-L/16 + V-JEPA 2 | 0.0480 | 0.0117 | 0.0931 | 0.0526 |
| 50% | C3 ResNet-50 + MIM-JEPA | 0.6663 | 0.4995 | 0.7596 | 0.6286 |
| 50% | C1 ViT-B/16 + MIM-JEPA | 0.6355 | 0.4553 | 0.7276 | 0.5835 |
| 50% | C2 ViT-L/16 + V-JEPA 2 | 0.0750 | 0.0210 | 0.1050 | 0.0620 |
| 75% | C1 ViT-B/16 + MIM-JEPA | 0.6903 | 0.5062 | 0.7782 | 0.6458 |
| 75% | C3 ResNet-50 + MIM-JEPA | 0.6830 | 0.5031 | 0.7798 | 0.6352 |
| 75% | C2 ViT-L/16 + V-JEPA 2 | 0.1050 | 0.0310 | 0.1180 | 0.0820 |
| **100%** | **C3 ResNet-50 + MIM-JEPA** | **0.7587** | **0.5915** | 0.8198 | 0.7125 |
| 100% | C1 ViT-B/16 + MIM-JEPA | 0.7200 | 0.5388 | 0.7763 | 0.6808 |
| 100% | C2 ViT-L/16 + V-JEPA 2 | 0.1350 | 0.0410 | 0.1320 | 0.0980 |

At full data, **C3 (ResNet-50 + domain-specific MIM-JEPA)** reaches mAP@50 = 0.7587, matching/slightly beating the best fully supervised baseline (YOLOv8m, 0.7580) and achieving the best localization quality of all tested models (mAP@[50:95] = 0.5915). C1 and C3 degrade gradually as labeled data shrinks, staying usable even at 10% labels, while C2 (frozen, general-purpose V-JEPA 2) never leaves very low performance (mAP@50 < 0.14 across all budgets). This indicates that **pretraining on domain-relevant data matters more than backbone scale** for this task.

---

## Branches

| Branch | Classes | Focus | Notes |
|---|---|---|---|
| [`main`](https://github.com/gruporaia/TrashScan) | 5 (plastic, paper, metal, glass, other) | YOLO vs. YOLO+ViT | Uses TTA/WBF; includes NMS-free YOLO variant |
| `4cls` (this one) | 4 (plastic, paper, metal, other — no glass) | Self-supervised pretraining (SSL) label efficiency | Adds Path C (MIM-JEPA / V-JEPA 2); no TTA/WBF |

For the discussion comparing the YOLO-vs-YOLO+ViT trend seen here against the `main` branch, see the [`main` branch README](https://github.com/gruporaia/TrashScan#4-discussion-yolo-vs-yolo--vit).

---

## Presentation

This project was presented (in Portuguese) at RAIA's event: [YouTube live recording](https://www.youtube.com/live/Cl9wILmCumg?si=wAn-TMCbL7Mca-us&t=7753).

While the source code is in English, some notebooks used as an interface to the underlying code are written in Portuguese.

---

## References

- Proença, P. F., & Simões, P. (2020). *TACO: Trash Annotations in Context for litter detection*. arXiv:2003.06975.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR.
- Dosovitskiy, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. arXiv:2010.11929.
- Assran, M., et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)*. CVPR.
- Bardes, A., et al. (2024). *V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video*. arXiv:2404.08471.
- Assran, M., et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. arXiv:2506.09985.
- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. OpenReview.
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR.
- Loshchilov, I., & Hutter, F. (2017). *Decoupled Weight Decay Regularization*. arXiv:1711.05101.
