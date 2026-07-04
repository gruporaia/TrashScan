# TrashScan

**TrashScan** is a benchmark pipeline for automated recyclabe litter detection and classification, built on an augmented version of the [TACO dataset](https://arxiv.org/abs/2003.06975).

This is the **`main` branch**, which targets a 5-class taxonomy (`plastic`, `paper`, `metal`, `glass`, `other`) and focuses on a practical architectural question: **does a two-stage detect-then-classify pipeline (YOLO + ViT) outperform a well-tuned single-stage YOLO detector for litter detection?**

> There is a second experimental line in this repository, on branch [`4cls`](https://github.com/gruporaia/TrashScan/tree/4cls), which drops the `glass` class and adds a self-supervised pretraining path. **The two branches are different experiment versions, not directly comparable** — they use different class configurations, dataset splits, and post-processing setups. See the [Branches](#branches) section below.

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

1. **Class consolidation** into 5 recycling-relevant categories: plastic, paper, metal, glass, other.
2. **Augmentation** with the community-extended [Roboflow TACO dataset](https://universe.roboflow.com/sadis-workspace/taco-dataset-ql1ng-atu1k), growing the base collection to ~5,000 images. Validation and test sets (1,392 images each) are split off *before* augmentation and kept fixed; training data is further augmented to 9,519 images / 22,362 annotations.
3. **Copy-paste oversampling** on the training split only, to correct for residual class imbalance (e.g. plastic vs. metal), resulting in a final training partition of 12,091 images.
4. **Inference-time refinements**: Test-Time Augmentation (TTA — horizontal flips and multi-scale resizing) combined with Weighted Boxes Fusion (WBF) instead of standard NMS, applied to both experimental paths.

All models share the same fixed test set and evaluation protocol (mAP@50, mAP@[50:95], precision, recall), with mAP@50 as the main ranking metric.

---

## Path A — Single-Stage Detection (YOLO)

One-stage detectors from the YOLO family, including a custom NMS-free YOLOv11m variant trained with a dual assignment head (dense one-to-many + sparse one-to-one), removing the need for NMS post-processing at inference.

| Configuration | Method | mAP@50 | mAP@[50:95] | Prec. | Rec. |
|---|---|---|---|---|---|
| Without TTA/WBF | YOLOv11m no NMS | 0.7082 | 0.4740 | 0.5828 | 0.5592 |
| Without TTA/WBF | YOLOv8m | 0.6041 | 0.4383 | 0.5564 | 0.5360 |
| Without TTA/WBF | YOLOv11m | 0.5699 | 0.3381 | 0.5613 | 0.4600 |
| **With TTA/WBF** | **YOLOv11m no NMS** | **0.7284** | **0.4929** | 0.5920 | 0.5592 |
| With TTA/WBF | YOLOv8m | 0.7080 | 0.4637 | 0.6023 | 0.5575 |
| With TTA/WBF | YOLOv11m | 0.5979 | 0.3469 | 0.5571 | 0.4401 |

Best result: **YOLOv11m no NMS + TTA/WBF, mAP@50 = 0.7284**.

---

## Path B — Two-Stage Detect-and-Classify (YOLO + ViT)

The frozen best detector from Path A (YOLOv11m no NMS) proposes boxes; crops (224×224) are classified by a second-stage model. Three ImageNet-pretrained classifiers are compared: ResNet-50 (B1), ViT-B/16 (B2), ViT-L/16 (B3).

| Configuration | Method | mAP@50 | mAP@[50:95] |
|---|---|---|---|
| Without TTA/WBF | B2 ViT-B/16 + ImageNet | 0.7097 | 0.4768 |
| Without TTA/WBF | B3 ViT-L/16 + ImageNet | 0.7039 | 0.4740 |
| Without TTA/WBF | B1 ResNet-50 + ImageNet | 0.6775 | 0.4574 |
| **With TTA/WBF** | **B3 ViT-L/16 + ImageNet** | **0.7112** | **0.4857** |
| With TTA/WBF | B1 ResNet-50 + ImageNet | 0.6907 | 0.4738 |
| With TTA/WBF | B2 ViT-B/16 + ImageNet | 0.6721 | 0.4628 |

Best result: **B3 ViT-L/16 + ImageNet + TTA/WBF, mAP@50 = 0.7112**.

---

## Discussion: YOLO vs. YOLO + ViT

In this 5-class setup, the single-stage detector (0.7284) edges out the best two-stage pipeline (0.7112). The two-stage design also roughly doubles inference latency, since it runs a full detector and a full classifier sequentially, without a matching gain in accuracy. Likely causes: error propagation from imperfect boxes into the classifier, and loss of surrounding context when litter objects are cropped and resized in isolation.

Interestingly, this ranking **flips** on the `4cls` branch: there, the two-stage YOLO + ViT pipeline slightly outperforms the single-stage YOLO baseline. We don't read this as a contradiction so much as a sign that the comparison is sensitive to setup — the two branches differ in class count (glass included vs. excluded), post-processing (TTA/WBF here vs. none there), and dataset partitioning. Because of these differences, **the numbers across branches should not be compared directly**; the flip itself is nonetheless a useful discussion point on how fragile single-stage vs. two-stage conclusions can be to experimental configuration.

---

## Branches

| Branch | Classes | Focus | Notes |
|---|---|---|---|
| `main` (this one) | 5 (plastic, paper, metal, glass, other) | YOLO vs. YOLO+ViT | Uses TTA/WBF; includes NMS-free YOLO variant |
| [`4cls`](https://github.com/gruporaia/TrashScan/tree/4cls) | 4 (plastic, paper, metal, other — no glass) | Self-supervised pretraining (SSL) label efficiency | Adds Path C (MIM-JEPA / V-JEPA 2); no TTA/WBF |

Path C (self-supervised pretraining) was only trained and evaluated on the 4-class setup and is therefore exclusive to the `4cls` branch.

---

## Presentation

This project was presented (in Portuguese) at RAIA's event: [YouTube live recording](https://www.youtube.com/live/Cl9wILmCumg?si=wAn-TMCbL7Mca-us&t=7753).

While the source code is in English, some notebooks used as an interface to the underlying code are written in Portuguese.

---

## References

- Proença, P. F., & Simões, P. (2020). *TACO: Trash Annotations in Context for litter detection*. arXiv:2003.06975.
- Majchrowska, S., et al. (2022). *Deep learning-based waste detection in natural and urban environments*. Waste Management, 138, 274–284.
- Tarimo, S. A., et al. (2024). *WBC YOLO-ViT: 2-way 2-stage white blood cell detection and classification with a combination of YOLOv5 and Vision Transformer*. Computers in Biology and Medicine, 169.
- Ghiasi, G., et al. (2021). *Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation*. CVPR.
- Solovyev, R., Wang, W., & Gabruseva, T. (2021). *Weighted Boxes Fusion*. Image and Vision Computing.