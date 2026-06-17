# TrashScan

Efficient solid waste management is a major environmental challenge in urban areas. The visual variability of discarded materials, such as crushed, dirty, or partially occluded objects, makes automated sorting using traditional computer vision methods difficult.

TrashScan is a deep learning-based waste detection and classification project. It serves as a benchmarking pipeline to evaluate and compare a single-stage detector and classifier against a two-stage approach.

## Problem Statement

Automating waste sorting faces significant challenges due to the high visual variability of discarded materials. Deformed packaging, contaminated waste, overlapping objects, and varying lighting conditions make detection and classification complex tasks.

TrashScan investigates deep learning solutions to:

* Automatically detect waste in images.
* Classify waste into recyclable and non-recyclable categories.
* Compare single-stage and two-stage architectures.
* Evaluate performance regarding accuracy and latency.

## Pipeline Architecture and Methods

This project implements a modular benchmarking pipeline to evaluate different object detection and classification strategies.

### Path A: Single-Stage Detector and Classifier (YOLO)

This method utilizes YOLO (You Only Look Once) as a single-stage architecture. The network performs both bounding box regression (localization) and class probability prediction (classification) simultaneously in one forward pass. This approach is highly optimized for fast inference and real-time processing.

### Path B: Two-Stage Detector and Classifier (YOLO + ViT)

This method isolates the tasks into two separate models:

1. **Detection (Stage 1):** A YOLO model is used strictly to identify the spatial location of objects and extract bounding boxes.
2. **Classification (Stage 2):** The cropped bounding boxes are fed into a Vision Transformer (ViT). The ViT handles the complex visual features to perform the final classification. This path evaluates if the robust feature extraction capabilities of ViTs can improve classification accuracy over the single-stage approach.

### Path C: Extended Pipeline

Additionally, the repository includes Path C, an alternative experimental approach. While the current primary benchmark focuses on comparing Path A and Path B, Path C is maintained within the codebase for broader research and extended evaluations.