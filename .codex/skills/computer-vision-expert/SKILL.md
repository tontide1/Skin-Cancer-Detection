---
name: computer-vision-expert
description: Design, train, evaluate, and deploy computer vision systems for classification, detection, segmentation, tracking, OCR, and vision-language tasks. Use when the task involves image data, model architecture choices, dataset strategy, augmentation, evaluation, or edge and server deployment tradeoffs.
---

# Computer Vision Expert

## Quick Start

1. Define the vision task, label format, and deployment target.
2. Check data quality, annotation consistency, class balance, and leakage risk.
3. Choose the simplest model family that fits latency, memory, and accuracy constraints.
4. Build training, evaluation, and inference paths around the real deployment environment.
5. Validate with error analysis on representative failure modes, not just headline metrics.

## Workflow

### Frame the task correctly

- Distinguish between classification, detection, segmentation, tracking, OCR, and VLM-style reasoning.
- Match labels, augmentations, and metrics to the task instead of reusing a generic recipe.
- Treat dataset bias and annotation noise as first-order problems.

### Design the pipeline

- Start from strong baselines before adding larger backbones or multimodal components.
- Keep preprocessing and resize policy identical between training and inference when possible.
- Prefer clear post-processing over magical threshold choices hidden in notebooks.
- For segmentation and detection, inspect failure cases visually on actual images.

### Optimize for deployment

- Measure memory, throughput, and export compatibility early.
- Validate preprocessing, color space, and normalization in the serving stack.
- Check robustness to blur, compression, lighting shifts, and input size changes.

### Report outcomes clearly

- Include dataset split policy, metrics, thresholds, and qualitative failures.
- Separate modeling problems from data problems in the final recommendation.

## Deliverables

- A task-appropriate architecture and training plan.
- A reproducible evaluation approach with metric and threshold choices.
- Clear deployment constraints, bottlenecks, and next experiments.
