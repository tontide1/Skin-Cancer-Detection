# Skin-Cancer-Detection — AGENTS.md

## Project Overview

This project researches and builds Computer Vision models for **skin lesion segmentation** using the [ISIC 2018 Challenge - Task 1](https://challenge.isic-archive.com/landing/2018/) dataset. The goal is to reach **Dice > 0.90** and eventually deploy the best model as a **web application**.

**Current Best Results:**
- Architecture: U-Net + ResNet34 encoder + SCSE attention decoder (via `segmentation-models-pytorch`)
- Test Dice: **0.9021** | Test IoU: **0.8368** (best epoch: 41)

---

## Tech Stack

| Component | Details |
|---|---|
| Language | Python 3.11 (Conda) |
| DL Framework | PyTorch 2.x + torchvision |
| Segmentation Library | `segmentation-models-pytorch >= 0.3` (SMP) |
| Augmentation | `albumentations >= 1.3` |
| Experiment Tracking | Weights & Biases (W&B) + local CSV/JSON fallback |
| Config System | YAML with `_base_` inheritance + CLI dot-notation overrides |
| Package | Editable install via `pip install -e .` (package root: `src/`) |
| Environment | Conda (`environment.yml`) or pip (`requirements.txt` for Kaggle) |

---

## Project Structure

```
Skin-Cancer-Detection/
├── AGENTS.md                        ← This file (OpenCode + Cursor rules)
├── configs/
│   ├── base.yaml                    ← Master defaults for ALL experiments
│   └── experiments/
│       └── resnet34_unet_v1.yaml    ← Experiment overrides (inherits base.yaml via _base_)
├── data/
│   ├── raw/                         ← ISIC2018_Task1_Input/ + ISIC2018_Task1_GroundTruth/
│   └── processed/                   ← train/val/test splits (after prepare_data.py)
├── outputs/                         ← Training outputs (checkpoints, curves, results)
├── notebooks/                       ← Jupyter notebooks for EDA/experiments
├── scripts/
│   ├── prepare_data.py              ← Dataset split 70/15/15
│   ├── train.py                     ← Main training entry point
│   ├── evaluate.py                  ← Test evaluation with TTA + threshold search
│   └── predict.py                   ← Inference + overlay visualization
├── src/                             ← Installable Python package (pip install -e .)
│   ├── data/                        ← ISICDataset, Albumentations transforms
│   ├── losses/                      ← FocalLoss, SoftDiceLoss, CombinedLoss
│   ├── metrics/                     ← dice_coefficient, iou_score, find_best_threshold
│   ├── models/                      ← Registry-based model factory (create_model)
│   ├── training/                    ← Trainer, EarlyStopping, ModelCheckpoint, callbacks
│   └── utils/                       ← Config loader, W&B logger, set_seed, misc
└── stage-1-segmentation/            ← Historical Kaggle notebook artifacts (reference only)
```

---

## Core Conventions & Rules

### 1. Model Registry Rule (CRITICAL)
All new model architectures **MUST** be registered in `src/models/segmentation.py` via the `_REGISTRY` dict.
Never hardcode model classes in training scripts or notebooks.

```python
# src/models/segmentation.py — the only place to define/register models
_REGISTRY = {
    "resnet34_unet":       _build_resnet34_unet,
    "efficientnet_unet":   _build_efficientnet_unet,  # <- add new models here
    "sam_finetune":        _build_sam_finetune,
}
# Always use create_model(config) as the single entry point
```

### 2. Experiment Config Rule
Every new experiment **MUST** have its own YAML in `configs/experiments/` inheriting from `base.yaml`:

```yaml
# configs/experiments/efficientnetb4_unet_v1.yaml
_base_: ../base.yaml
model:
  name: efficientnet_unet
  encoder_name: efficientnet-b4
  encoder_weights: imagenet
output:
  dir: outputs/efficientnetb4_unet_v1
```

Never modify `base.yaml` for experiment-specific settings.

### 3. Metric Reporting Rule
All evaluation results **MUST** report these 4 metrics together — no exceptions:
- `test_dice` (threshold = 0.5)
- `test_iou` (threshold = 0.5)
- `test_dice_best` (at optimal threshold from grid search)
- `best_threshold`

### 4. Reproducibility Rule
- Set `seed` in every YAML config (default: 42)
- Call `set_seed(config.seed)` at the start of every training/eval script
- Save `model_config` alongside `state_dict` in every checkpoint

### 5. Loss Function Rule
Default: `CombinedLoss(0.5 × FocalLoss + 0.5 × SoftDiceLoss)`.
New losses → add to `src/losses/segmentation.py` and wire via `training.loss` in config.

### 6. Data Augmentation Rule
- Train augmentations live **only** in `src/data/transforms.py`
- Val/test use **only** `Resize + Normalize + ToTensorV2`
- When adding augmentations, maintain the Albumentations pipeline structure

---

## Adding a New Architecture — Checklist

1. Add `_build_<name>(cfg)` in `src/models/segmentation.py`
2. Register: `_REGISTRY["<name>"] = _build_<name>`
3. Create `configs/experiments/<name>_v1.yaml` (inherit `_base_: ../base.yaml`)
4. Train: `python scripts/train.py --config configs/experiments/<name>_v1.yaml`
5. Evaluate: `python scripts/evaluate.py --checkpoint outputs/<name>/best_model.pth`
6. Results auto-save to `outputs/<experiment_name>/`

**Candidate architectures to try (ordered by expected impact):**
- `efficientnet-b4` / `efficientnet-b6` encoder (SMP) — drop-in swap
- `mit_b2` / `mit_b4` Mix Transformer encoder (SegFormer backbone + UNet decoder)
- `timm-resnext50_32x4d` or `timm-senet154` encoder
- `DeepLabV3+` decoder with ResNet50/EfficientNet encoder
- SAM (Segment Anything Model) fine-tuned on ISIC Task 1
- Swin-UNet / TransUNet for Transformer-based segmentation

---

## Web Product Target

The project aims to ship a web application for skin lesion segmentation. Keep this in mind during development:

- `scripts/predict.py` supports `--overlay` flag (3-panel PNG: original | mask | overlay)
- Target inference: **< 1s per image** at 256×256 on CPU (for cloud deployment)
- Planned backend: **FastAPI** + model loading
- ONNX export for lighter deployment
- Consider `BentoML` or `TorchServe` for production model serving

---

## Available Skills

The following skills are in `.agents/skills/` — auto-loaded by **Cursor** and **OpenCode**:

| Skill | Purpose |
|---|---|
| `computer-vision-expert` | SOTA CV: SAM, SegFormer, VLMs, deployment optimization |
| `ml-engineer` | PyTorch 2.x production, SMP patterns, TTA, model serving |
| `mlops-engineer` | W&B tracking, experiment lifecycle, pipeline automation |
| `python-pro` | Python 3.11+ best practices, type hints, performance |
| `data-scientist` | Statistical analysis, EDA, ablation study, model comparison |

Use `/skill-name` in Cursor Agent chat to invoke manually, or they auto-load based on context.

---

## Training Commands Reference

```bash
# 1. Prepare dataset (run once)
python scripts/prepare_data.py --raw-dir data/raw --output-dir data/processed

# 2. Train a model
python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml

# 3. Train with CLI overrides (no need to edit YAML)
python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml \
    training.batch_size=32 \
    model.encoder_name=efficientnet-b4 \
    output.dir=outputs/efficientnetb4_test

# 4. Evaluate on test set (with TTA + best threshold search)
python scripts/evaluate.py \
    --config configs/experiments/resnet34_unet_v1.yaml \
    --checkpoint outputs/resnet34_unet_v1/best_model.pth

# 5. Predict with overlay visualization
python scripts/predict.py \
    --image data/processed/test/images/ISIC_0024306.jpg \
    --checkpoint outputs/resnet34_unet_v1/best_model.pth \
    --overlay --tta

# 6. Kaggle-compatible run
python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml \
    data.root=/kaggle/input/isic-2018 \
    output.dir=/kaggle/working \
    logging.use_wandb=false
```
