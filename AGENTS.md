# Skin-Cancer-Detection — AGENTS.md

## Project Overview

Binary **skin lesion segmentation** on the [ISIC 2018 Challenge – Task 1](https://challenge.isic-archive.com/landing/2018/) dataset.
Given a dermoscopy image, the model predicts a binary mask delineating the lesion boundary.

**Current Best Results:**
- Architecture: U-Net + ResNet34 encoder + SCSE attention decoder (via `segmentation-models-pytorch`)
- Test Dice: **0.9021** | Test IoU: **0.8368** (best epoch: 41)
- Goal: Beat Dice > 0.90 with new architectures, then deploy as FastAPI web application

---

## Tech Stack

| Component | Details |
|---|---|
| Language | Python 3.11 (Conda) |
| DL Framework | PyTorch 2.x + AMP (`torch.amp`) |
| Segmentation | `segmentation-models-pytorch >= 0.3` (SMP) |
| Augmentation | `albumentations >= 1.3` |
| Experiment Tracking | Weights & Biases (W&B) + local CSV/JSON fallback |
| Config System | YAML with `_base_` inheritance + CLI dot-notation overrides |
| Package | Editable install via `pip install -e .` (package root: `src/`) |
| Environment | Conda (`environment.yml`) or pip (`requirements.txt` for Kaggle) |

---

## Project Structure

```
Skin-Cancer-Detection/
├── configs/
│   ├── base.yaml                    ← Master defaults for ALL experiments
│   └── experiments/
│       └── resnet34_unet_v1.yaml    ← Overrides only (inherits base.yaml via _base_)
├── data/
│   ├── ISIC2018_Task1-2_Training_Input/       ← 2594 raw images
│   ├── ISIC2018_Task1_Training_GroundTruth/   ← 2594 raw masks
│   ├── ISIC2018_Task1-2_Validation_Input/     ← 100 raw images
│   ├── ISIC2018_Task1_Validation_GroundTruth/ ← 100 raw masks
│   ├── ISIC2018_Task1-2_Test_Input/           ← 1000 raw images
│   ├── ISIC2018_Task1_Test_GroundTruth/       ← 1000 raw masks
│   └── processed/                             ← train/val/test splits (after prepare_data.py)
├── outputs/                         ← Training outputs (checkpoints, curves, results)
├── notebooks/                       ← Jupyter notebooks for EDA/experiments
├── scripts/
│   ├── prepare_data.py              ← Copy official ISIC 2018 split (2594/100/1000)
│   ├── train.py                     ← Main training entry point
│   ├── evaluate.py                  ← Test evaluation with TTA + threshold search
│   └── predict.py                   ← Inference + overlay visualization
└── src/                             ← Installable Python package
    ├── data/                        ← ISICDataset, get_transforms  ⚠ MISSING — must create
    ├── losses/                      ← FocalLoss, SoftDiceLoss, CombinedLoss
    ├── metrics/                     ← dice_coefficient, iou_score, find_best_threshold
    ├── models/                      ← Registry-based model factory (create_model)
    ├── training/                    ← Trainer, EarlyStopping, ModelCheckpoint
    └── utils/                       ← Config loader (Config, load_config), Logger, set_seed
```

> **BLOCKING GAP:** `src/data/dataset.py` (`ISICDataset`) and `src/data/transforms.py`
> (`get_transforms`) are imported by all scripts but **do not exist**. Create these before
> attempting to run any script.

---

## Commands Reference

```bash
# Environment setup
conda env create -f environment.yml && conda activate skin-cancer
pip install -e .                          # editable install (required once)

# Data preparation (run once)
# Requires 6 official ISIC 2018 folders under data/ (see Project Structure above)
python scripts/prepare_data.py
python scripts/prepare_data.py --data-dir data/ --out-dir data/processed

# Train
python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml

# Train with CLI overrides (no YAML editing needed)
python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml \
    training.batch_size=32 \
    model.encoder_name=efficientnet-b4 \
    output.dir=outputs/efficientnetb4_test

# Evaluate on test set (TTA + optimal threshold search)
python scripts/evaluate.py \
    --config configs/experiments/resnet34_unet_v1.yaml \
    --checkpoint outputs/resnet34_unet_v1/best_model.pth

# Predict with 3-panel overlay (original | mask | overlay)
python scripts/predict.py \
    --config configs/experiments/resnet34_unet_v1.yaml \
    --input data/processed/test/images/ISIC_0024306.jpg \
    --checkpoint outputs/resnet34_unet_v1/best_model.pth \
    --overlay --tta

# Kaggle-compatible run (no W&B)
python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml \
    data.root=/kaggle/input/isic-2018 \
    output.dir=/kaggle/working \
    logging.use_wandb=false
```

> **No tests, no linter config, no CI** exist yet. When adding, use `pytest` for tests and
> `ruff` for linting. Run a single test with:
> `pytest tests/test_metrics.py::test_dice_coefficient -v`

---

## Core Conventions & Rules

### 1. Model Registry (CRITICAL)
All architectures **MUST** be registered in `src/models/segmentation.py` via `_REGISTRY`.
The only currently registered key is `"unet"`. Never hardcode model classes elsewhere.
Use `create_model(config)` as the sole entry point. `config.model.name` must match a key.

```python
_REGISTRY = {
    "unet":              _build_unet,
    "efficientnet_unet": _build_efficientnet_unet,   # add new entries here
}
```

### 2. Experiment Config
Every experiment **MUST** have its own YAML in `configs/experiments/` inheriting `base.yaml`.
**Never** modify `base.yaml` for experiment-specific settings.

```yaml
_base_: ../base.yaml
model:
  name: unet
  encoder_name: efficientnet-b4
output:
  dir: outputs/efficientnetb4_unet_v1
logging:
  experiment_name: efficientnetb4_unet_v1
```

### 3. Metric Reporting (mandatory 4-tuple)
All evaluation results **MUST** report all four metrics — no exceptions:
- `test_dice` — at threshold 0.5
- `test_iou` — at threshold 0.5
- `test_dice_best` — at optimal threshold (grid search 0.30–0.70, step 0.05)
- `best_threshold`

### 4. Reproducibility
- `seed: 42` in every YAML; call `set_seed(config.seed)` at start of every script
- Save `model_config` alongside `state_dict` in every checkpoint
  - Current implementation stores `config.model.to_dict()` under checkpoint key `model_config`

### 5. Loss Function
Default: `CombinedLoss(0.5 × FocalLoss(γ=2) + 0.5 × SoftDiceLoss)`.
New losses → add **only** to `src/losses/segmentation.py`, wire via `training.loss` in config.

### 6. Data Augmentation
- Train augmentations live **only** in `src/data/transforms.py` → `get_transforms("train", config)`
- Val/test: **only** `Resize + Normalize + ToTensorV2` — zero augmentation
- ImageNet normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

---

## Code Style Guidelines

### Imports
Every Python file begins with `from __future__ import annotations`.
Scripts inject repo root into `sys.path` for direct execution:
```python
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
```
Import order: stdlib → third-party → project-local (PEP 8). No `isort` enforced yet.

### Type Hints
- Required on all public function signatures
- Use built-in generics: `list[str]`, `dict[str, Any]`, `tuple[float, float]`
- **Do not** use `from typing import Dict/List/Tuple` — use lowercase builtins (Python 3.9+)
- Existing violation in `src/utils/misc.py` (`from typing import Dict`) — fix when touching that file

### Naming Conventions
| Kind | Style | Examples |
|---|---|---|
| Classes | `PascalCase` | `ISICDataset`, `CombinedLoss`, `EarlyStopping` |
| Functions / methods | `snake_case` | `dice_coefficient`, `build_dataloaders` |
| Private helpers | `_leading_underscore` | `_build_unet`, `_deep_merge`, `_cast_value` |
| Module-level constants | `_SCREAMING_SNAKE` | `_REGISTRY`, `_REPO_ROOT`, `_SPLIT_DIRS` |
| Config keys | `snake_case` | `encoder_name`, `batch_size` |

### Docstrings
All public classes and functions require docstrings with `Args:` / `Returns:` sections.
Comments may be in English or Vietnamese — both are acceptable.

```python
def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Dice Coefficient (macro-averaged over batch).
    Args:
        pred:      Raw logits or probabilities (B, 1, H, W)
        target:    Binary ground truth (B, 1, H, W)
        threshold: Binarization cutoff
    Returns:
        float in [0, 1]
    """
```

### Error Handling
- **Scripts (missing path):** `log.error(...) + sys.exit(1)` — hard exit
- **Factory (unknown key):** `raise ValueError(f"... Available: {list(_REGISTRY.keys())}")`
- **Optional deps (W&B):** `try/except ImportError` → `logger.warning()` + graceful fallback
- **Data validation:** `assert` with message is acceptable outside hot paths
- **Checkpoint loading:** `.get("model_state_dict", ckpt)` to handle both formats
- **Multi-GPU:** `model.module if hasattr(model, "module") else model` before attribute access
- No bare `except:`, no silent swallowing of errors in hot paths

### PyTorch Patterns
```python
# AMP — PyTorch 2.x non-deprecated API only
with torch.amp.autocast(device_type="cuda", enabled=cfg.training.mixed_precision):
    ...
scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.mixed_precision)

# DataParallel-safe access
model_ref = model.module if hasattr(model, "module") else model
```

---

## Adding a New Architecture — Checklist

1. Add `_build_<name>(cfg) -> nn.Module` in `src/models/segmentation.py`
2. Register: `_REGISTRY["<name>"] = _build_<name>`
3. Create `configs/experiments/<name>_v1.yaml` (set `_base_: ../base.yaml`, `model.name: <name>`)
4. Train: `python scripts/train.py --config configs/experiments/<name>_v1.yaml`
5. Evaluate: `python scripts/evaluate.py --checkpoint outputs/<name>_v1/best_model.pth --config ...`

**Candidate architectures (ordered by expected impact):**
- `efficientnet-b4` / `efficientnet-b6` encoder (SMP, drop-in swap)
- `mit_b2` / `mit_b4` Mix Transformer (SegFormer backbone + UNet decoder)
- `DeepLabV3+` decoder with ResNet50 / EfficientNet encoder
- SAM (Segment Anything Model) fine-tuned on ISIC Task 1
- Swin-UNet / TransUNet for Transformer-based segmentation

---

## Web Product Target

- `scripts/predict.py --overlay` produces a 3-panel PNG (original | mask | overlay)
- Overlay path auto-resizes predicted mask to original image resolution before blending
- Target inference: **< 1s per image** at 256×256 on CPU
- Planned backend: **FastAPI** + model loading at startup
- ONNX / TorchScript export for lighter cloud deployment

---

## Available Skills

Skills in `.opencode/skills/` (OpenCode) and `.cursor/rules/` (Cursor):

| Skill | Purpose |
|---|---|
| `computer-vision-expert` | SOTA CV: SAM, SegFormer, VLMs, deployment optimization |
| `ml-engineer` | PyTorch 2.x production, SMP patterns, TTA, model serving |
| `mlops-engineer` | W&B tracking, experiment lifecycle, pipeline automation |
| `python-pro` | Python 3.11+ best practices, type hints, performance |
| `data-scientist` | Statistical analysis, EDA, ablation study, model comparison |
