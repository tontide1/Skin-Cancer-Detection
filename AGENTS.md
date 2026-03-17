# Skin-Cancer-Detection — AGENTS.md

## Project Overview

Binary **skin lesion segmentation** on the [ISIC 2018 Challenge – Task 1](https://challenge.isic-archive.com/landing/2018/) dataset.

**Current Best:** U-Net + ResNet34 + SCSE attention (SMP) — Test Dice **0.9021** | IoU **0.8368** (epoch 41)
**Registered models:** `"unet"`, `"deeplabv3"` (MobileNetV3-Large, torchvision)
**Goal:** Dice > 0.90, then deploy as FastAPI web app.

---

## Tech Stack

| Component | Details |
|---|---|
| Language | Python 3.11 (Conda env: `CV`) |
| DL Framework | PyTorch 2.x + AMP (`torch.amp`) |
| Segmentation | `segmentation-models-pytorch >= 0.3` (SMP) |
| Augmentation | `albumentations >= 1.3` |
| Experiment Tracking | W&B + local CSV/JSON fallback |
| Config System | YAML with `_base_` inheritance + CLI dot-notation overrides |
| Package | Editable install: `pip install -e .` (package root: `src/`) |

---

## Commands Reference

```bash
# Environment (Conda env is "CV", NOT "skin-cancer")
conda activate CV
pip install -e .

# Lint (check code style)
ruff check src/ scripts/ tests/
ruff check src/ scripts/ tests/ --fix  # auto-fix

# Data preparation (new dataset structure - HA10000 remove-hair)
# Input: data/data-HA10000-remove-hair/remove-hair/images + masks/
# Output: data/processed/train|val|test/images + masks/
python scripts/prepare_data.py \
    --data-dir data/data-HA10000-remove-hair \
    --out-dir data/processed

# Custom split ratios (default: 80/10/10)
python scripts/prepare_data.py \
    --data-dir data/data-HA10000-remove-hair \
    --out-dir data/processed \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 42

# Train
python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml
python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml \
    training.batch_size=32 model.encoder_name=efficientnet-b4 logging.use_wandb=false

# Evaluate (TTA + threshold search)
python scripts/evaluate.py \
    --config configs/experiments/resnet34_unet_v1.yaml \
    --checkpoint outputs/resnet34_unet_v1/best_model.pth

# Predict (3-panel overlay: original | mask | overlay)
python scripts/predict.py \
    --config configs/experiments/resnet34_unet_v1.yaml \
    --input data/processed/test/images/ISIC_0024306.jpg \
    --checkpoint outputs/resnet34_unet_v1/best_model.pth --overlay --tta
```

### Kaggle Training & Evaluation

```bash
# Train on Kaggle (2-GPU T4)
%cd /kaggle/working
!git clone https://github.com/tontiphan/Skin-Cancer-Detection.git
%cd Skin-Cancer-Detection
!git checkout stage1
!pip install -r requirements.txt -q
!torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train.py \
  --device-mode ddp \
  --config configs/experiments/resnet34_unet_kaggle_t4.yaml \
  data.root=/kaggle/input/datasets/tntiphan/isic-2018-task-1 \
  output.dir=/kaggle/working \
  logging.use_wandb=false

# Evaluate on Kaggle test set
!python scripts/evaluate.py \
  --config configs/experiments/resnet34_unet_kaggle_t4.yaml \
  --checkpoint /kaggle/working/resnet34_unet_kaggle_t4/best_model.pth \
  data.root=/kaggle/input/datasets/tntiphan/isic-2018-task-1 \
  output.dir=/kaggle/working \
  logging.use_wandb=false
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run a single test FILE
pytest tests/test_trainer_robustness.py -v
pytest tests/models/test_deeplabv3_factory.py -v

# Run a single test FUNCTION
pytest tests/test_trainer_robustness.py::test_warmup_preserves_differential_lr_ratio -v
pytest tests/models/test_deeplabv3_factory.py::test_deeplabv3_factory_uses_backbone_imagenet_weights -v

# Skip integration tests (require real torchvision/GPU)
pytest tests/ -v --ignore=tests/models/test_deeplabv3_integration.py
```

> Linter: `ruff` — `line-length = 100`, `target-version = "py311"` (configured in `pyproject.toml`).
> Integration tests auto-skip via `pytest.importorskip("torchvision")`.

---

## Core Conventions & Rules

### 1. Model Registry (CRITICAL)
All architectures **MUST** be registered in `src/models/segmentation.py` via `_REGISTRY`.
`create_model(config)` is the **sole entry point** — `config.model.name` must match a registry key.

### 2. Experiment Config
Every experiment **MUST** have its own YAML in `configs/experiments/` with `_base_: ../base.yaml`.
**Never** modify `base.yaml` for experiment-specific settings.
**Do NOT** set `output.dir` in experiment YAMLs — `train.py` builds `outputs/<experiment_name>/`
automatically. Setting both causes double-nesting (e.g. `outputs/my_exp/my_exp/`).

### 3. Metric Reporting (mandatory 5-key contract)
All evaluation results **MUST** contain all five keys — no exceptions:
- `{split}_dice` — Dice at threshold 0.5 (e.g. `test_dice`, `val_dice`)
- `{split}_iou` — IoU at threshold 0.5
- `{split}_dice_best` — Dice at optimal threshold (grid search 0.30–0.70, step 0.05)
- `{split}_iou_best` — IoU at that same optimal threshold
- `best_threshold` — the optimal threshold value

Generic aliases (`dice`, `iou`, `best_dice_at_best_thr`, `best_iou_at_best_thr`) are kept in the
return dict for backward compatibility but are **not** part of the required contract.

### 4. Reproducibility
- `seed: 42` in every YAML; call `set_seed(config.seed)` at the start of every script.
- Save `config.model.to_dict()` as `model_config` key in every checkpoint.

### 5. Loss Function
Default: `CombinedLoss(0.5 × FocalLoss(γ=2) + 0.5 × SoftDiceLoss)`.
New losses → add **only** to `src/losses/segmentation.py`, wire via `training.loss` in config.

### 6. Data Augmentation
- Train augmentations live **only** in `src/data/transforms.py:get_transforms("train", config)`.
- Val/test: `Resize + Normalize + ToTensorV2` only — **zero** augmentation.
- `get_transforms()` raises `ValueError` for any unrecognized split name (fail-fast).
- ImageNet normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

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
Import order: stdlib → third-party → project-local (PEP 8).

### Type Hints
Required on all public function signatures. Use built-in generics (`list[str]`, `dict[str, Any]`).
Do **not** use `from typing import Dict/List/Tuple` — use lowercase builtins (Python 3.9+).

### Naming Conventions
| Kind | Style | Examples |
|---|---|---|
| Classes | `PascalCase` | `ISICDataset`, `CombinedLoss`, `EarlyStopping` |
| Functions / methods | `snake_case` | `dice_coefficient`, `build_dataloaders` |
| Private helpers | `_leading_underscore` | `_build_unet`, `_deep_merge`, `_cast_value` |
| Module-level constants | `_SCREAMING_SNAKE` | `_REGISTRY`, `_REPO_ROOT`, `_VAL_METRIC_SEMANTICS` |
| Config keys | `snake_case` | `encoder_name`, `batch_size` |

### Docstrings
All public classes and functions require docstrings with `Args:` / `Returns:` sections.
Comments may be in English or Vietnamese — both are acceptable.

### Error Handling
- **Scripts (missing path):** `log.error(...) + sys.exit(1)`
- **Factory (unknown key):** `raise ValueError(f"... Available: {list(_REGISTRY.keys())}")`
- **Builder constraints:** `raise ValueError` with a clear message (e.g. unsupported `in_channels`)
- **Informational-only fields:** `logging.getLogger(__name__).warning(...)` — warn, do not raise
- **Optional deps (W&B):** `try/except ImportError` → `logger.warning()` + graceful fallback
- **Checkpoint loading:** `.get("model_state_dict", ckpt)` to handle both formats; use
  `load_state_dict_with_aux_compat()` from `src/utils/checkpoint.py` for DeepLabV3 checkpoints
- **Multi-GPU:** `model.module if hasattr(model, "module") else model` before attribute access
- No bare `except:`, no silent swallowing in hot paths

### PyTorch Patterns
```python
# AMP — PyTorch 2.x non-deprecated API only
with torch.amp.autocast(device_type="cuda", enabled=cfg.training.mixed_precision): ...
scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.mixed_precision)

# DataParallel-safe model access
model_ref = model.module if hasattr(model, "module") else model
```

---

## Adding a New Architecture — Checklist

1. Add `_build_<name>(cfg) -> nn.Module` in `src/models/segmentation.py`
2. Register: `_REGISTRY["<name>"] = _build_<name>`
3. If using torchvision (not SMP): wrap with a dict-unwrapping `nn.Module` (see `DeepLabV3Wrapper`)
4. Raise `ValueError` for unsupported config fields at build time; warn for informational-only fields
5. Create `configs/experiments/<name>_v1.yaml` (`_base_: ../base.yaml`, `model.name: <name>`)
6. Add unit tests in `tests/models/` (mock heavy deps with `monkeypatch`; skip with `importorskip`)
7. Train → Evaluate → report all 5 metric keys

**Candidate architectures:** `efficientnet-b4/b6` (SMP, drop-in), `mit_b2/b4` (SegFormer),
`DeepLabV3+` (ResNet50/EfficientNet), SAM fine-tuned on ISIC, Swin-UNet / TransUNet.

---

## DeepLabV3 — Known Constraints

- `in_channels` **must** be 3 — torchvision hard-codes RGB; builder raises `ValueError` otherwise
- `encoder_name` is informational only — builder always uses MobileNetV3-Large; wrong value → warning
- `aux_loss=False` hard-coded; use `load_state_dict_with_aux_compat()` for legacy checkpoints
- Minimum input size: **≥ 128×128** (ASPP BatchNorm crashes smaller) — default `[256, 256]` is safe

---

## Checkpoint Versioning

`_VAL_METRIC_SEMANTICS = "macro_per_sample_v1"` is saved in both `best_model.pth` and
`last_checkpoint.pth`. On resume, a mismatch (e.g. old micro-average checkpoint) triggers an
automatic reset of best-metric tracking with a logged warning. Do not change this sentinel string
without bumping to `v2`.
