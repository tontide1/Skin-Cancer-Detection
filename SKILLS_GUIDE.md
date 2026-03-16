# Skin-Cancer-Detection — Skills Usage Guide

> **Hướng dẫn sử dụng Agentic Skills tối ưu cho project.**
> File này là companion của `AGENTS.md`.
> `AGENTS.md` = source of truth cho project rules.
> `SKILLS_GUIDE.md` = hướng dẫn dùng AI skills hiệu quả nhất.

---

## Table of Contents

0. [Current Source Snapshot (Mar 2026)](#current-source-snapshot-mar-2026)
1. [Quick Reference Table](#1-quick-reference-table)
2. [Invocation Syntax](#2-invocation-syntax)
3. [Context Template — Truyền context đúng cách](#3-context-template)
4. [Task → Skill Mapping](#4-task--skill-mapping)
5. [Combo Workflows](#5-combo-workflows)
6. [Skills không phù hợp — Tránh lãng phí thời gian](#6-skills-không-phù-hợp)
7. [Checklist theo giai đoạn phát triển](#7-checklist-theo-giai-đoạn-phát-triển)
8. [Tips & Anti-Patterns](#8-tips--anti-patterns)

---

## Current Source Snapshot (Mar 2026)

- `scripts/predict.py --overlay` đã xử lý mismatch shape: predicted mask được resize về kích thước ảnh gốc bằng `nearest-neighbor` trước khi blend.
- `src/training/callbacks.py::ModelCheckpoint.step()` đã lưu `model_config` trong checkpoint payload.
- `src/training/trainer.py` hiện truyền `self.config.model.to_dict()` vào key `model_config` khi lưu best checkpoint.

---

## 1. Quick Reference Table

> Bảng tra nhanh 9 skills — đọc cột "When to Use in This Project" để chọn đúng ngay.

| Skill | Vai trò | When to Use in This Project | Tools |
| :--- | :--- | :--- | :--- |
| `computer-vision-expert` | SOTA CV: SAM 3, segmentation, VLMs | Model architecture, augmentation, segmentation pipeline | OpenCode · Codex · Antigravity |
| `ml-engineer` | PyTorch 2.x production, model serving | Training loop, AMP, DataLoader, inference | OpenCode · Codex · Antigravity |
| `ml-pipeline-workflow` | ML pipeline từ data → deployment | End-to-end ML workflow, feature engineering, model validation | OpenCode · Codex · Antigravity |
| `python-pro` | Python 3.11+, modern tooling | Type hints, docstrings, project setup | OpenCode · Codex · Antigravity |
| `python-testing-patterns` | Pytest patterns, test coverage | Test suite, unit tests, integration tests | OpenCode · Codex · Antigravity |
| `python-performance-optimization` | Performance profiling, optimization | GPU optimization, memory optimization, profiling | OpenCode · Codex · Antigravity |
| `systematic-debugging` | Systematic debugging, root cause | NaN loss, GPU OOM, training issues | OpenCode · Codex · Antigravity |
| `error-debugging-error-analysis` | Error analysis, diagnostics | Error trace analysis, exception handling | OpenCode · Codex · Antigravity |
| `debugging-toolkit-smart-debug` | Smart debugging tools | Interactive debugging, breakpoint strategies | OpenCode · Codex · Antigravity |

---

## 2. Invocation Syntax

> Cú pháp gọi skill chính xác cho từng tool. Dùng đúng cú pháp để skill được load.

### OpenCode

```bash
# Trong OpenCode chat session:
"Use @computer-vision-expert to implement ISICDataset in src/data/dataset.py"
"Use @ml-engineer to optimize the DataLoader in scripts/train.py"
"Use @systematic-debugging to diagnose why val_dice is not improving after epoch 20"
```

### Codex CLI

```bash
# Trong Codex CLI:
"Use computer-vision-expert to implement ISICDataset in src/data/dataset.py"
"Use ml-engineer to optimize the DataLoader in scripts/train.py"
"Use systematic-debugging to diagnose why val_dice is not improving after epoch 20"
```

### Cursor

```
# Trong Cursor Chat (CMD+L hoặc CTRL+L), tag skill ở đầu prompt:
@computer-vision-expert implement get_transforms() with Albumentations for ISIC 2018
@ml-engineer add EfficientNet-B4 encoder to the model registry in src/models/segmentation.py
@python-performance-optimization create an ablation study comparison table for resnet34 vs efficientnet-b4
```

### Antigravity (Agent Mode)

```
# Trong Antigravity Agent Mode:
"Use computer-vision-expert skill to design a SegFormer-based architecture for binary segmentation"
"Use ml-pipeline-workflow skill to set up W&B sweeps for hyperparameter search"
"Use systematic-debugging skill to diagnose training issues"
```

---

## 3. Context Template

> **Quy tắc vàng:** Càng nhiều context cụ thể → AI càng ít hallucinate, code càng đúng ngay lần đầu.
>
> Luôn đính kèm 4 thành phần: **stack**, **relevant files**, **constraints**, **expected output**.

### Template chuẩn cho project này

```
Use @<skill-name> to <task description>.

Stack: Python 3.11, PyTorch 2.x with torch.amp (non-deprecated API),
segmentation-models-pytorch >= 0.3, Albumentations >= 1.3.

Relevant files: <list file paths>

Constraints:
- Follow _REGISTRY pattern in src/models/segmentation.py
- Use torch.amp.autocast(device_type="cuda") — NOT torch.cuda.amp.autocast (deprecated)
- All evaluation must report 4 metrics: dice@0.5, iou@0.5, dice@best_threshold, best_threshold
- Seed: 42, set via set_seed() from src/utils/misc.py
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Type hints required on all public function signatures (built-in generics: list[], dict[], tuple[])
- Docstrings with Args:/Returns: sections on all public classes and functions

Expected output: <mô tả output mong muốn>
```

### Ví dụ thực tế — tạo ISICDataset

```
Use @computer-vision-expert to implement src/data/dataset.py.

Stack: Python 3.11, PyTorch 2.x, Albumentations >= 1.3, PIL.

Relevant files:
- scripts/train.py (shows how ISICDataset is called: img_dir, mask_dir, transform)
- scripts/evaluate.py (shows how dataset is used for eval)
- configs/base.yaml (data.input_size = [256, 256])

Constraints:
- Class name: ISICDataset(img_dir: Path, mask_dir: Path, transform)
- Images: .jpg files, Masks: _segmentation.png files (ISIC naming convention)
- Return: (image_tensor: FloatTensor C×H×W, mask_tensor: FloatTensor 1×H×W) in range [0,1]
- Must handle missing mask gracefully with assertion + error message
- Type hints on all public methods, docstring with Args:/Returns:

Expected output: Complete src/data/dataset.py ready to import.
```

---

## 4. Task → Skill Mapping

> Map từng nhiệm vụ cụ thể sang skill phù hợp + prompt mẫu copy-paste.
> Sắp xếp theo thứ tự ưu tiên (blocking → research → quality → production).

---

### 4.1 BLOCKING — Phải làm trước khi chạy bất kỳ script nào

#### Task 1 — Tạo `src/data/dataset.py` (class `ISICDataset`)

- **Skill chính:** `computer-vision-expert`
- **Skill hỗ trợ:** `python-pro`
- **Tại sao:** `ISICDataset` là PyTorch Dataset cho ảnh dermoscopy — domain knowledge CV + Python packaging

**Prompt mẫu:**

```
Use @computer-vision-expert to implement src/data/dataset.py containing class ISICDataset.

Stack: Python 3.11, PyTorch 2.x, PIL, Albumentations >= 1.3.

Relevant files:
- scripts/train.py lines 40-55 (how ISICDataset is instantiated)
- scripts/evaluate.py lines 70-78 (eval usage)
- configs/base.yaml (data.input_size: [256, 256])

Constraints:
- Signature: ISICDataset(img_dir: Path, mask_dir: Path, transform=None)
- Images: *.jpg, Masks: *_segmentation.png (match by stem)
- __getitem__ returns (FloatTensor C×H×W, FloatTensor 1×H×W) both normalized to [0,1]
- transform receives dict with keys "image" (H×W×C uint8) and "mask" (H×W float32)
- __len__ returns number of image files
- Assert img_dir and mask_dir exist, raise FileNotFoundError with clear message
- from __future__ import annotations at top
- Full docstring with Args:/Returns:

Expected output: Complete, importable src/data/dataset.py.
```

---

#### Task 2 — Tạo `src/data/transforms.py` (function `get_transforms`)

- **Skill chính:** `computer-vision-expert`
- **Skill hỗ trợ:** `python-pro`
- **Tại sao:** Albumentations pipeline cho medical imaging — CV expertise cần thiết để chọn đúng augmentation cho dermoscopy

**Prompt mẫu:**

```
Use @computer-vision-expert to implement src/data/transforms.py containing get_transforms().

Stack: Python 3.11, Albumentations >= 1.3, PyTorch 2.x.

Relevant files:
- AGENTS.md section "Data Augmentation" (augmentation rules)
- configs/base.yaml (data.input_size: [256, 256])

Constraints:
- Signature: get_transforms(split: str, config) -> albumentations.Compose
- split must be "train" | "val" | "test"
- Train: meaningful augmentations suitable for dermoscopy (skin lesion images)
  Recommended: RandomRotate90, Flip, ColorJitter, ElasticTransform, GridDistortion
  Do NOT use augmentations that destroy lesion boundaries (e.g., extreme crops)
- Val/Test: ONLY Resize(H, W) + Normalize(ImageNet) + ToTensorV2 — zero augmentation
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Raise ValueError for unknown split
- from __future__ import annotations at top

Expected output: Complete src/data/transforms.py, importable by all scripts.
```

---

### 4.2 Architecture Research — Thêm model mới

#### Task 3 — Thêm EfficientNet-B4 / B6 encoder

- **Skill chính:** `computer-vision-expert`
- **Skill hỗ trợ:** `ml-engineer`

**Prompt mẫu:**

```
Use @computer-vision-expert to add EfficientNet-B4 encoder to the model registry.

Relevant files:
- src/models/segmentation.py (existing _build_unet and _REGISTRY pattern)
- configs/experiments/resnet34_unet_v1.yaml (example experiment config)
- AGENTS.md section "Adding a New Architecture — Checklist"

Constraints:
- Add _build_efficientnet_unet(config) -> nn.Module using segmentation_models_pytorch
- Register as "efficientnet_unet" in _REGISTRY
- Create configs/experiments/efficientnetb4_unet_v1.yaml inheriting _base_: ../base.yaml
  with model.name: efficientnet_unet, model.encoder_name: efficientnet-b4
- Keep decoder_channels and decoder_attention_type: scse from base config
- No changes to base.yaml

Expected output:
1. Updated src/models/segmentation.py with new builder + registry entry
2. New configs/experiments/efficientnetb4_unet_v1.yaml
```

---

#### Task 4 — Thêm SegFormer (`mit_b2` / `mit_b4`) encoder

- **Skill chính:** `computer-vision-expert`
- **Skill hỗ trợ:** `ml-engineer`

**Prompt mẫu:**

```
Use @computer-vision-expert to add SegFormer (Mix Transformer) encoder to the model registry.

Relevant files:
- src/models/segmentation.py (_REGISTRY pattern)
- AGENTS.md "Candidate architectures" section

Constraints:
- Use segmentation_models_pytorch with encoder_name "mit_b2" (SMP supports Mix Transformer)
- Register as "segformer_unet" in _REGISTRY
- Create configs/experiments/segformer_unet_v1.yaml
  model.name: segformer_unet, model.encoder_name: mit_b2, encoder_weights: imagenet
- Note: mit_b* encoders require encoder_weights="imagenet" (not None)

Expected output:
1. Updated src/models/segmentation.py
2. New configs/experiments/segformer_unet_v1.yaml
```

---

#### Task 5 — Thêm DeepLabV3+ decoder

- **Skill chính:** `computer-vision-expert`
- **Skill hỗ trợ:** `ml-engineer`

**Prompt mẫu:**

```
Use @computer-vision-expert to add DeepLabV3+ decoder to the model registry.

Relevant files:
- src/models/segmentation.py (_REGISTRY pattern)
- AGENTS.md "Candidate architectures" section

Constraints:
- Use smp.DeepLabV3Plus with ResNet50 encoder
- Register as "deeplabv3plus" in _REGISTRY
- DeepLabV3+ has different decoder architecture — no decoder_channels or decoder_attention_type
  Handle this cleanly: builder should only pass relevant params to smp.DeepLabV3Plus
- Create configs/experiments/deeplabv3plus_resnet50_v1.yaml

Expected output:
1. Updated src/models/segmentation.py (builder handles param differences cleanly)
2. New configs/experiments/deeplabv3plus_resnet50_v1.yaml
```

---

#### Task 6 — Fine-tune SAM trên ISIC Task 1

- **Skill chính:** `computer-vision-expert`
- **Skill hỗ trợ:** `ml-engineer`

**Prompt mẫu:**

```
Use @computer-vision-expert to design a SAM fine-tuning strategy for ISIC 2018 Task 1 binary segmentation.

Context: Binary mask segmentation (single lesion per image), 2594 training images 256×256.
Current best: UNet + ResNet34 + SCSE, Test Dice = 0.9021.

Constraints:
- Must integrate with existing _REGISTRY pattern in src/models/segmentation.py
- Must be trainable with existing Trainer in src/training/trainer.py (forward pass returns logits)
- Prefer SAM 2 / MedSAM adapter approach over full fine-tuning (VRAM constraints)
- Output must be (B, 1, H, W) logits compatible with CombinedLoss

Expected output:
1. Strategy recommendation with VRAM estimate
2. _build_sam() function for src/models/segmentation.py
3. Any required changes to configs/base.yaml
```

---

### 4.3 Training Optimization — Tối ưu pipeline

#### Task 7 — Tối ưu DataLoader

- **Skill chính:** `python-performance-optimization`
- **Skill hỗ trợ:** `ml-engineer`

**Prompt mẫu:**

```
Use @python-performance-optimization to optimize DataLoader configuration for ISIC 2018 training.

Relevant files:
- scripts/train.py build_dataloaders() function
- configs/base.yaml (data.num_workers: 2, pin_memory: true, persistent_workers: true)

Context: 2594 training images, batch_size=64, 256×256 RGB → 1-channel mask.
Goal: Minimize DataLoader bottleneck, maximize GPU utilization.

Constraints:
- Remain compatible with Kaggle (num_workers must be configurable via config)
- Use PyTorch 2.x DataLoader best practices
- Do not break persistent_workers logic (already guarded: "and config.data.num_workers > 0")

Expected output: Optimized build_dataloaders() + updated base.yaml data section with rationale.
```

---

#### Task 8 — Tối ưu AMP / mixed precision

- **Skill chính:** `python-performance-optimization`
- **Skill hỗ trợ:** `ml-engineer`, `systematic-debugging`

**Prompt mẫu:**

```
Use @ml-engineer to review and optimize mixed precision training in the Trainer.

Relevant files:
- src/training/trainer.py (full file)
- configs/base.yaml (training.mixed_precision: true, training.grad_clip: null)

Constraints:
- Use ONLY non-deprecated PyTorch 2.x API:
  torch.amp.autocast(device_type="cuda") — NOT torch.cuda.amp.autocast
  torch.amp.GradScaler("cuda") — NOT torch.cuda.amp.GradScaler
- Check for potential loss scaling issues with CombinedLoss (FocalLoss + SoftDiceLoss)
- Recommend grad_clip value if beneficial

Expected output: Reviewed/optimized train_one_epoch() with inline comments explaining each AMP step.
```

---

#### Task 9 — Multi-GPU DDP training

- **Skill chính:** `ml-engineer`
- **Skill hỗ trợ:** `mlops-engineer`

**Prompt mẫu:**

```
Use @ml-engineer to add PyTorch DDP (DistributedDataParallel) support to the training pipeline.

Relevant files:
- scripts/train.py (main() function)
- src/training/trainer.py (Trainer class)
- AGENTS.md PyTorch Patterns section (model.module unwrapping pattern)

Constraints:
- Single-GPU path must remain unchanged (backward compatible)
- Use torchrun launcher, not deprecated torch.distributed.launch
- AGENTS.md requires: model_ref = model.module if hasattr(model, "module") else model
- Checkpoint must save model_ref.state_dict() (unwrapped), not DDP-wrapped state
- Logger (W&B) should only log on rank 0

Expected output:
1. Updated scripts/train.py with DDP support
2. Updated src/training/trainer.py with rank-aware logging + checkpoint
3. Launch command example in a code comment
```

---

### 4.4 Experiment Tracking & Analysis

#### Task 10 — W&B Sweeps (Hyperparameter Search)

- **Skill chính:** `ml-pipeline-workflow`
- **Skill hỗ trợ:** `python-pro`

**Prompt mẫu:**

```
Use @ml-pipeline-workflow to set up W&B sweeps for hyperparameter search on this project.

Relevant files:
- configs/base.yaml (all hyperparameters)
- scripts/train.py (entry point)
- src/utils/logger.py (existing W&B logger)

Search space to explore:
- model.encoder_name: [resnet34, efficientnet-b4, mit_b2]
- training.lr: [1e-4, 2e-4, 5e-4]
- training.loss.focal_gamma: [1.5, 2.0, 2.5]
- training.loss.focal_weight: [0.3, 0.5, 0.7]

Constraints:
- Use Bayesian optimization strategy (not grid search)
- Must be compatible with existing Logger class in src/utils/logger.py
- Sweep config as a separate YAML: configs/sweeps/architecture_lr_sweep.yaml
- Max 30 runs, early terminate if val_dice < 0.85 after epoch 20

Expected output:
1. configs/sweeps/architecture_lr_sweep.yaml
2. scripts/run_sweep.py (sweep agent launcher)
3. Instructions to launch: wandb sweep + wandb agent
```

---

#### Task 11 — Ablation Study so sánh architectures

- **Skill chính:** `python-performance-optimization`
- **Skill hỗ trợ:** `ml-engineer`

**Prompt mẫu:**

```
Use @python-performance-optimization to create an ablation study analysis for comparing segmentation architectures.

Context: We have (or will have) evaluation results for:
- resnet34_unet (baseline): Test Dice=0.9021, IoU=0.8368
- efficientnetb4_unet (TBD)
- segformer_unet (TBD)
- deeplabv3plus (TBD)

Each result is a JSON file at outputs/<experiment_name>/eval_test_results.json with fields:
dice, iou, best_threshold, best_dice_at_best_thr, best_iou_at_best_thr

Constraints:
- Script: scripts/ablation_study.py
- Accept list of output directories as CLI arguments
- Output: console table + matplotlib bar chart saved to outputs/ablation_study.png
- Include: model name, #params (from checkpoint model_config), dice@0.5, iou@0.5, dice@best_thr
- Sort by dice@best_thr descending

Expected output: Complete scripts/ablation_study.py
```

---

#### Task 12 — Phân tích training curves + diagnose overfitting

- **Skill chính:** `error-debugging-error-analysis`
- **Skill hỗ trợ:** `systematic-debugging`

**Prompt mẫu:**

```
Use @error-debugging-error-analysis to analyze training curves and diagnose potential issues.

Relevant files:
- outputs/<experiment_name>/training_history.json (fields: epoch, train_loss, val_loss, train_dice, val_dice, lr)
- src/utils/misc.py plot_curves() function

Context: Current best stops at epoch 41 with val_dice=0.8972.
Questions to answer:
1. Is the model overfitting, underfitting, or well-fitted?
2. Is the learning rate schedule (ReduceLROnPlateau, patience=5) triggering too early/late?
3. Would increasing early_stopping.patience from 10 benefit?

Expected output:
1. Updated plot_curves() that plots train vs val divergence clearly
2. Diagnosis comments based on typical training curve patterns
3. 2-3 concrete config changes to try (with rationale)
```

---

### 4.5 Debugging

#### Task 13 — Debug NaN loss / GPU OOM

- **Skill chính:** `systematic-debugging`
- **Skill hỗ trợ:** `ml-engineer`, `python-performance-optimization`

**Prompt mẫu:**

```
Use @systematic-debugging to systematically debug [NaN loss / GPU OOM] in the training pipeline.

Relevant files:
- src/training/trainer.py train_one_epoch()
- src/losses/segmentation.py CombinedLoss, FocalLoss, SoftDiceLoss
- configs/base.yaml (training section)

Observed symptom: [describe exactly — e.g., "loss becomes NaN at epoch 3, batch 47"]

Constraints:
- Do not modify model architecture
- Fixes must be backward compatible with Kaggle (no extra dependencies)
- If adding gradient checks, guard with if cfg.debug_mode: to avoid production overhead

Expected output:
1. Root cause hypothesis
2. Step-by-step debugging checklist (what to print/check and where)
3. Concrete fix with explanation
```

---

#### Task 14 — Debug: Val Dice không tăng

- **Skill chính:** `systematic-debugging`
- **Skill hỗ trợ:** `error-debugging-error-analysis`, `debugging-toolkit-smart-debug`

**Prompt mẫu:**

```
Use @systematic-debugging to diagnose why validation Dice is plateauing / not improving.

Relevant files:
- src/training/trainer.py validate()
- src/data/dataset.py ISICDataset
- configs/base.yaml (training section)

Context: Training runs for 50 epochs but val_dice stays around 0.85-0.87 while train_dice reaches 0.95+.
Questions to explore:
1. Is data leakage happening (val set too easy)?
2. Is augmentation too aggressive on training?
3. Is the learning rate too high/low for the later epochs?

Expected output:
1. Diagnostic checklist specific to validation plateau
2. 3-5 hypotheses ranked by likelihood
3. Suggested experiments to differentiate between hypotheses
```

**Additional Tools for Debugging:**

- **Error Analysis:** Use `error-debugging-error-analysis` for deep error trace analysis and exception handling patterns
- **Smart Debugging:** Use `debugging-toolkit-smart-debug` for interactive debugging strategies and breakpoint placement

```

---

### 4.6 Code Quality

#### Task 15 — Tạo test suite pytest cho `src/`

- **Skill chính:** `python-testing-patterns`
- **Skill hỗ trợ:** `python-pro`

**Prompt mẫu:**

```
Use @python-testing-patterns to create a pytest test suite for the src/ package.

Relevant files:
- src/losses/segmentation.py (FocalLoss, SoftDiceLoss, CombinedLoss)
- src/metrics/segmentation.py (dice_coefficient, iou_score)
- src/models/segmentation.py (create_model, _REGISTRY)
- src/utils/config.py (load_config, override_config)
- AGENTS.md ("No tests exist yet. When adding, use pytest")

Constraints:
- Directory: tests/ at repo root
- files: tests/test_losses.py, tests/test_metrics.py, tests/test_models.py, tests/test_config.py
- Use pytest fixtures, not unittest classes
- Tests must run WITHOUT GPU (use device="cpu", small tensors B=2, H=32, W=32)
- Mock W&B in any test that touches Logger (use monkeypatch or unittest.mock)
- Run single test: pytest tests/test_metrics.py::test_dice_coefficient -v
- Minimum coverage targets: losses 90%, metrics 95%, config 80%

Expected output: Complete tests/ directory with all 4 test files.
```

---

#### Task 16 — Setup `ruff` linter + `pyproject.toml`

- **Skill chính:** `python-patterns`
- **Skill hỗ trợ:** `python-pro`

**Prompt mẫu:**

```
Use @python-patterns to set up ruff linter and pyproject.toml for this project.

Relevant files:
- setup.py (existing editable install config — to be migrated)
- requirements.txt
- AGENTS.md "Code Style Guidelines" section (import order, type hints rules)

Constraints:
- Create pyproject.toml (migrate from setup.py, keep editable install working)
- ruff config: line-length=100, target Python 3.11
- Enable rules: E, W, F, I (isort), UP (pyupgrade), B (bugbear)
- Ignore: E501 (line too long — handled by formatter), B008
- Exclude: data/, outputs/, notebooks/
- Add ruff format config (replace black)
- Add pre-commit config: .pre-commit-config.yaml with ruff + ruff-format hooks

Expected output:
1. pyproject.toml (complete — replaces setup.py)
2. .pre-commit-config.yaml
3. Command to run: ruff check src/ scripts/ && ruff format src/ scripts/
```

---

#### Task 17 — Type hints + docstrings audit cho `src/`

- **Skill chính:** `python-pro`
- **Skill hỗ trợ:** `python-patterns`

**Prompt mẫu:**

```
Use @python-pro to audit and complete type hints and docstrings in src/.

Relevant files: entire src/ directory

Constraints:
- Use built-in generics only: list[], dict[], tuple[] — NOT typing.List/Dict/Tuple
- Fix existing violation in src/utils/misc.py (uses from typing import Dict)
- Return type annotations required on all public functions
- Docstrings format: summary line + Args: + Returns: sections (as per AGENTS.md)
- from __future__ import annotations must be first line in every file

Expected output: List of files needing changes + the corrected versions.
```

---

### 4.7 Production & Deployment

#### Task 18 — Dockerize training pipeline

- **Skill chính:** `docker-expert`
- **Skill hỗ trợ:** `mlops-engineer`

**Prompt mẫu:**

```
Use @docker-expert to write a multi-stage Dockerfile for the training pipeline.

Relevant files:
- environment.yml (conda deps: pytorch 2.x, cuda 12.1)
- requirements.txt (pip deps for Kaggle)
- setup.py (editable install)

Constraints:
- Base image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
- Multi-stage: builder stage (install deps) → runtime stage (copy only needed artifacts)
- Non-root user with UID/GID 1000
- Mount data/ and outputs/ as volumes (not COPY into image)
- WORKDIR: /workspace
- Entrypoint: python scripts/train.py
- CMD: --config configs/experiments/resnet34_unet_v1.yaml
- Include .dockerignore excluding: data/, outputs/, .git/, notebooks/, __pycache__/

Expected output:
1. Dockerfile
2. .dockerignore
3. docker run command example with volume mounts
```

---

#### Task 19 — FastAPI inference container

- **Skill chính:** `docker-expert`
- **Skill hỗ trợ:** `architecture`

**Prompt mẫu:**

```
Use @docker-expert to write a production-ready Dockerfile for FastAPI inference service.

Context: Planned FastAPI app will load best_model.pth at startup and serve predictions
via POST /predict (accepts image file, returns binary mask PNG).
Target: < 1s inference on CPU at 256×256.

Constraints:
- CPU-only image (no CUDA) — use pytorch/pytorch:2.1.0-cpu or python:3.11-slim + torch CPU wheel
- Multi-stage build: minimize final image size
- Non-root user
- HEALTHCHECK: GET /health endpoint
- Expose port 8000
- Model path configurable via MODEL_PATH env variable

Expected output:
1. Dockerfile.inference (separate from training Dockerfile)
2. docker-compose.yml with inference service + volume for models
3. Size estimate of final image
```

---

#### Task 20 — Thiết kế FastAPI web app

- **Skill chính:** `architecture`
- **Skill hỗ trợ:** `ml-engineer`

**Prompt mẫu:**

```
Use @architecture to design the FastAPI web application for skin lesion segmentation inference.

Context:
- scripts/predict.py already implements single-image inference logic (reuse it)
- overlay path in predict already resizes predicted mask to original image size safely
- Target: < 1s per image on CPU at 256×256
- Output: 3-panel PNG (original | mask | overlay) OR raw binary mask
- AGENTS.md: "Planned backend: FastAPI + model loading at startup"

Requirements:
- POST /predict: accepts image upload, returns segmentation result
- GET /health: liveness check with model status
- Model loaded once at startup (not per-request)
- Optional ONNX/TorchScript backend for faster CPU inference
- No database needed (stateless inference service)

Constraints:
- Use FastAPI with Pydantic v2
- Follow python-patterns skill: FastAPI structure by layer (routes/, services/, schemas/)
- Must integrate with existing src/ package (reuse ISICDataset transforms, model factory)
- Error handling: return 422 for invalid image format, 503 if model not loaded

Expected output:
1. Architecture diagram (ASCII or Mermaid)
2. Directory structure for app/
3. Key interfaces (endpoint signatures, service contracts)
4. ADR: ONNX vs TorchScript vs raw PyTorch for CPU inference
```

---

#### Task 21 — ONNX / TorchScript export

- **Skill chính:** `ml-engineer`
- **Skill hỗ trợ:** `docker-expert`

**Prompt mẫu:**

```
Use @ml-engineer to implement ONNX and TorchScript export for the trained model.

Relevant files:
- src/models/segmentation.py (create_model)
- scripts/evaluate.py (model loading pattern)
- AGENTS.md "Web Product Target" section (target: < 1s on CPU at 256×256)

Constraints:
- Script: scripts/export_model.py
- Support both formats: --format onnx | torchscript
- Input shape: (1, 3, 256, 256) float32
- Output shape: (1, 1, 256, 256) float32 (raw logits)
- ONNX: opset 17, dynamic batch axis, verify with onnxruntime
- TorchScript: torch.jit.trace (not script — SMP models have control flow issues)
- Benchmark: report inference time on CPU (10 warmup + 100 runs, report mean ± std)

Expected output:
1. scripts/export_model.py
2. Benchmark results format
3. Recommended format for production with rationale
```

---

## 5. Combo Workflows

> Khi một task lớn cần nhiều skills phối hợp. Thực hiện theo thứ tự — output của step trước là input của step sau.

---

### Workflow A — Thêm một architecture mới từ đầu đến cuối

> **Mục tiêu:** Implement, train, evaluate và so sánh một architecture mới với baseline ResNet34-UNet.

```
Step 1 — Design & Implement  → @computer-vision-expert
  Task: Implement builder function + register in _REGISTRY + create experiment YAML
  Prompt: Task 3/4/5 tùy architecture chọn (xem Section 4.2)

Step 2 — Verify Training Loop  → @ml-engineer
  Task: Ensure new architecture integrates cleanly with Trainer (AMP, grad clipping, checkpoint)
  Prompt: "Use @ml-engineer to verify [architecture_name] is compatible with Trainer in
           src/training/trainer.py. Check: output shape (B,1,H,W), AMP compatibility,
           parameter count with count_parameters()."

Step 3 — Experiment Config  → @ml-pipeline-workflow
  Task: Create W&B experiment config, run naming convention
  Prompt: "Use @ml-pipeline-workflow to set up W&B config for experiment [name].
           Relevant files: src/utils/logger.py, configs/experiments/[new_config].yaml.
           Ensure experiment_name, tags, and config logging are correct."

Step 4 — Evaluate & Compare  → @python-performance-optimization
  Task: Run ablation_study.py, create comparison report
  Prompt: "Use @python-performance-optimization to compare [new_architecture] vs resnet34_unet baseline.
           Input: outputs/[new_exp]/eval_test_results.json vs outputs/resnet34_unet_v1/eval_test_results.json.
           Report: dice@0.5, iou@0.5, dice@best_thr, inference time, #params. Include statistical note."
```

---

### Workflow B — Debug training bị stuck / không hội tụ

> **Mục tiêu:** Xác định và fix nguyên nhân model không học được.

```
Step 1 — Identify Problem Type  → @systematic-debugging
  Task: Phân loại vấn đề (data bug vs loss bug vs optimizer bug vs architecture bug)
  Prompt: Task 13 hoặc Task 14 tùy triệu chứng (xem Section 4.5)

Step 2 — Technical Fix  → @ml-engineer
  Task: Implement fix cụ thể (AMP issue, loss scale, LR adjustment)
  Prompt: "Use @ml-engineer to fix [specific issue identified in Step 1].
           Relevant files: [files from debugging output].
           Constraint: fix must not break Kaggle compatibility."

Step 3 — Validate Fix  → @python-performance-optimization
  Task: Xác nhận fix hoạt động bằng metrics analysis
  Prompt: "Use @python-performance-optimization to compare training curves before/after fix.
           Before: outputs/[old_run]/training_history.json
           After: outputs/[new_run]/training_history.json
           Confirm: loss decreasing smoothly, val_dice improving, no divergence."
```

**Additional Debugging Tools:**
- Use `error-debugging-error-analysis` for deep error trace analysis
- Use `debugging-toolkit-smart-debug` for interactive debugging strategies

---

### Workflow C — Chuẩn bị production deployment

> **Mục tiêu:** Từ trained model → production-ready FastAPI service trong Docker.

```
Step 1 — System Design  → @architecture
  Task: Thiết kế FastAPI app architecture
  Prompt: Task 20 (xem Section 4.7)

Step 2 — Model Optimization  → @ml-engineer
  Task: Export to ONNX/TorchScript, benchmark CPU inference
  Prompt: Task 21 (xem Section 4.7)

Step 3 — Containerization  → @docker-expert
  Task: Multi-stage Dockerfile cho inference service
  Prompt: Task 19 (xem Section 4.7)

Step 4 — MLOps Integration  → @mlops-engineer
  Task: Health checks, logging, model versioning cho production
  Prompt: "Use @mlops-engineer to add production monitoring to the FastAPI inference service.
           Requirements: structured JSON logging, /health endpoint with model version info,
           request latency tracking, Prometheus metrics endpoint /metrics.
           Relevant files: Dockerfile.inference, app/ directory from Step 1."
```

---

### Workflow D — Setup project từ đầu (onboarding)

> **Dành cho:** Người mới clone repo, cần chạy được training lần đầu tiên.

```
Step 1 — Environment Setup  → @python-pro
  Prompt: "Use @python-pro to verify the conda environment setup for this project.
           Relevant files: environment.yml, setup.py, requirements.txt.
           Check: conda env create, pip install -e ., import src works correctly."

Step 2 — Fix BLOCKING Gap  → @computer-vision-expert + @python-pro
  Prompt: Task 1 → Task 2 (Section 4.1) — create src/data/dataset.py then src/data/transforms.py

Step 3 — Verify Pipeline  → @ml-engineer
  Prompt: "Use @ml-engineer to write a smoke test for the full training pipeline.
           Create scripts/smoke_test.py that:
           - Loads config: configs/experiments/resnet34_unet_v1.yaml
           - Runs 2 training batches + 1 val batch
           - Verifies: loss is finite, dice in [0,1], checkpoint saves correctly
           - Completes in < 60 seconds on CPU"
```

---

## 6. Skills không phù hợp

> Tiết kiệm thời gian bằng cách biết skill nào KHÔNG dùng cho project này và tại sao.

| Skill | Lý do không phù hợp | Thay thế |
| :--- | :--- | :--- |
| `react-best-practices` | Project không có React frontend. FastAPI sẽ serve HTML thuần hoặc trả về ảnh/JSON. | Không cần |
| `rag-engineer` | Không có RAG pipeline, vector database, hay document retrieval trong project này. | Không cần |
| `python-patterns` | Skill này tập trung vào architecture decisions, framework selection. Dùng `python-pro` cho code quality. | `python-pro` |
| `angular` / `angular-*` | Không có Angular frontend. | Không cần |
| `nextjs-best-practices` | Không có Next.js. | Không cần |
| `docker-expert` | Không cần containerize trong giai đoạn hiện tại. Chỉ hữu ích khi cần deployment. | Không cần |
| `architecture` *(giai đoạn hiện tại)* | Chỉ hữu ích ở **Giai đoạn 4 (FastAPI)**. Đừng dùng cho ML research tasks hiện tại — sẽ cho advice về web architecture không liên quan. | `computer-vision-expert` cho phase hiện tại |
| `mlops-engineer` | Thay thế bởi `ml-pipeline-workflow` cho end-to-end ML workflow. | `ml-pipeline-workflow` |
| `data-scientist` | Thay thế bởi `python-performance-optimization` cho metrics analysis và `error-debugging-error-analysis` cho error analysis. | `python-performance-optimization`, `error-debugging-error-analysis` |
| `debugging-strategies` | Thay thế bởi 3 skills chuyên biệt: `systematic-debugging`, `error-debugging-error-analysis`, `debugging-toolkit-smart-debug` | `systematic-debugging` · `error-debugging-error-analysis` · `debugging-toolkit-smart-debug` |

---

## 7. Checklist theo giai đoạn phát triển

> Gắn skills vào roadmap thực tế của project. Tick từng item khi hoàn thành.

---

### Giai đoạn 1 — Fix Blocking Gap *(Ngay bây giờ)*

> Không thể chạy bất kỳ script nào cho đến khi hoàn thành giai đoạn này.

- [ ] `@computer-vision-expert` → Tạo `src/data/dataset.py` (ISICDataset) — Task 1
- [ ] `@computer-vision-expert` → Tạo `src/data/transforms.py` (get_transforms) — Task 2
- [ ] Verify: `python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml` chạy được (2 batches)

**Skills:** `computer-vision-expert` · `python-pro`

---

### Giai đoạn 2 — Architecture Research *(Sau khi baseline chạy được)*

> Mục tiêu: Beat Dice > 0.90 với architectures mới.

- [ ] `@computer-vision-expert` → Thêm EfficientNet-B4 encoder — Task 3
- [ ] `@computer-vision-expert` → Thêm SegFormer mit_b2 encoder — Task 4
- [ ] `@computer-vision-expert` → Thêm DeepLabV3+ decoder — Task 5
- [ ] `@ml-pipeline-workflow` → Setup W&B sweeps cho LR + loss weight search — Task 10
- [ ] `@python-performance-optimization` → Ablation study so sánh tất cả architectures — Task 11
- [ ] `@error-debugging-error-analysis` → Phân tích training curves, diagnose overfitting — Task 12

**Skills:** `computer-vision-expert` · `ml-engineer` · `ml-pipeline-workflow` · `python-performance-optimization` · `error-debugging-error-analysis`

---

### Giai đoạn 3 — Training Optimization *(Song song với Giai đoạn 2)*

> Mục tiêu: Maximize throughput, minimize training time.

- [ ] `@python-performance-optimization` → Tối ưu DataLoader (num_workers, prefetch) — Task 7
- [ ] `@python-performance-optimization` → Review AMP / GradScaler implementation — Task 8
- [ ] `@ml-engineer` → Multi-GPU DDP support (nếu cần) — Task 9

**Skills:** `python-performance-optimization` · `ml-engineer` · `systematic-debugging`

---

### Giai đoạn 4 — Code Quality *(Bất kỳ lúc nào)*

> Mục tiêu: Production-ready codebase, maintainable, testable.

- [ ] `@python-testing-patterns` → Tạo test suite pytest cho `src/` — Task 15
- [ ] `@python-pro` → Setup ruff + pyproject.toml — Task 16
- [ ] `@python-pro` → Audit type hints + docstrings cho `src/` — Task 17

**Skills:** `python-testing-patterns` · `python-pro`

---

### Giai đoạn 5 — Production / Web App *(Sau khi đạt Dice > 0.90)*

> Mục tiêu: Deploy model như FastAPI service trong Docker.

- [ ] `@architecture` → Thiết kế FastAPI app structure — Task 20
- [ ] `@ml-engineer` → ONNX / TorchScript export + benchmark — Task 21
- [ ] `@docker-expert` → Dockerfile cho training pipeline — Task 18
- [ ] `@docker-expert` → Dockerfile cho inference / FastAPI — Task 19
- [ ] `@ml-pipeline-workflow` → Production monitoring, health checks, structured logging

**Skills:** `architecture` · `ml-engineer` · `docker-expert` · `ml-pipeline-workflow`

---

## 8. Tips & Anti-Patterns

### Tips — Để skills hoạt động tốt nhất

**1. Luôn đề cập PyTorch version và AMP API**

```
# LUÔN nói rõ trong prompt:
"Use PyTorch 2.x non-deprecated AMP API:
 torch.amp.autocast(device_type='cuda') — NOT torch.cuda.amp.autocast"
```

Nếu không đề cập, AI có thể generate code dùng deprecated API cũ — sẽ gây DeprecationWarning.

---

**2. Luôn nhắc `_REGISTRY` pattern khi thêm model mới**

```
# LUÔN thêm vào prompt khi liên quan đến model:
"Follow _REGISTRY pattern in src/models/segmentation.py.
 Register as _REGISTRY['<name>'] = _build_<name>.
 Never hardcode model classes outside this file."
```

---

**3. Luôn nhắc 4-tuple metrics requirement khi liên quan đến evaluation**

```
# LUÔN thêm vào prompt liên quan đến evaluation:
"Must report all 4 metrics as per AGENTS.md:
 dice@0.5, iou@0.5, dice@best_threshold, best_threshold"
```

---

**4. Cung cấp file paths cụ thể thay vì mô tả chung chung**

```
# Kém hiệu quả:
"Use @ml-engineer to fix the training loop"

# Tốt hơn:
"Use @ml-engineer to fix train_one_epoch() in src/training/trainer.py lines 45-80.
 Specific issue: GradScaler is using deprecated torch.cuda.amp.GradScaler"
```

---

**5. Một skill cho một task tập trung**

```
# Kém hiệu quả — quá nhiều scope:
"Use @computer-vision-expert to implement dataset, transforms, add EfficientNet,
 and set up W&B"

# Tốt hơn — tách thành tasks riêng:
Session 1: "Use @computer-vision-expert to implement src/data/dataset.py"
Session 2: "Use @computer-vision-expert to implement src/data/transforms.py"
Session 3: "Use @computer-vision-expert to add EfficientNet-B4 to _REGISTRY"
```

---

**6. `AGENTS.md` là source of truth — luôn ưu tiên hơn skill output**

Khi skill đề xuất code/convention mâu thuẫn với `AGENTS.md`:
- **Ưu tiên `AGENTS.md`** — đó là project rules đã được thiết lập
- Nói rõ với AI: `"Follow the convention in AGENTS.md, not the default pattern you'd normally use"`

---

### Anti-Patterns — Những gì cần tránh

| Anti-Pattern | Vấn đề | Cách đúng |
| :--- | :--- | :--- |
| Dùng `@architecture` cho ML tasks | Skill này cho system design, sẽ cho advice về web patterns không liên quan | Dùng `@computer-vision-expert` hoặc `@ml-engineer` |
| Dùng `@data-scientist` để implement code | Skill này cho analysis/visualization, không phải engineering | Dùng `@ml-engineer` hoặc `@python-pro` để implement |
| Dùng `@mlops-engineer` để debug training | Skill này cho pipelines/infrastructure, không phải model debugging | Dùng `@debugging-strategies` trước |
| Gọi skill mà không cung cấp relevant files | AI không biết project conventions, sẽ generate generic code | Luôn list file paths trong prompt |
| Dùng skill để thay thế đọc `AGENTS.md` | Skills không biết project-specific rules | Đọc `AGENTS.md` trước, dùng skill để implement |
| Combine 2+ unrelated skills trong 1 prompt | AI bị conflicting context, output kém chất lượng | Tách thành separate sessions |

---

## Quick Reference Card

> Bảng tra 30 giây — khi không chắc dùng skill nào.

```
TASK                                         SKILL(S)
─────────────────────────────────────────────────────────────────
ISICDataset / get_transforms (BLOCKING)  →  computer-vision-expert + python-pro
Thêm encoder/decoder architecture        →  computer-vision-expert
Training loop, AMP, DataLoader           →  ml-engineer
W&B, sweeps, experiment YAML             →  mlops-engineer
Dice/IoU analysis, ablation study        →  data-scientist
Training curves, overfitting diagnosis   →  data-scientist + debugging-strategies
NaN loss / GPU OOM / deadlock            →  debugging-strategies + ml-engineer
Val Dice plateau / not converging        →  debugging-strategies + data-scientist
pytest test suite                        →  python-pro
ruff / pyproject.toml / linter           →  python-patterns + python-pro
Type hints / docstrings audit            →  python-pro
Dockerize training pipeline              →  docker-expert + mlops-engineer
FastAPI inference container              →  docker-expert + architecture
FastAPI app design                       →  architecture + ml-engineer
ONNX / TorchScript export               →  ml-engineer
Multi-GPU DDP                            →  ml-engineer + mlops-engineer
─────────────────────────────────────────────────────────────────

INVOCATION SYNTAX
─────────────────────────────────────────────────────────────────
OpenCode    →  "Use @skill-name to ..."
Cursor      →  @skill-name [in Chat]
Antigravity →  "Use skill-name skill to ..."
─────────────────────────────────────────────────────────────────

ALWAYS INCLUDE IN PROMPT
─────────────────────────────────────────────────────────────────
Stack       →  Python 3.11, PyTorch 2.x, SMP>=0.3, Albumentations>=1.3
AMP         →  torch.amp.autocast(device_type="cuda") — NOT torch.cuda.amp
Registry    →  Follow _REGISTRY pattern in src/models/segmentation.py
Metrics     →  Report 4-tuple: dice@0.5, iou@0.5, dice@best_thr, best_thr
─────────────────────────────────────────────────────────────────
```

---

*Last updated: 2026-03-03 | Project: Skin-Cancer-Detection (ISIC 2018 Task 1)*
*Skills installed: `.agents/skills/` (OpenCode) · `.cursor/skills/` (Cursor) · `.agent/skills/` (Antigravity)*
