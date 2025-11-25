# Plan Implementation: ConvNeXt-tiny cho ISIC 2019

Plan dưới đây thiết kế riêng cho **ConvNeXt-tiny trên Kaggle (GPU T4x2)**, tối ưu hóa cho bài toán phân loại ung thư da ISIC 2019 (8 lớp).

> **ConvNeXt vs EfficientNet:**
> | Model | Parameters | FLOPs | ImageNet Top-1 |
> |-------|------------|-------|----------------|
> | EfficientNet-B0 | 5.3M | 0.39B | 77.1% |
> | **ConvNeXt-tiny** | **28.6M** | **4.5B** | **82.1%** |
>
> ConvNeXt-tiny mạnh hơn đáng kể nhưng cần điều chỉnh hyperparameters phù hợp.

> **Target mapping:**  
> - 0: MEL (Melanoma) ⚠️ MALIGNANT
> - 1: NV (Melanocytic nevus)  
> - 2: BCC (Basal cell carcinoma) ⚠️ MALIGNANT
> - 3: AK (Actinic keratosis)  
> - 4: BKL (Benign keratosis)  
> - 5: DF (Dermatofibroma)  
> - 6: VASC (Vascular lesion)  
> - 7: SCC (Squamous cell carcinoma) ⚠️ MALIGNANT

***

## 🎯 Các tối ưu hóa đặc biệt cho ConvNeXt-tiny

| Kỹ thuật | Mô tả | Kỳ vọng cải thiện |
|----------|-------|-------------------|
| **Layer-wise LR Decay (LLRD)** | `get_optimizer_params_with_llrd()` - Giảm LR theo layers | +1-2% accuracy |
| **Drop Path** | Stochastic depth thay vì dropout | Better regularization |
| **Mixup + CutMix** | `timm.data.Mixup` - Regularization mạnh | +1-3% accuracy |
| **Progressive Resizing** | 224 → 288 → 384 | +0.5-1% accuracy |
| **EMA (Exponential Moving Average)** | `timm.utils.ModelEmaV2` - Smoothing weights | +0.3-0.5% accuracy |
| **Gradient Accumulation** | Simulate larger batch size | Stable training |
| **AdamW với β₂=0.999** | Như ConvNeXt paper | Optimal convergence |
| **Cosine Annealing + Warmup** | Standard cho ConvNeXt | Smooth learning |

***

## 1. Cấu trúc notebook & config

**Cell 1 – Imports**

```python
import os
import cv2
import json
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2  # EMA có sẵn trong timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, 
    classification_report, roc_auc_score, roc_curve, auc,
    precision_recall_curve, f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
```

**Cell 2 – Config tối ưu cho ConvNeXt-tiny**

```python
@dataclass
class Config:
    # ==================== PATHS ====================
    CSV_PATH: str = "/kaggle/input/isic-2019-task-1/ISIC_2019_5folds_metadata.csv"
    IMG_ROOT: str = "/kaggle/input/isic-2019-task-1/cropped_lesions/cropped_lesions"
    OUTPUT_DIR: str = "/kaggle/working"
    TEST_CSV_PATH: str = "/kaggle/input/isic-2019-task-1/ISIC_2019_test_metadata.csv"
    TEST_IMG_ROOT: str = "/kaggle/input/isic-2019-task-1/cropped_lesions_testset/cropped_lesions_testset"
    
    # ==================== MODEL ====================
    MODEL_NAME: str = "convnext_tiny.fb_in22k_ft_in1k"  # Pretrained on ImageNet-22k, fine-tuned on ImageNet-1k
    N_CLASSES: int = 8
    PRETRAINED: bool = True
    DROP_PATH_RATE: float = 0.2  # Stochastic depth (thay vì dropout)
    
    # ==================== TRAINING ====================
    # Progressive Resizing: 224 → 288 (có thể thêm 384 nếu đủ VRAM)
    IMG_SIZE: int = 288  # ConvNeXt hoạt động tốt với sizes lớn hơn
    N_FOLDS: int = 5
    BATCH_SIZE: int = 32  # Giảm từ 64 do model lớn hơn (~28M params)
    ACCUMULATION_STEPS: int = 2  # Effective batch size = 32 * 2 = 64
    EPOCHS: int = 100
    MIN_EPOCHS: int = 10
    PATIENCE: int = 10
    
    # ==================== OPTIMIZER (ConvNeXt-specific) ====================
    BASE_LR: float = 5e-5  # Lower LR cho ConvNeXt fine-tuning
    MIN_LR: float = 1e-7
    WEIGHT_DECAY: float = 0.05  # ConvNeXt paper recommends 0.05
    WARMUP_EPOCHS: int = 3
    
    # Layer-wise Learning Rate Decay
    USE_LLRD: bool = True
    LLRD_DECAY: float = 0.75  # Mỗi layer group giảm LR * 0.75
    
    # ==================== REGULARIZATION ====================
    LABEL_SMOOTHING: float = 0.1
    
    # Mixup & CutMix (rất hiệu quả cho ConvNeXt)
    USE_MIXUP: bool = True
    MIXUP_ALPHA: float = 0.8
    CUTMIX_ALPHA: float = 1.0
    MIXUP_PROB: float = 0.5
    CUTMIX_PROB: float = 0.5
    MIXUP_SWITCH_PROB: float = 0.5  # Probability to switch between mixup and cutmix
    
    # ==================== LOSS ====================
    USE_FOCAL_LOSS: bool = True
    FOCAL_GAMMA: float = 2.0
    
    # ==================== CLASS BALANCING ====================
    USE_CLASS_BALANCED_SAMPLING: bool = True
    MALIGNANT_BOOST: float = 1.5
    SCC_EXTRA_BOOST: float = 2.0
    
    # ==================== EMA ====================
    USE_EMA: bool = True
    EMA_DECAY: float = 0.9998
    
    # ==================== HARDWARE ====================
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    
    # ==================== CLASS INFO ====================
    CLASS_NAMES: tuple = ('MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC')
    MALIGNANT_INDICES: tuple = (0, 2, 7)  # MEL, BCC, SCC

cfg = Config()

# Print config
print("="*60)
print("CONFIGURATION: ConvNeXt-tiny (Optimized)")
print("="*60)
print(f"Model: {cfg.MODEL_NAME}")
print(f"Parameters: ~28.6M")
print(f"Image Size: {cfg.IMG_SIZE}")
print(f"Batch Size: {cfg.BATCH_SIZE} (effective: {cfg.BATCH_SIZE * cfg.ACCUMULATION_STEPS})")
print(f"Learning Rate: {cfg.BASE_LR}")
print(f"Weight Decay: {cfg.WEIGHT_DECAY}")
print(f"Drop Path Rate: {cfg.DROP_PATH_RATE}")
print(f"Label Smoothing: {cfg.LABEL_SMOOTHING}")
print(f"Use LLRD: {cfg.USE_LLRD} (decay={cfg.LLRD_DECAY})")
print(f"Use Mixup/CutMix: {cfg.USE_MIXUP}")
print(f"Use Focal Loss: {cfg.USE_FOCAL_LOSS}")
print(f"Use EMA: {cfg.USE_EMA} (decay={cfg.EMA_DECAY})")
print(f"Use Class-balanced Sampling: {cfg.USE_CLASS_BALANCED_SAMPLING}")
print(f"Malignant Boost: {cfg.MALIGNANT_BOOST}x, SCC Extra: {cfg.SCC_EXTRA_BOOST}x")
print(f"Epochs: {cfg.EPOCHS}")
print(f"Device: {cfg.DEVICE}")
print("="*60)
```

***

## 2. Data Loading & Class Weights

**Cell 3 – Đọc CSV & tính class weight với Malignant Boost**

```python
# Đọc CSV
df = pd.read_csv(cfg.CSV_PATH)
print(f"Dataset size: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nClass distribution:")
print(df['target'].value_counts().sort_index())

# Tính class weight (inverse frequency) với MALIGNANT BOOST
counts = df["target"].value_counts().sort_index().values
N = len(df)
weights = N / counts
weights = weights / weights.mean()

# Áp dụng Malignant Boost
print(f"\n{'='*50}")
print("APPLYING MALIGNANT CLASS BOOST")
print(f"{'='*50}")

original_weights = weights.copy()
for idx in cfg.MALIGNANT_INDICES:
    weights[idx] *= cfg.MALIGNANT_BOOST
    print(f"  {cfg.CLASS_NAMES[idx]}: {original_weights[idx]:.4f} -> {weights[idx]:.4f} (x{cfg.MALIGNANT_BOOST})")

# SCC Extra Boost
scc_idx = 7
weights[scc_idx] *= cfg.SCC_EXTRA_BOOST
print(f"  SCC extra boost: {weights[scc_idx]/cfg.SCC_EXTRA_BOOST:.4f} -> {weights[scc_idx]:.4f} (x{cfg.SCC_EXTRA_BOOST})")

class_weight = torch.tensor(weights, dtype=torch.float).to(cfg.DEVICE)

print(f"\nFinal Class weights:")
for i, (name, w) in enumerate(zip(cfg.CLASS_NAMES, weights)):
    malignant_marker = "⚠️ MALIGNANT" if i in cfg.MALIGNANT_INDICES else ""
    print(f"  {i}: {name} = {w:.4f} {malignant_marker}")

# Fold distribution
print(f"\nFold distribution:")
print(df['fold'].value_counts().sort_index())
```

***

## 3. Augmentation (Mạnh hơn cho ConvNeXt)

**Cell 4 – Augmentation pipeline**

```python
def get_train_transforms(img_size):
    """
    Augmentation mạnh cho ConvNeXt.
    ConvNeXt được train với RandAugment, nên cần augmentation tương đương.
    """
    return A.Compose([
        # Geometric transforms
        A.RandomResizedCrop(
            size=(img_size, img_size), 
            scale=(0.6, 1.0),  # Rộng hơn cho diversity
            ratio=(0.8, 1.2),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.2, 
            rotate_limit=180,  # Full rotation cho skin lesions
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ),
        
        # Color transforms (quan trọng cho skin tone variation)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3, 
                contrast_limit=0.3, 
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=30, 
                sat_shift_limit=40, 
                val_shift_limit=30, 
                p=1.0
            ),
            A.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1, 
                p=1.0
            ),
        ], p=0.8),
        
        # Noise & Blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.4),
        
        # Advanced transforms
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.1, p=1.0),
            A.GridDistortion(distort_limit=0.2, p=1.0),
            A.ElasticTransform(
                alpha=120, 
                sigma=120 * 0.05, 
                alpha_affine=120 * 0.03, 
                p=1.0
            ),
        ], p=0.3),
        
        # CLAHE (tăng contrast local - rất tốt cho dermatology)
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
        
        # Dropout/Cutout variants
        A.OneOf([
            A.CoarseDropout(
                max_holes=12, 
                max_height=img_size//6, 
                max_width=img_size//6,
                min_holes=4,
                min_height=img_size//12, 
                min_width=img_size//12,
                fill_value=0, 
                p=1.0
            ),
            A.GridDropout(
                ratio=0.3, 
                unit_size_min=img_size//8, 
                unit_size_max=img_size//4, 
                p=1.0
            ),
        ], p=0.5),
        
        # Normalize với ImageNet stats (ConvNeXt pretrained on ImageNet)
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def get_val_transforms(img_size):
    """Validation transforms - chỉ resize và normalize"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size):
    """
    Test Time Augmentation transforms.
    Returns list of transform pipelines.
    """
    base_transforms = [
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    
    tta_list = [
        A.Compose(base_transforms),  # Original
        A.Compose([A.HorizontalFlip(p=1.0)] + base_transforms),
        A.Compose([A.VerticalFlip(p=1.0)] + base_transforms),
        A.Compose([A.Transpose(p=1.0)] + base_transforms),
        A.Compose([A.Rotate(limit=(90, 90), p=1.0)] + base_transforms),
        A.Compose([A.Rotate(limit=(180, 180), p=1.0)] + base_transforms),
        A.Compose([A.Rotate(limit=(270, 270), p=1.0)] + base_transforms),
    ]
    
    return tta_list
```

**Cell 5 – Dataset class**

```python
class ISICDataset(Dataset):
    def __init__(self, df, transforms=None, img_root=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.img_root = img_root
        self.is_test = is_test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = os.path.join(self.img_root, os.path.basename(row["path"]))
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        
        if self.is_test:
            return image, row['image']  # Return image name for test
        
        target = torch.tensor(row["target"], dtype=torch.long)
        return image, target


# Test dataset
print("Testing dataset...")
test_ds = ISICDataset(df.head(5), get_val_transforms(cfg.IMG_SIZE), cfg.IMG_ROOT)
img, target = test_ds[0]
print(f"Image shape: {img.shape}")
print(f"Target: {target} ({cfg.CLASS_NAMES[target]})")
print("✓ Dataset working correctly!")
```

***

## 4. Model Architecture & Advanced Components

**Cell 6 – Model Factory với Layer-wise Learning Rate Decay**

```python
def create_model(cfg):
    """
    Tạo ConvNeXt-tiny model với drop_path.
    """
    model = timm.create_model(
        cfg.MODEL_NAME,
        pretrained=cfg.PRETRAINED,
        num_classes=cfg.N_CLASSES,
        drop_path_rate=cfg.DROP_PATH_RATE,  # Stochastic depth
    )
    
    return model.to(cfg.DEVICE)


def get_optimizer_params_with_llrd(model, cfg):
    """
    Layer-wise Learning Rate Decay (LLRD).
    Chia parameters thành groups với LR giảm dần từ head xuống backbone.
    
    ConvNeXt structure:
    - stem (downsample_layers.0)
    - stages (0, 1, 2, 3) 
    - head (norm + fc)
    
    Args:
        model: ConvNeXt model
        cfg: Config object với BASE_LR, LLRD_DECAY, WEIGHT_DECAY
        
    Returns:
        List of parameter groups với different learning rates
    """
    if not cfg.USE_LLRD:
        return [{'params': model.parameters(), 'lr': cfg.BASE_LR, 'weight_decay': cfg.WEIGHT_DECAY}]
    
    # Định nghĩa layer groups (từ head xuống stem)
    # LR giảm dần khi đi sâu vào backbone
    layer_names = [
        'head',           # Classifier head (highest LR)
        'norm',           # Final LayerNorm
        'stages.3',       # Stage 3
        'stages.2',       # Stage 2
        'stages.1',       # Stage 1
        'stages.0',       # Stage 0
        'stem',           # Stem/Patch embed (lowest LR)
        'downsample_layers',  # Downsample layers between stages
    ]
    
    # Tính LR scale cho mỗi layer group
    # head: 1.0, norm: 0.75, stages.3: 0.5625, ...
    lr_scales = [cfg.LLRD_DECAY ** i for i in range(len(layer_names))]
    
    param_groups = []
    assigned_params = set()
    
    for layer_name, lr_scale in zip(layer_names, lr_scales):
        group_params = []
        group_params_no_decay = []  # Bias và LayerNorm không cần weight decay
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if layer_name in name and id(param) not in assigned_params:
                assigned_params.add(id(param))
                
                # Không áp dụng weight decay cho bias và norm layers
                if 'bias' in name or 'norm' in name or 'gamma' in name:
                    group_params_no_decay.append(param)
                else:
                    group_params.append(param)
        
        if group_params:
            param_groups.append({
                'params': group_params,
                'lr': cfg.BASE_LR * lr_scale,
                'weight_decay': cfg.WEIGHT_DECAY,
                'name': f'{layer_name}_decay'
            })
        
        if group_params_no_decay:
            param_groups.append({
                'params': group_params_no_decay,
                'lr': cfg.BASE_LR * lr_scale,
                'weight_decay': 0.0,
                'name': f'{layer_name}_no_decay'
            })
    
    # Params còn lại (nếu có)
    remaining_params = []
    remaining_params_no_decay = []
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in assigned_params:
            if 'bias' in name or 'norm' in name:
                remaining_params_no_decay.append(param)
            else:
                remaining_params.append(param)
    
    if remaining_params:
        param_groups.append({
            'params': remaining_params,
            'lr': cfg.BASE_LR * lr_scales[-1],
            'weight_decay': cfg.WEIGHT_DECAY,
            'name': 'remaining_decay'
        })
    
    if remaining_params_no_decay:
        param_groups.append({
            'params': remaining_params_no_decay,
            'lr': cfg.BASE_LR * lr_scales[-1],
            'weight_decay': 0.0,
            'name': 'remaining_no_decay'
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("LAYER-WISE LEARNING RATE DECAY (LLRD)")
    print(f"{'='*60}")
    total_params = 0
    for group in param_groups:
        n_params = sum(p.numel() for p in group['params'])
        total_params += n_params
        print(f"  {group['name']:25s}: LR={group['lr']:.2e}, WD={group['weight_decay']:.2e}, params={n_params:,}")
    print(f"  {'TOTAL':25s}: params={total_params:,}")
    
    return param_groups


# ==================== EMA (sử dụng timm.utils.ModelEmaV2) ====================
# ModelEmaV2 từ timm đã được import ở đầu file
# Cách sử dụng:
#   ema = ModelEmaV2(model, decay=0.9998, device=cfg.DEVICE)
#   ema.update(model)  # Gọi sau mỗi optimizer step
#   
# Khi validate/inference với EMA weights:
#   outputs = ema.module(images)  # Sử dụng ema.module thay vì model
#
# Không cần apply_shadow/restore như custom EMA class


# ==================== TEST MODEL & LLRD ====================
print("\nTesting model creation...")
test_model = create_model(cfg)
n_params = sum(p.numel() for p in test_model.parameters())
n_trainable = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
print(f"Model: {cfg.MODEL_NAME}")
print(f"Total parameters: {n_params:,}")
print(f"Trainable parameters: {n_trainable:,}")

# Test LLRD parameter groups
print("\n--- Testing LLRD ---")
test_param_groups = get_optimizer_params_with_llrd(test_model, cfg)
print(f"Number of param groups: {len(test_param_groups)}")

# Test forward pass
with torch.no_grad():
    dummy_input = torch.randn(2, 3, cfg.IMG_SIZE, cfg.IMG_SIZE).to(cfg.DEVICE)
    output = test_model(dummy_input)
    print(f"Output shape: {output.shape}")

# Test EMA (ModelEmaV2 từ timm)
print("\n--- Testing EMA (ModelEmaV2) ---")
test_ema = ModelEmaV2(test_model, decay=cfg.EMA_DECAY, device=cfg.DEVICE)
print(f"EMA model type: {type(test_ema.module)}")
with torch.no_grad():
    ema_output = test_ema.module(dummy_input)
    print(f"EMA output shape: {ema_output.shape}")

del test_model, test_ema
torch.cuda.empty_cache()
print("\n✓ Model, LLRD, and EMA working correctly!")
```

**Cell 7 – Loss Functions & Mixup**

```python
# ==================== FOCAL LOSS ====================
class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    FL(p_t) = -α(1-p_t)^γ * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.alpha, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        p = torch.softmax(inputs, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ==================== MIXUP / CUTMIX (sử dụng timm.data.Mixup) ====================
def setup_mixup(cfg):
    """
    Setup Mixup/CutMix sử dụng timm.data.Mixup.
    
    Mixup/CutMix là kỹ thuật regularization rất hiệu quả:
    - Mixup: Trộn 2 images và labels theo tỷ lệ λ ~ Beta(α, α)
    - CutMix: Cắt patch từ image này paste vào image khác
    
    timm.data.Mixup tự động:
    - Chọn Mixup hoặc CutMix theo probability
    - Convert hard labels sang soft labels (one-hot với mixing)
    - Áp dụng label smoothing
    
    Args:
        cfg: Config object
        
    Returns:
        mixup_fn: Mixup function từ timm (hoặc None)
        mixup_criterion: SoftTargetCrossEntropy cho soft labels
    """
    if not cfg.USE_MIXUP:
        return None, None
    
    # timm.data.Mixup tự động xử lý:
    # 1. Mixup với probability = prob * (1 - switch_prob)
    # 2. CutMix với probability = prob * switch_prob
    # 3. mode='batch': Áp dụng trên cả batch (hiệu quả hơn 'pair' hay 'elem')
    mixup_fn = Mixup(
        mixup_alpha=cfg.MIXUP_ALPHA,       # Beta distribution α cho Mixup
        cutmix_alpha=cfg.CUTMIX_ALPHA,     # Beta distribution α cho CutMix
        cutmix_minmax=None,                # Min/max ratio cho CutMix patch (None = use α)
        prob=cfg.MIXUP_PROB,               # Probability để apply Mixup/CutMix
        switch_prob=cfg.MIXUP_SWITCH_PROB, # Probability để switch từ Mixup sang CutMix
        mode='batch',                       # 'batch', 'pair', 'elem'
        label_smoothing=cfg.LABEL_SMOOTHING,
        num_classes=cfg.N_CLASSES
    )
    
    # Khi dùng Mixup, targets trở thành soft labels (one-hot với mixed values)
    # => Cần SoftTargetCrossEntropy thay vì CrossEntropyLoss
    # SoftTargetCrossEntropy: loss = -sum(target * log_softmax(pred))
    mixup_criterion = SoftTargetCrossEntropy()
    
    print("✓ Mixup/CutMix enabled (timm.data.Mixup)")
    print(f"  Mixup α: {cfg.MIXUP_ALPHA}")
    print(f"  CutMix α: {cfg.CUTMIX_ALPHA}")
    print(f"  Apply Prob: {cfg.MIXUP_PROB}")
    print(f"  Switch Prob: {cfg.MIXUP_SWITCH_PROB}")
    print(f"  Label Smoothing: {cfg.LABEL_SMOOTHING}")
    
    return mixup_fn, mixup_criterion


# ==================== CRITERION SETUP ====================
# Standard criterion (không Mixup)
if cfg.USE_FOCAL_LOSS:
    criterion = FocalLoss(
        alpha=class_weight,
        gamma=cfg.FOCAL_GAMMA,
        label_smoothing=cfg.LABEL_SMOOTHING,
        reduction='mean'
    )
    print(f"✓ Using Focal Loss (γ={cfg.FOCAL_GAMMA})")
else:
    criterion = nn.CrossEntropyLoss(
        weight=class_weight, 
        label_smoothing=cfg.LABEL_SMOOTHING
    )
    print(f"✓ Using CrossEntropyLoss")

# Mixup setup
mixup_fn, mixup_criterion = setup_mixup(cfg)
```

**Cell 8 – Scheduler**

```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    """
    Cosine schedule với linear warmup.
    """
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale to [min_lr_ratio, 1.0]
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

***

## 5. Training Loop với tất cả optimizations

**Cell 9 – Training function**

```python
def train_one_fold(fold, df, cfg):
    print(f"\n{'='*60}")
    print(f"FOLD {fold}")
    print(f"{'='*60}")
    
    # ==================== DATA PREPARATION ====================
    train_df = df[df.fold != fold].reset_index(drop=True)
    val_df = df[df.fold == fold].reset_index(drop=True)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    train_dataset = ISICDataset(train_df, get_train_transforms(cfg.IMG_SIZE), cfg.IMG_ROOT)
    val_dataset = ISICDataset(val_df, get_val_transforms(cfg.IMG_SIZE), cfg.IMG_ROOT)
    
    # Class-balanced sampling
    if cfg.USE_CLASS_BALANCED_SAMPLING:
        train_counts = train_df['target'].value_counts().sort_index().values
        train_sample_weights = 1.0 / train_counts[train_df['target'].values]
        sampler = WeightedRandomSampler(
            weights=train_sample_weights,
            num_samples=len(train_sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.BATCH_SIZE,
            sampler=sampler,
            num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=True
        )
        print(f"✓ Using Class-balanced Sampling")
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    
    # ==================== MODEL & OPTIMIZER ====================
    model = create_model(cfg)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Optimizer với LLRD (Layer-wise Learning Rate Decay)
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    param_groups = get_optimizer_params_with_llrd(base_model, cfg)
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.BASE_LR,
        betas=(0.9, 0.999),  # ConvNeXt paper settings
        # weight_decay đã được set trong param_groups
    )
    
    # Scheduler
    num_training_steps = (len(train_loader) // cfg.ACCUMULATION_STEPS) * cfg.EPOCHS
    num_warmup_steps = (len(train_loader) // cfg.ACCUMULATION_STEPS) * cfg.WARMUP_EPOCHS
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps, 
        num_training_steps,
        min_lr_ratio=cfg.MIN_LR / cfg.BASE_LR
    )
    
    # AMP Scaler
    scaler = GradScaler()
    
    # EMA (sử dụng ModelEmaV2 từ timm)
    if cfg.USE_EMA:
        ema = ModelEmaV2(
            model, 
            decay=cfg.EMA_DECAY, 
            device=cfg.DEVICE
        )
        print(f"✓ Using EMA with decay={cfg.EMA_DECAY}")
    else:
        ema = None
    
    # ==================== TRAINING STATE ====================
    best_bal_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, f"convnext_tiny_fold{fold}_best.pth")
    
    oof_preds = []
    oof_targets = []
    oof_images = []
    
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'bal_acc': [], 'learning_rate': []
    }
    
    # ==================== TRAINING LOOP ====================
    for epoch in range(1, cfg.EPOCHS + 1):
        # ---- TRAIN PHASE ----
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Train]")
        for step, (images, targets) in enumerate(pbar):
            images = images.to(cfg.DEVICE)
            targets = targets.to(cfg.DEVICE)
            
            # Apply Mixup/CutMix (timm.data.Mixup)
            # mixup_fn tự động:
            # 1. Random chọn Mixup hoặc CutMix
            # 2. Mix images và convert targets sang soft labels
            if mixup_fn is not None:
                images, targets_mixed = mixup_fn(images, targets)
                # targets_mixed là soft labels (N, num_classes) tensor
                current_criterion = mixup_criterion  # SoftTargetCrossEntropy
            else:
                targets_mixed = targets
                current_criterion = criterion  # FocalLoss hoặc CrossEntropyLoss
            
            # Forward pass với AMP
            with autocast():
                outputs = model(images)
                loss = current_criterion(outputs, targets_mixed)
                loss = loss / cfg.ACCUMULATION_STEPS
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % cfg.ACCUMULATION_STEPS == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                # Update EMA (ModelEmaV2 từ timm)
                if ema is not None:
                    ema.update(model)
            
            train_loss += loss.item() * cfg.ACCUMULATION_STEPS
            pbar.set_postfix({
                'loss': f"{loss.item() * cfg.ACCUMULATION_STEPS:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_train_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # ---- VALIDATION PHASE ----
        # Sử dụng EMA model (ema.module) nếu có, không cần apply_shadow/restore
        eval_model = ema.module if ema is not None else model
        eval_model.eval()
        
        val_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Val]"):
                images = images.to(cfg.DEVICE)
                targets = targets.to(cfg.DEVICE)
                
                with autocast():
                    outputs = eval_model(images)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        all_probs = np.vstack(all_probs)
        bal_acc = balanced_accuracy_score(all_targets, all_preds)
        
        # Save history
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['bal_acc'].append(bal_acc)
        history['learning_rate'].append(current_lr)
        
        # Print metrics
        gap = avg_train_loss - avg_val_loss
        overfit_warning = "⚠️ Overfitting!" if gap < -0.3 else ""
        ema_tag = "[EMA]" if ema is not None else ""
        
        print(f"\nEpoch {epoch} {ema_tag}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, bal_acc={bal_acc:.4f} {overfit_warning}")
        
        # ---- EARLY STOPPING ----
        if bal_acc > best_bal_acc + 1e-3:
            best_bal_acc = bal_acc
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            # Save checkpoint (EMA weights nếu có, không cần apply_shadow/restore)
            # Với ModelEmaV2 từ timm, EMA weights nằm trong ema.module
            if ema is not None:
                save_model = ema.module
            else:
                save_model = model.module if isinstance(model, nn.DataParallel) else model
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_bal_acc': best_bal_acc,
                'best_val_loss': best_val_loss,
                'config': {
                    'model_name': cfg.MODEL_NAME,
                    'img_size': cfg.IMG_SIZE,
                    'drop_path_rate': cfg.DROP_PATH_RATE,
                }
            }, checkpoint_path)
            
            print(f"✓ Saved best model: bal_acc={best_bal_acc:.4f}, val_loss={best_val_loss:.4f}")
            
            # Save OOF predictions
            oof_preds = all_probs
            oof_targets = all_targets
            oof_images = val_df['image'].tolist()
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
        
        if epoch >= cfg.MIN_EPOCHS and epochs_no_improve >= cfg.PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Cleanup
    del model, optimizer, scaler
    if ema is not None:
        del ema
    torch.cuda.empty_cache()
    
    return {
        'fold': fold,
        'best_bal_acc': best_bal_acc,
        'best_val_loss': best_val_loss,
        'checkpoint_path': checkpoint_path,
        'oof_preds': oof_preds,
        'oof_targets': oof_targets,
        'oof_images': oof_images,
        'history': history,
    }
```

**Cell 10 – Run 5-Fold Training**

```python
# Chạy training cho tất cả folds
results = []
all_oof = []
all_histories = []

for fold in range(cfg.N_FOLDS):
    fold_result = train_one_fold(fold, df, cfg)
    results.append(fold_result)
    all_histories.append(fold_result['history'])
    
    # Thu thập OOF
    fold_oof = pd.DataFrame({
        'image': fold_result['oof_images'],
        'fold': fold,
        'target': fold_result['oof_targets'],
        **{f'prob_{i}': fold_result['oof_preds'][:, i] for i in range(cfg.N_CLASSES)}
    })
    all_oof.append(fold_oof)

# Tổng kết
bal_accs = [r['best_bal_acc'] for r in results]
print(f"\n{'='*60}")
print(f"RESULTS SUMMARY - ConvNeXt-tiny")
print(f"{'='*60}")
for i, acc in enumerate(bal_accs):
    print(f"Fold {i}: {acc:.4f}")
print(f"Mean: {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}")
print(f"{'='*60}")
```

***

## 6. OOF Evaluation & Threshold Optimization

**Cell 11 – OOF Evaluation**

```python
# Concat OOF predictions
oof_df = pd.concat(all_oof, ignore_index=True)
oof_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'oof_convnext_tiny.csv'), index=False)

# Calculate metrics
oof_preds_class = oof_df[[f'prob_{i}' for i in range(cfg.N_CLASSES)]].values.argmax(axis=1)
oof_bal_acc = balanced_accuracy_score(oof_df['target'], oof_preds_class)

print(f"\nOOF Balanced Accuracy: {oof_bal_acc:.4f}")
print("\nClassification Report:")
print(classification_report(oof_df['target'], oof_preds_class, target_names=cfg.CLASS_NAMES))

# Per-class ROC-AUC
print("\nPer-class ROC-AUC:")
for i, class_name in enumerate(cfg.CLASS_NAMES):
    y_true_binary = (oof_df['target'] == i).astype(int)
    y_prob = oof_df[f'prob_{i}'].values
    try:
        auc_score = roc_auc_score(y_true_binary, y_prob)
        malignant = "⚠️ MALIGNANT" if class_name in ['MEL', 'BCC', 'SCC'] else ""
        print(f"  {class_name}: {auc_score:.4f} {malignant}")
    except:
        print(f"  {class_name}: N/A")

# Confusion matrix
cm = confusion_matrix(oof_df['target'], oof_preds_class)
```

**Cell 12 – Threshold Optimization** (như trong EfficientNet plan)

```python
# Threshold optimization functions (giữ nguyên như EfficientNet plan)
# find_optimal_thresholds(), apply_optimized_thresholds(), evaluate_with_optimized_thresholds()
# ... (copy từ EfficientNet plan)
```

***

## 7. Inference với TTA

**Cell 13 – Inference với TTA mạnh**

```python
def inference_with_tta(test_csv_path, test_img_root, cfg, n_tta=7):
    """
    Inference với ensemble 5 folds + TTA (7 augmentations).
    """
    test_df = pd.read_csv(test_csv_path)
    tta_transforms = get_tta_transforms(cfg.IMG_SIZE)
    
    print(f"\n{'='*60}")
    print(f"TEST SET INFERENCE - ConvNeXt-tiny")
    print(f"{'='*60}")
    print(f"Test samples: {len(test_df)}")
    print(f"TTA augmentations: {n_tta}")
    
    # Load models
    models = []
    for fold in range(cfg.N_FOLDS):
        model = create_model(cfg)
        checkpoint_path = os.path.join(cfg.OUTPUT_DIR, f"convnext_tiny_fold{fold}_best.pth")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        print(f"  ✓ Loaded fold {fold} (bal_acc: {checkpoint['best_bal_acc']:.4f})")
    
    # Predict with TTA
    all_preds = []
    all_images = []
    
    for idx in tqdm(range(len(test_df)), desc="Inference"):
        row = test_df.iloc[idx]
        img_path = os.path.join(test_img_root, os.path.basename(row["path"]))
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Cannot read {img_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_images.append(row['image'])
        
        # Ensemble: 5 folds × 7 TTA = 35 predictions
        ensemble_preds = []
        
        for model in models:
            for tta_idx, tta_transform in enumerate(tta_transforms[:n_tta]):
                img_tensor = tta_transform(image=image)["image"].unsqueeze(0).to(cfg.DEVICE)
                
                with torch.no_grad(), autocast():
                    pred = torch.softmax(model(img_tensor), dim=1)
                
                ensemble_preds.append(pred)
        
        # Average all predictions
        avg_pred = torch.stack(ensemble_preds).mean(dim=0).cpu().numpy()[0]
        all_preds.append(avg_pred)
    
    all_preds = np.array(all_preds)
    pred_classes = all_preds.argmax(axis=1)
    
    # Create submission
    submission = pd.DataFrame({
        'image': all_images,
        **{f'prob_{i}': all_preds[:, i] for i in range(cfg.N_CLASSES)},
        'pred': pred_classes,
        'pred_label': [cfg.CLASS_NAMES[p] for p in pred_classes]
    })
    
    output_path = os.path.join(cfg.OUTPUT_DIR, 'convnext_tiny_test_predictions.csv')
    submission.to_csv(output_path, index=False)
    print(f"\n✓ Saved predictions to {output_path}")
    
    # Statistics
    print(f"\nPrediction Distribution:")
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        count = (pred_classes == i).sum()
        pct = count / len(pred_classes) * 100
        print(f"  {class_name}: {count} ({pct:.1f}%)")
    
    # Cleanup
    for model in models:
        del model
    torch.cuda.empty_cache()
    
    return submission


# Run inference
test_submission = inference_with_tta(
    cfg.TEST_CSV_PATH, 
    cfg.TEST_IMG_ROOT, 
    cfg, 
    n_tta=7
)
```

***

## 8. Save Results

**Cell 14 – Save experiment results**

```python
# Giống EfficientNet plan nhưng cập nhật config cho ConvNeXt
def save_experiment_results(cfg, results, all_histories, oof_df, test_submission, save_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config_dict = {
        'experiment_timestamp': timestamp,
        'model_name': cfg.MODEL_NAME,
        'model_params': '28.6M',
        'n_classes': cfg.N_CLASSES,
        'img_size': cfg.IMG_SIZE,
        'n_folds': cfg.N_FOLDS,
        'batch_size': cfg.BATCH_SIZE,
        'accumulation_steps': cfg.ACCUMULATION_STEPS,
        'effective_batch_size': cfg.BATCH_SIZE * cfg.ACCUMULATION_STEPS,
        'epochs': cfg.EPOCHS,
        'base_lr': cfg.BASE_LR,
        'weight_decay': cfg.WEIGHT_DECAY,
        'drop_path_rate': cfg.DROP_PATH_RATE,
        'label_smoothing': cfg.LABEL_SMOOTHING,
        # ConvNeXt-specific
        'use_llrd': cfg.USE_LLRD,
        'llrd_decay': cfg.LLRD_DECAY,
        'use_mixup': cfg.USE_MIXUP,
        'mixup_alpha': cfg.MIXUP_ALPHA,
        'cutmix_alpha': cfg.CUTMIX_ALPHA,
        'use_ema': cfg.USE_EMA,
        'ema_decay': cfg.EMA_DECAY,
        # Loss & Balancing
        'use_focal_loss': cfg.USE_FOCAL_LOSS,
        'focal_gamma': cfg.FOCAL_GAMMA,
        'use_class_balanced_sampling': cfg.USE_CLASS_BALANCED_SAMPLING,
        'malignant_boost': cfg.MALIGNANT_BOOST,
        'scc_extra_boost': cfg.SCC_EXTRA_BOOST,
        'class_names': list(cfg.CLASS_NAMES),
    }
    
    # Save config
    with open(os.path.join(save_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # ... (phần còn lại giống EfficientNet plan)

save_experiment_results(cfg, results, all_histories, oof_df, test_submission, cfg.OUTPUT_DIR)
```

***

## 9. Tổng kết Config

| Config | ConvNeXt-tiny | EfficientNet-B0 | Lý do thay đổi |
|--------|---------------|-----------------|----------------|
| `MODEL_NAME` | convnext_tiny.fb_in22k_ft_in1k | efficientnet_b0 | Model mạnh hơn |
| `IMG_SIZE` | **288** | 256 | ConvNeXt tốt hơn với size lớn |
| `BATCH_SIZE` | **32** | 64 | Model lớn hơn (~28M vs 5M params) |
| `ACCUMULATION_STEPS` | **2** | 1 | Effective batch = 64 |
| `BASE_LR` | **5e-5** | 3e-4 | Lower LR cho ConvNeXt fine-tuning |
| `WEIGHT_DECAY` | **0.05** | 5e-4 | ConvNeXt paper recommends |
| `DROP_PATH_RATE` | **0.2** | 0.4 (dropout) | Stochastic depth thay dropout |
| `USE_LLRD` | **True** | False | Layer-wise LR decay |
| `LLRD_DECAY` | **0.75** | - | Standard value |
| `USE_MIXUP` | **True** | False | Regularization mạnh |
| `MIXUP_ALPHA` | **0.8** | - | Standard value |
| `CUTMIX_ALPHA` | **1.0** | - | Standard value |
| `USE_EMA` | **True** | False | Improve generalization (ModelEmaV2 từ timm) |
| `EMA_DECAY` | **0.9998** | - | Standard value |
| `PATIENCE` | **10** | 10 | Cho model đủ thời gian converge |

***

## 10. Output Files

| File | Mô tả |
|------|-------|
| `experiment_config.json` | Config với ConvNeXt-specific params |
| `training_results.json` | Kết quả từng fold |
| `training_history.csv` | History chi tiết |
| `oof_metrics.json` | OOF metrics |
| `oof_convnext_tiny.csv` | OOF predictions |
| `optimal_thresholds.json` | Optimal thresholds per class |
| `convnext_tiny_test_predictions.csv` | Test predictions |
| `convnext_tiny_fold*_best.pth` | Model checkpoints |
| `*.png` | Visualizations |

***

## 11. Kỳ vọng Performance

| Metric | EfficientNet-B0 (v2) | ConvNeXt-tiny | Improvement |
|--------|---------------------|---------------|-------------|
| **Balanced Accuracy** | 0.70-0.75 | **0.75-0.80** | +5% |
| **Mean AUC** | ~0.87 | **0.90+** | +3% |
| **SCC AUC** | 0.85 | **0.88+** | +3% |
| **MEL AUC** | 0.88 | **0.91+** | +3% |
| **Minority F1 (DF, VASC)** | ~0.30 | **0.40+** | +10% |

**Lý do ConvNeXt-tiny tốt hơn:**
1. **Capacity lớn hơn** (28M vs 5M params) - học được features phức tạp hơn
2. **Pretrained on ImageNet-22k** - transfer learning tốt hơn
3. **Modern architecture** - ConvNeXt được thiết kế để cạnh tranh với ViT
4. **Mixup/CutMix** - regularization mạnh cho imbalanced data
5. **EMA** - giúp model generalize tốt hơn
6. **LLRD** - fine-tuning tối ưu cho pretrained models

***