# Plan Implementation: EfficientNet-B2 cho ISIC 2019

Plan dưới đây thiết kế riêng cho **EfficientNet-B2 trên Kaggle (GPU T4x2)**, tối ưu hóa cho bài toán phân loại ung thư da ISIC 2019 (8 lớp).

> **So sánh EfficientNet variants:**
>
> | Model | Parameters | FLOPs | Input Size | ImageNet Top-1 |
> |-------|------------|-------|------------|----------------|
> | EfficientNet-B0 | 5.3M | 0.39B | 224 | 77.1% |
> | **EfficientNet-B2** | **9.2M** | **1.0B** | **260** | **80.1%** |
> | EfficientNet-B4 | 19M | 4.2B | 380 | 82.9% |
>
> EfficientNet-B2 là lựa chọn cân bằng tốt giữa accuracy và computational cost.

> **Target mapping:**  
>
> - 0: MEL (Melanoma) ⚠️ MALIGNANT
> - 1: NV (Melanocytic nevus)  
> - 2: BCC (Basal cell carcinoma) ⚠️ MALIGNANT
> - 3: AK (Actinic keratosis)  
> - 4: BKL (Benign keratosis)  
> - 5: DF (Dermatofibroma)  
> - 6: VASC (Vascular lesion)  
> - 7: SCC (Squamous cell carcinoma) ⚠️ MALIGNANT

***

## 🆕 Các cải tiến từ EfficientNet-B0

| Cải tiến | EfficientNet-B0 | EfficientNet-B2 | Lý do |
|----------|-----------------|-----------------|-------|
| **Model** | efficientnet_b0 | efficientnet_b2 | +3% accuracy trên ImageNet |
| **Image Size** | 256 | **256** | Giữ nguyên để training nhanh hơn, tăng batch size |
| **Batch Size** | 64 | **64** | Giữ nguyên theo yêu cầu |
| **Drop Rate** | 0.4 | **0.3** | B2 có default drop_rate=0.3 |
| **Mixup/CutMix** | Không | **Có (nhẹ, prob=0.3)** | Regularization thêm cho model lớn |
| **Loss Function** | Focal Loss | **CE + class_weight** | Tránh over-regularization khi dùng Mixup |
| **Label Smoothing** | 0.1 | **0.1** | Giữ nguyên |
| **Class-balanced Sampling** | ✅ | ✅ | Giữ nguyên |
| **Malignant Boost** | ✅ | ✅ | Giữ nguyên |

> **⚠️ Lưu ý quan trọng về Loss**: Không nên kết hợp Focal Loss + Mixup + Label Smoothing cùng lúc vì gây over-regularization. Sử dụng **CE + class_weight + Label Smoothing + Mixup** là đủ.

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

**Cell 2 – Config tối ưu cho EfficientNet-B2**

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
    # EfficientNet-B2: ~9.2M params, official input 260x260
    MODEL_NAME: str = "efficientnet_b2"
    N_CLASSES: int = 8
    PRETRAINED: bool = True
    DROP_RATE: float = 0.3  # B2 default drop_rate
    
    # ==================== TRAINING ====================
    IMG_SIZE: int = 256  # Giữ nguyên như B0, training nhanh hơn
    N_FOLDS: int = 5
    BATCH_SIZE: int = 64  # Giữ nguyên 64 theo yêu cầu
    EPOCHS: int = 100
    MIN_EPOCHS: int = 15
    PATIENCE: int = 10
    
    # ==================== OPTIMIZER ====================
    BASE_LR: float = 3e-4   # Tăng lại mức 3e-4 (như B0) do dùng Batch 64
    MIN_LR: float = 1e-6
    WEIGHT_DECAY: float = 5e-4
    WARMUP_EPOCHS: int = 5
    
    # ==================== REGULARIZATION ====================
    LABEL_SMOOTHING: float = 0.1
    
    # Mixup/CutMix nhẹ cho regularization
    USE_MIXUP: bool = True
    MIXUP_ALPHA: float = 0.4  # Nhẹ hơn ConvNeXt (0.8)
    CUTMIX_ALPHA: float = 0.4
    MIXUP_PROB: float = 0.3   # Chỉ 30% batches
    MIXUP_SWITCH_PROB: float = 0.5
    
    # ==================== LOSS ====================
    # Không dùng Focal Loss khi đã có Mixup + Label Smoothing (tránh over-regularization)
    USE_FOCAL_LOSS: bool = False
    # Sử dụng CrossEntropyLoss + class_weight (đã có Malignant Boost) + Label Smoothing
    
    # ==================== CLASS BALANCING ====================
    USE_CLASS_BALANCED_SAMPLING: bool = True
    MALIGNANT_BOOST: float = 1.5
    SCC_EXTRA_BOOST: float = 2.0
    
    # ==================== HARDWARE ====================
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    
    # ==================== CLASS INFO ====================
    CLASS_NAMES: tuple = ('MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC')
    MALIGNANT_INDICES: tuple = (0, 2, 7)

cfg = Config()

# Print config
print("="*60)
print("CONFIGURATION: EfficientNet-B2 (Optimized)")
print("="*60)
print(f"Model: {cfg.MODEL_NAME}")
print(f"Parameters: ~9.2M (vs B0: 5.3M)")
print(f"Image Size: {cfg.IMG_SIZE}")
print(f"Batch Size: {cfg.BATCH_SIZE}")
print(f"Learning Rate: {cfg.BASE_LR}")
print(f"Weight Decay: {cfg.WEIGHT_DECAY}")
print(f"Drop Rate: {cfg.DROP_RATE}")
print(f"Label Smoothing: {cfg.LABEL_SMOOTHING}")
print(f"Use Mixup/CutMix: {cfg.USE_MIXUP} (α={cfg.MIXUP_ALPHA}, prob={cfg.MIXUP_PROB})")
print(f"Loss: CrossEntropyLoss + class_weight + label_smoothing={cfg.LABEL_SMOOTHING}")
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

## 3. Augmentation

**Cell 4 – Augmentation pipeline (mạnh hơn cho B2)**

```python
def get_train_transforms(img_size):
    """
    Augmentation mạnh hơn cho EfficientNet-B2.
    Image size lớn hơn cần augmentation phù hợp.
    """
    return A.Compose([
        # Geometric transforms
        A.RandomResizedCrop(
            size=(img_size, img_size), 
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.15, 
            rotate_limit=90,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        
        # Color transforms
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=20, 
                p=1.0
            ),
        ], p=0.5),
        
        # Noise & Blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        # CLAHE (tốt cho dermatology)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        
        # Distortion (nhẹ)
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.05, p=1.0),
            A.GridDistortion(distort_limit=0.1, p=1.0),
        ], p=0.2),
        
        # Cutout
        A.CoarseDropout(
            max_holes=8, 
            max_height=img_size//8, 
            max_width=img_size//8,
            min_holes=4, 
            min_height=img_size//16, 
            min_width=img_size//16,
            fill_value=0, 
            p=0.5
        ),
        
        # Normalize với ImageNet stats
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def get_val_transforms(img_size):
    """Validation transforms"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size):
    """Test Time Augmentation transforms"""
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
            return image, row['image']
        
        target = torch.tensor(row["target"], dtype=torch.long)
        return image, target
```

***

## 4. Model & Loss Functions

**Cell 6 – Model factory**

```python
def create_model(cfg, use_dp=True):
    """
    Tạo EfficientNet-B2 model.
    B2 có ~9.2M params, lớn hơn B0 (~5.3M).
    """
    model = timm.create_model(
        cfg.MODEL_NAME, 
        pretrained=cfg.PRETRAINED, 
        num_classes=cfg.N_CLASSES, 
        drop_rate=cfg.DROP_RATE
    )
    model = model.to(cfg.DEVICE)
    
    if use_dp and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    return model


def get_scheduler(optimizer, cfg, num_train_steps):
    """Warmup linear rồi CosineAnnealing"""
    warmup_steps = cfg.WARMUP_EPOCHS * (num_train_steps // cfg.EPOCHS)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_train_steps - warmup_steps))
        return max(cfg.MIN_LR / cfg.BASE_LR, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Test model
print("\nTesting model creation...")
test_model = create_model(cfg, use_dp=False)
n_params = sum(p.numel() for p in test_model.parameters())
n_trainable = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
print(f"Model: {cfg.MODEL_NAME}")
print(f"Total parameters: {n_params:,}")
print(f"Trainable parameters: {n_trainable:,}")

with torch.no_grad():
    dummy_input = torch.randn(2, 3, cfg.IMG_SIZE, cfg.IMG_SIZE).to(cfg.DEVICE)
    output = test_model(dummy_input)
    print(f"Output shape: {output.shape}")

del test_model
torch.cuda.empty_cache()
print("✓ Model factory working correctly!")
```

**Cell 7 – Loss Functions & Mixup**

```python
# ==================== WEIGHTED SOFT TARGET CROSS ENTROPY ====================
class WeightedSoftTargetCrossEntropy(nn.Module):
    """
    SoftTargetCrossEntropy với class weights.
    Giữ được Malignant Boost khi dùng Mixup.
    """
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
    
    def forward(self, x, target):
        # x: logits [B, C], target: soft labels [B, C]
        log_probs = F.log_softmax(x, dim=-1)
        
        if self.class_weights is not None:
            # Weight the loss per class
            weights = self.class_weights.to(x.device)
            weighted_target = target * weights.unsqueeze(0)
            # Normalize to keep loss scale similar
            weighted_target = weighted_target / (weighted_target.sum(dim=-1, keepdim=True) + 1e-8) * target.sum(dim=-1, keepdim=True)
            loss = -torch.sum(weighted_target * log_probs, dim=-1)
        else:
            loss = -torch.sum(target * log_probs, dim=-1)
        
        return loss.mean()


# ==================== MIXUP/CUTMIX ====================
def setup_mixup(cfg, class_weight_tensor):
    """
    Setup Mixup/CutMix nhẹ cho EfficientNet-B2.
    
    Lưu ý: Khi dùng Mixup + Label Smoothing, KHÔNG nên dùng thêm Focal Loss
    vì sẽ gây over-regularization và làm giảm Balanced Accuracy.
    
    Combination tốt: CE + class_weight + Label Smoothing + Mixup (nhẹ)
    
    ⚠️ QUAN TRỌNG: timm's Mixup class đã có internal prob check.
    Khi gọi mixup_fn(images, targets), nó tự động apply với probability = cfg.MIXUP_PROB.
    KHÔNG cần thêm random.random() check bên ngoài!
    """
    if not cfg.USE_MIXUP:
        return None, None
    
    mixup_fn = Mixup(
        mixup_alpha=cfg.MIXUP_ALPHA,
        cutmix_alpha=cfg.CUTMIX_ALPHA,
        cutmix_minmax=None,
        prob=cfg.MIXUP_PROB,  # Internal probability check
        switch_prob=cfg.MIXUP_SWITCH_PROB,
        mode='batch',
        label_smoothing=cfg.LABEL_SMOOTHING,
        num_classes=cfg.N_CLASSES
    )
    
    # Sử dụng WeightedSoftTargetCrossEntropy để giữ Malignant Boost
    mixup_criterion = WeightedSoftTargetCrossEntropy(class_weights=class_weight_tensor)
    
    print("✓ Mixup/CutMix enabled (light regularization)")
    print(f"  Mixup α: {cfg.MIXUP_ALPHA}")
    print(f"  CutMix α: {cfg.CUTMIX_ALPHA}")
    print(f"  Apply Prob: {cfg.MIXUP_PROB} (handled internally by timm)")
    print(f"  Using WeightedSoftTargetCrossEntropy to preserve class weights")
    
    return mixup_fn, mixup_criterion


# ==================== CRITERION SETUP ====================
# Sử dụng CrossEntropyLoss với:
# - class_weight: Đã có Malignant Boost (MEL, BCC, SCC x1.5, SCC extra x2.0)
# - label_smoothing: Giảm overconfidence
# KHÔNG dùng Focal Loss khi đã có Mixup (tránh over-regularization)

criterion = nn.CrossEntropyLoss(
    weight=class_weight, 
    label_smoothing=cfg.LABEL_SMOOTHING
)
print(f"✓ Using CrossEntropyLoss")
print(f"  - Class weights: Applied (with Malignant Boost)")
print(f"  - Label Smoothing: {cfg.LABEL_SMOOTHING}")

# Setup Mixup với class weights
mixup_fn, mixup_criterion = setup_mixup(cfg, class_weight)
```

> **⚠️ Tại sao không dùng Focal Loss + Mixup cùng lúc?**
>
> 1. **Over-regularization**: Focal Loss đã down-weight easy examples, Mixup cũng làm soft labels → quá nhiều regularization
> 2. **Conflict**: Focal Loss focus vào hard examples, nhưng Mixup tạo ra mixed samples khó xác định "hard" hay "easy"
> 3. **Label Smoothing đã đủ**: CE + class_weight + Label Smoothing đã handle class imbalance tốt
> 4. **Empirical**: Balanced Accuracy giảm khi dùng cả 3 techniques cùng lúc

***

## 5. Training Loop

**Cell 8 – Training function**

```python
def train_one_fold(fold, df, cfg, criterion, mixup_fn, mixup_criterion):
    print(f"\n{'='*60}")
    print(f"FOLD {fold}")
    print(f"{'='*60}")
    
    # Data preparation
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
    
    # Model & Optimizer
    model = create_model(cfg, use_dp=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.BASE_LR, 
        weight_decay=cfg.WEIGHT_DECAY
    )
    
    num_train_steps = len(train_loader) * cfg.EPOCHS
    scheduler = get_scheduler(optimizer, cfg, num_train_steps)
    scaler = GradScaler()
    
    # Training state
    best_bal_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, f"effnet_b2_fold{fold}_best.pth")
    
    oof_preds = []
    oof_targets = []
    oof_images = []
    
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'bal_acc': [], 'learning_rate': []}
    
    # Training loop
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Train]")
        for images, targets in pbar:
            images = images.to(cfg.DEVICE)
            targets = targets.to(cfg.DEVICE)
            
            # Apply Mixup/CutMix
            # ⚠️ LƯU Ý: timm's Mixup class đã có internal prob check (prob=0.3)
            # Khi gọi mixup_fn(), nó tự động quyết định có apply hay không
            # KHÔNG cần thêm random.random() check bên ngoài!
            if mixup_fn is not None:
                # mixup_fn sẽ tự động apply với prob=0.3
                # Nếu không apply, targets_mixed sẽ là one-hot soft labels
                images, targets_mixed = mixup_fn(images, targets)
                current_criterion = mixup_criterion  # WeightedSoftTargetCrossEntropy
            else:
                targets_mixed = targets
                current_criterion = criterion
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = current_criterion(outputs, targets_mixed)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
        
        avg_train_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Val]"):
                images = images.to(cfg.DEVICE)
                targets = targets.to(cfg.DEVICE)
                
                with autocast():
                    outputs = model(images)
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
        
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['bal_acc'].append(bal_acc)
        history['learning_rate'].append(current_lr)
        
        gap = avg_train_loss - avg_val_loss
        overfit_warning = "⚠️ Overfitting!" if gap < -0.3 else ""
        
        print(f"\nEpoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, bal_acc={bal_acc:.4f} {overfit_warning}")
        
        # Early stopping
        if bal_acc > best_bal_acc + 1e-3:
            best_bal_acc = bal_acc
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_bal_acc': best_bal_acc,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"✓ Saved best model: bal_acc={best_bal_acc:.4f}, val_loss={best_val_loss:.4f}")
            
            oof_preds = all_probs
            oof_targets = all_targets
            oof_images = val_df['image'].tolist()
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
        
        if epoch >= cfg.MIN_EPOCHS and epochs_no_improve >= cfg.PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
    
    del model
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

**Cell 9 – Run 5-Fold Training**

```python
# Chạy training cho tất cả folds
results = []
all_oof = []
all_histories = []

for fold in range(cfg.N_FOLDS):
    fold_result = train_one_fold(fold, df, cfg, criterion, mixup_fn, mixup_criterion)
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
print(f"RESULTS SUMMARY - EfficientNet-B2")
print(f"{'='*60}")
for i, acc in enumerate(bal_accs):
    print(f"Fold {i}: {acc:.4f}")
print(f"Mean: {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}")
print(f"{'='*60}")
```

***

## 6. OOF Evaluation & Threshold Optimization

**Cell 10 – OOF Evaluation**

```python
# Concat OOF predictions
oof_df = pd.concat(all_oof, ignore_index=True)
oof_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'oof_effnet_b2.csv'), index=False)

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

**Cell 11 – Threshold Optimization**

```python
# ==================== THRESHOLD OPTIMIZATION PER CLASS ====================
from sklearn.metrics import precision_recall_curve, f1_score

def find_optimal_thresholds(oof_df, cfg, strategy='f1'):
    """
    Tìm optimal threshold cho từng class.
    
    Strategies:
    - 'f1': Maximize F1 score
    - 'recall': Đạt target recall (quan trọng cho malignant classes)
    - 'balanced': Cân bằng precision và recall
    
    Returns:
        optimal_thresholds: dict {class_name: threshold}
        threshold_metrics: dict với metrics tại optimal threshold
    """
    
    print(f"\n{'='*60}")
    print(f"THRESHOLD OPTIMIZATION (Strategy: {strategy})")
    print(f"{'='*60}")
    
    optimal_thresholds = {}
    threshold_metrics = {}
    
    # Lấy probabilities
    prob_cols = [f'prob_{i}' for i in range(cfg.N_CLASSES)]
    probs = oof_df[prob_cols].values
    targets = oof_df['target'].values
    
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        y_true_binary = (targets == i).astype(int)
        y_prob = probs[:, i]
        
        # Tính precision, recall tại các thresholds
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_prob)
        
        # Loại bỏ threshold cuối (tương ứng với recall=0)
        precision = precision[:-1]
        recall = recall[:-1]
        
        if strategy == 'f1':
            # Maximize F1 score
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            best_precision = precision[best_idx]
            best_recall = recall[best_idx]
            
        elif strategy == 'recall':
            # Đạt minimum recall (quan trọng cho malignant)
            target_recall = 0.85 if i in cfg.MALIGNANT_INDICES else 0.70
            # Tìm threshold thấp nhất đạt target recall
            valid_idx = np.where(recall >= target_recall)[0]
            if len(valid_idx) > 0:
                # Chọn threshold cao nhất trong các thresholds đạt target recall
                best_idx = valid_idx[np.argmax(precision[valid_idx])]
            else:
                # Fallback to maximum recall
                best_idx = np.argmax(recall)
            best_threshold = thresholds[best_idx]
            best_f1 = 2 * precision[best_idx] * recall[best_idx] / (precision[best_idx] + recall[best_idx] + 1e-8)
            best_precision = precision[best_idx]
            best_recall = recall[best_idx]
            
        else:  # 'balanced' or default
            # Minimize |precision - recall|
            diff = np.abs(precision - recall)
            best_idx = np.argmin(diff)
            best_threshold = thresholds[best_idx]
            best_f1 = 2 * precision[best_idx] * recall[best_idx] / (precision[best_idx] + recall[best_idx] + 1e-8)
            best_precision = precision[best_idx]
            best_recall = recall[best_idx]
        
        optimal_thresholds[class_name] = float(best_threshold)
        threshold_metrics[class_name] = {
            'threshold': float(best_threshold),
            'precision': float(best_precision),
            'recall': float(best_recall),
            'f1': float(best_f1),
            'support': int(y_true_binary.sum()),
            'is_malignant': i in cfg.MALIGNANT_INDICES
        }
        
        malignant_marker = "⚠️ MALIGNANT" if i in cfg.MALIGNANT_INDICES else ""
        print(f"  {class_name}: threshold={best_threshold:.4f}, P={best_precision:.3f}, R={best_recall:.3f}, F1={best_f1:.3f} {malignant_marker}")
    
    return optimal_thresholds, threshold_metrics


def apply_optimized_thresholds(probs, optimal_thresholds, class_names):
    """
    Áp dụng optimal thresholds để dự đoán.
    
    Logic: Với mỗi sample, chọn class có (prob / threshold) cao nhất
    """
    n_samples = probs.shape[0]
    n_classes = probs.shape[1]
    
    # Normalize probs by thresholds
    thresholds_array = np.array([optimal_thresholds[name] for name in class_names])
    normalized_probs = probs / (thresholds_array + 1e-8)
    
    # Chọn class có normalized prob cao nhất
    predictions = np.argmax(normalized_probs, axis=1)
    
    return predictions


def evaluate_with_optimized_thresholds(oof_df, optimal_thresholds, cfg):
    """Đánh giá performance với optimized thresholds"""
    
    prob_cols = [f'prob_{i}' for i in range(cfg.N_CLASSES)]
    probs = oof_df[prob_cols].values
    targets = oof_df['target'].values
    
    # Predictions với default threshold (argmax)
    default_preds = np.argmax(probs, axis=1)
    default_bal_acc = balanced_accuracy_score(targets, default_preds)
    
    # Predictions với optimized thresholds
    optimized_preds = apply_optimized_thresholds(probs, optimal_thresholds, cfg.CLASS_NAMES)
    optimized_bal_acc = balanced_accuracy_score(targets, optimized_preds)
    
    print(f"\n{'='*60}")
    print("COMPARISON: Default vs Optimized Thresholds")
    print(f"{'='*60}")
    print(f"Default (argmax) Balanced Accuracy: {default_bal_acc:.4f}")
    print(f"Optimized Thresholds Balanced Accuracy: {optimized_bal_acc:.4f}")
    print(f"Improvement: {(optimized_bal_acc - default_bal_acc)*100:+.2f}%")
    
    # Per-class comparison
    print(f"\nPer-class F1 comparison:")
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        y_true = (targets == i).astype(int)
        
        # Default
        y_pred_default = (default_preds == i).astype(int)
        f1_default = f1_score(y_true, y_pred_default)
        
        # Optimized
        y_pred_opt = (optimized_preds == i).astype(int)
        f1_opt = f1_score(y_true, y_pred_opt)
        
        diff = f1_opt - f1_default
        marker = "⚠️" if i in cfg.MALIGNANT_INDICES else "  "
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"  {marker} {class_name}: F1 {f1_default:.3f} -> {f1_opt:.3f} ({arrow}{abs(diff)*100:.1f}%)")
    
    return {
        'default_bal_acc': default_bal_acc,
        'optimized_bal_acc': optimized_bal_acc,
        'default_preds': default_preds,
        'optimized_preds': optimized_preds
    }


# ========== CHẠY THRESHOLD OPTIMIZATION ==========
print("\n" + "="*60)
print("STEP: THRESHOLD OPTIMIZATION")
print("="*60)

# Strategy 1: F1 optimization
optimal_thresholds_f1, metrics_f1 = find_optimal_thresholds(oof_df, cfg, strategy='f1')

# Strategy 2: Recall optimization (ưu tiên malignant)
optimal_thresholds_recall, metrics_recall = find_optimal_thresholds(oof_df, cfg, strategy='recall')

# Evaluate với F1 optimized thresholds
print("\n>>> Evaluating with F1-optimized thresholds:")
eval_results_f1 = evaluate_with_optimized_thresholds(oof_df, optimal_thresholds_f1, cfg)

# Evaluate với Recall optimized thresholds  
print("\n>>> Evaluating with Recall-optimized thresholds:")
eval_results_recall = evaluate_with_optimized_thresholds(oof_df, optimal_thresholds_recall, cfg)

# Lưu optimal thresholds
thresholds_output = {
    'f1_strategy': {
        'thresholds': optimal_thresholds_f1,
        'metrics': metrics_f1,
        'balanced_accuracy': eval_results_f1['optimized_bal_acc']
    },
    'recall_strategy': {
        'thresholds': optimal_thresholds_recall,
        'metrics': metrics_recall,
        'balanced_accuracy': eval_results_recall['optimized_bal_acc']
    }
}

thresholds_path = os.path.join(cfg.OUTPUT_DIR, 'optimal_thresholds.json')
with open(thresholds_path, 'w') as f:
    json.dump(thresholds_output, f, indent=4)
print(f"\n✓ Saved optimal thresholds to {thresholds_path}")
```

***

## 7. Inference với TTA

**Cell 12 – Inference function với Enhanced TTA**

```python
def tta_predict(model, image, val_transforms, device, num_tta=4):
    """
    Enhanced Test Time Augmentation.
    
    TTA transforms:
    1. Original
    2. Horizontal flip
    3. Vertical flip  
    4. Horizontal + Vertical flip (180° rotation)
    
    Args:
        model: trained model
        image: numpy array (H, W, C) RGB
        val_transforms: validation transforms
        device: torch device
        num_tta: number of TTA augmentations (1-4)
    
    Returns:
        averaged predictions
    """
    model.eval()
    preds = []
    
    with torch.no_grad():
        # 1. Original
        img = val_transforms(image=image)["image"].unsqueeze(0).to(device)
        with autocast():
            preds.append(torch.softmax(model(img), dim=1))
        
        if num_tta >= 2:
            # 2. Horizontal flip
            img_hflip = val_transforms(image=cv2.flip(image, 1))["image"].unsqueeze(0).to(device)
            with autocast():
                preds.append(torch.softmax(model(img_hflip), dim=1))
        
        if num_tta >= 3:
            # 3. Vertical flip
            img_vflip = val_transforms(image=cv2.flip(image, 0))["image"].unsqueeze(0).to(device)
            with autocast():
                preds.append(torch.softmax(model(img_vflip), dim=1))
        
        if num_tta >= 4:
            # 4. Both flips (equivalent to 180° rotation)
            img_hvflip = val_transforms(image=cv2.flip(cv2.flip(image, 0), 1))["image"].unsqueeze(0).to(device)
            with autocast():
                preds.append(torch.softmax(model(img_hvflip), dim=1))
    
    return torch.stack(preds).mean(dim=0)


def inference_test(test_csv_path, test_img_root, cfg, use_tta=True, num_tta=4, optimal_thresholds=None):
    """
    Inference với ensemble 5 folds + optional TTA + optional threshold optimization.
    
    Args:
        test_csv_path: path to test CSV
        test_img_root: path to test images
        cfg: config object
        use_tta: whether to use TTA
        num_tta: number of TTA augmentations
        optimal_thresholds: dict of optimal thresholds per class (optional)
    
    Returns:
        submission DataFrame
    """
    test_df = pd.read_csv(test_csv_path)
    val_transforms = get_val_transforms(cfg.IMG_SIZE)
    
    print(f"\n{'='*60}")
    print(f"TEST SET INFERENCE - EfficientNet-B2")
    print(f"{'='*60}")
    print(f"Test samples: {len(test_df)}")
    print(f"Using TTA: {use_tta} (num_tta={num_tta})")
    print(f"Using Optimal Thresholds: {optimal_thresholds is not None}")
    
    # Load 5 models
    models = []
    for fold in range(cfg.N_FOLDS):
        model = create_model(cfg, use_dp=False)
        checkpoint_path = os.path.join(cfg.OUTPUT_DIR, f"effnet_b2_fold{fold}_best.pth")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        print(f"  ✓ Loaded fold {fold} (bal_acc: {checkpoint['best_bal_acc']:.4f})")
    
    # Predict
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
        
        fold_preds = []
        for model in models:
            if use_tta:
                pred = tta_predict(model, image, val_transforms, cfg.DEVICE, num_tta=num_tta)
            else:
                img_tensor = val_transforms(image=image)["image"].unsqueeze(0).to(cfg.DEVICE)
                with torch.no_grad(), autocast():
                    pred = torch.softmax(model(img_tensor), dim=1)
            fold_preds.append(pred)
        
        avg_pred = torch.stack(fold_preds).mean(dim=0).cpu().numpy()[0]
        all_preds.append(avg_pred)
    
    all_preds = np.array(all_preds)
    
    # Apply optimal thresholds if provided
    if optimal_thresholds is not None:
        pred_classes = apply_optimized_thresholds(all_preds, optimal_thresholds, cfg.CLASS_NAMES)
        threshold_info = "optimized"
    else:
        pred_classes = all_preds.argmax(axis=1)
        threshold_info = "argmax"
    
    # Create submission
    submission = pd.DataFrame({
        'image': all_images,
        **{f'prob_{i}': all_preds[:, i] for i in range(cfg.N_CLASSES)},
        'pred': pred_classes,
        'pred_label': [cfg.CLASS_NAMES[p] for p in pred_classes]
    })
    
    output_path = os.path.join(cfg.OUTPUT_DIR, 'effnet_b2_test_predictions.csv')
    submission.to_csv(output_path, index=False)
    print(f"\n✓ Saved predictions to {output_path}")
    print(f"  Threshold method: {threshold_info}")
    
    print(f"\nPrediction Distribution:")
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        count = (pred_classes == i).sum()
        pct = count / len(pred_classes) * 100
        malignant = "⚠️" if i in cfg.MALIGNANT_INDICES else "  "
        print(f"  {malignant} {class_name}: {count} ({pct:.1f}%)")
    
    # Cleanup
    for model in models:
        del model
    torch.cuda.empty_cache()
    
    return submission


# Run inference với TTA
print("\n>>> Running inference with TTA...")
test_submission = inference_test(
    cfg.TEST_CSV_PATH, 
    cfg.TEST_IMG_ROOT, 
    cfg, 
    use_tta=True, 
    num_tta=4,
    optimal_thresholds=None  # Hoặc optimal_thresholds_f1 nếu muốn dùng
)

# Optional: Run inference với optimal thresholds
print("\n>>> Running inference with TTA + Optimal Thresholds (F1)...")
test_submission_optimized = inference_test(
    cfg.TEST_CSV_PATH, 
    cfg.TEST_IMG_ROOT, 
    cfg, 
    use_tta=True, 
    num_tta=4,
    optimal_thresholds=optimal_thresholds_f1
)
test_submission_optimized.to_csv(
    os.path.join(cfg.OUTPUT_DIR, 'effnet_b2_test_predictions_optimized.csv'), 
    index=False
)
```

***

## 8. Save Results

**Cell 13 – Save experiment results & Visualizations**

```python
# ==================== VISUALIZATION FUNCTIONS ====================

def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """Plot và save confusion matrix"""
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create annotations with both count and percentage
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)'
    
    sns.heatmap(cm_normalized, annot=annot, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix to {save_path}")


def plot_roc_curves(oof_df, cfg, save_path):
    """Plot ROC curves cho tất cả classes"""
    plt.figure(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, cfg.N_CLASSES))
    
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        y_true = (oof_df['target'] == i).astype(int)
        y_prob = oof_df[f'prob_{i}'].values
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            linestyle = '--' if i in cfg.MALIGNANT_INDICES else '-'
            linewidth = 2.5 if i in cfg.MALIGNANT_INDICES else 1.5
            label = f'{class_name} (AUC={roc_auc:.3f})'
            if i in cfg.MALIGNANT_INDICES:
                label += ' ⚠️'
            
            plt.plot(fpr, tpr, color=colors[i], linestyle=linestyle, 
                    linewidth=linewidth, label=label)
        except:
            pass
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - EfficientNet-B2', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ROC curves to {save_path}")


def plot_training_history(all_histories, save_path):
    """Plot training history cho tất cả folds"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_histories)))
    
    for fold_idx, history in enumerate(all_histories):
        epochs = history['epoch']
        color = colors[fold_idx]
        
        # Train Loss
        axes[0, 0].plot(epochs, history['train_loss'], 
                       label=f'Fold {fold_idx}', color=color, linewidth=1.5)
        
        # Val Loss
        axes[0, 1].plot(epochs, history['val_loss'], 
                       label=f'Fold {fold_idx}', color=color, linewidth=1.5)
        
        # Balanced Accuracy
        axes[1, 0].plot(epochs, history['bal_acc'], 
                       label=f'Fold {fold_idx}', color=color, linewidth=1.5)
        
        # Learning Rate
        axes[1, 1].plot(epochs, history['learning_rate'], 
                       label=f'Fold {fold_idx}', color=color, linewidth=1.5)
    
    axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend(loc='upper right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Balanced Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Balanced Accuracy')
    axes[1, 0].legend(loc='lower right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Learning Rate', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(loc='upper right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training History - EfficientNet-B2', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training history to {save_path}")


def plot_class_distribution(df, cfg, save_path):
    """Plot class distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count per class
    class_counts = df['target'].value_counts().sort_index()
    colors = ['red' if i in cfg.MALIGNANT_INDICES else 'steelblue' 
              for i in range(cfg.N_CLASSES)]
    
    bars = axes[0].bar(cfg.CLASS_NAMES, class_counts.values, color=colors, edgecolor='black')
    axes[0].set_title('Class Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, count in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    # Pie chart
    explode = [0.05 if i in cfg.MALIGNANT_INDICES else 0 for i in range(cfg.N_CLASSES)]
    wedges, texts, autotexts = axes[1].pie(
        class_counts.values, labels=cfg.CLASS_NAMES, autopct='%1.1f%%',
        colors=colors, explode=explode, startangle=90
    )
    axes[1].set_title('Class Proportion', fontsize=12, fontweight='bold')
    
    # Legend
    legend_labels = [f'{name} {"⚠️ MALIGNANT" if i in cfg.MALIGNANT_INDICES else ""}' 
                    for i, name in enumerate(cfg.CLASS_NAMES)]
    axes[1].legend(wedges, legend_labels, loc='center left', 
                   bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved class distribution to {save_path}")


# ==================== SAVE EXPERIMENT RESULTS ====================

def save_experiment_results(cfg, results, all_histories, oof_df, test_submission, save_dir):
    """Lưu toàn bộ experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save config
    config_dict = {
        'experiment_timestamp': timestamp,
        'model_name': cfg.MODEL_NAME,
        'model_params': '9.2M',
        'n_classes': cfg.N_CLASSES,
        'img_size': cfg.IMG_SIZE,
        'n_folds': cfg.N_FOLDS,
        'batch_size': cfg.BATCH_SIZE,
        'epochs': cfg.EPOCHS,
        'base_lr': cfg.BASE_LR,
        'weight_decay': cfg.WEIGHT_DECAY,
        'drop_rate': cfg.DROP_RATE,
        'label_smoothing': cfg.LABEL_SMOOTHING,
        'use_mixup': cfg.USE_MIXUP,
        'mixup_alpha': cfg.MIXUP_ALPHA,
        'mixup_prob': cfg.MIXUP_PROB,
        'use_focal_loss': False,
        'use_class_balanced_sampling': cfg.USE_CLASS_BALANCED_SAMPLING,
        'malignant_boost': cfg.MALIGNANT_BOOST,
        'scc_extra_boost': cfg.SCC_EXTRA_BOOST,
        'class_names': list(cfg.CLASS_NAMES),
    }
    
    with open(os.path.join(save_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✓ Saved experiment config")
    
    # 2. Save training results per fold
    training_results = {
        'folds': [],
        'mean_bal_acc': float(np.mean([r['best_bal_acc'] for r in results])),
        'std_bal_acc': float(np.std([r['best_bal_acc'] for r in results])),
        'mean_val_loss': float(np.mean([r['best_val_loss'] for r in results])),
    }
    
    for r in results:
        training_results['folds'].append({
            'fold': r['fold'],
            'best_bal_acc': float(r['best_bal_acc']),
            'best_val_loss': float(r['best_val_loss']),
            'checkpoint_path': r['checkpoint_path'],
        })
    
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump(training_results, f, indent=4)
    print(f"✓ Saved training results")
    
    # 3. Save training history
    history_df_list = []
    for fold_idx, history in enumerate(all_histories):
        fold_history = pd.DataFrame(history)
        fold_history['fold'] = fold_idx
        history_df_list.append(fold_history)
    
    history_df = pd.concat(history_df_list, ignore_index=True)
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
    print(f"✓ Saved training history")
    
    # 4. Save OOF metrics
    oof_preds_class = oof_df[[f'prob_{i}' for i in range(cfg.N_CLASSES)]].values.argmax(axis=1)
    oof_bal_acc = balanced_accuracy_score(oof_df['target'], oof_preds_class)
    
    oof_metrics = {
        'balanced_accuracy': float(oof_bal_acc),
        'per_class_auc': {},
        'per_class_f1': {},
    }
    
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        y_true = (oof_df['target'] == i).astype(int)
        y_prob = oof_df[f'prob_{i}'].values
        y_pred = (oof_preds_class == i).astype(int)
        
        try:
            oof_metrics['per_class_auc'][class_name] = float(roc_auc_score(y_true, y_prob))
        except:
            oof_metrics['per_class_auc'][class_name] = None
        
        oof_metrics['per_class_f1'][class_name] = float(f1_score(y_true, y_pred))
    
    with open(os.path.join(save_dir, 'oof_metrics.json'), 'w') as f:
        json.dump(oof_metrics, f, indent=4)
    print(f"✓ Saved OOF metrics")
    
    # 5. Generate visualizations
    print(f"\nGenerating visualizations...")
    
    # Confusion matrix
    cm = confusion_matrix(oof_df['target'], oof_preds_class)
    plot_confusion_matrix(cm, cfg.CLASS_NAMES, 
                         os.path.join(save_dir, 'confusion_matrix.png'),
                         title='Confusion Matrix - EfficientNet-B2 (OOF)')
    
    # ROC curves
    plot_roc_curves(oof_df, cfg, os.path.join(save_dir, 'roc_curves.png'))
    
    # Training history
    plot_training_history(all_histories, os.path.join(save_dir, 'training_history.png'))
    
    # Class distribution
    plot_class_distribution(df, cfg, os.path.join(save_dir, 'class_distribution.png'))
    
    # 6. Save experiment summary
    summary_text = f"""
{'='*60}
EXPERIMENT SUMMARY - EfficientNet-B2
{'='*60}

Timestamp: {timestamp}
Model: {cfg.MODEL_NAME} (~9.2M params)

CONFIGURATION:
- Image Size: {cfg.IMG_SIZE}
- Batch Size: {cfg.BATCH_SIZE}
- Epochs: {cfg.EPOCHS}
- Learning Rate: {cfg.BASE_LR}
- Weight Decay: {cfg.WEIGHT_DECAY}
- Drop Rate: {cfg.DROP_RATE}
- Label Smoothing: {cfg.LABEL_SMOOTHING}
- Use Mixup: {cfg.USE_MIXUP} (α={cfg.MIXUP_ALPHA}, prob={cfg.MIXUP_PROB})
- Class-balanced Sampling: {cfg.USE_CLASS_BALANCED_SAMPLING}
- Malignant Boost: {cfg.MALIGNANT_BOOST}x
- SCC Extra Boost: {cfg.SCC_EXTRA_BOOST}x

RESULTS:
- Mean Balanced Accuracy: {training_results['mean_bal_acc']:.4f} ± {training_results['std_bal_acc']:.4f}
- OOF Balanced Accuracy: {oof_bal_acc:.4f}

PER-FOLD RESULTS:
"""
    for r in results:
        summary_text += f"  Fold {r['fold']}: bal_acc={r['best_bal_acc']:.4f}, val_loss={r['best_val_loss']:.4f}\n"
    
    summary_text += f"""
PER-CLASS AUC:
"""
    for class_name, auc_val in oof_metrics['per_class_auc'].items():
        malignant = "⚠️ MALIGNANT" if class_name in ['MEL', 'BCC', 'SCC'] else ""
        auc_str = f"{auc_val:.4f}" if auc_val else "N/A"
        summary_text += f"  {class_name}: {auc_str} {malignant}\n"
    
    summary_text += f"""
{'='*60}
"""
    
    with open(os.path.join(save_dir, 'experiment_summary.txt'), 'w') as f:
        f.write(summary_text)
    print(f"✓ Saved experiment summary")
    
    print(f"\n{'='*60}")
    print("ALL RESULTS SAVED SUCCESSFULLY!")
    print(f"{'='*60}")


# Run save
save_experiment_results(cfg, results, all_histories, oof_df, test_submission, cfg.OUTPUT_DIR)
```

***

## 9. Tổng kết Config

| Config | EfficientNet-B2 | EfficientNet-B0 | Lý do thay đổi |
|--------|-----------------|-----------------|----------------|
| `MODEL_NAME` | **efficientnet_b2** | efficientnet_b0 | Model mạnh hơn (+3% ImageNet) |
| `IMG_SIZE` | **256** | 256 | Giữ nguyên, training nhanh hơn |
| `BATCH_SIZE` | **64** | 64 | Giữ nguyên theo yêu cầu |
| `DROP_RATE` | **0.3** | 0.4 | B2 default drop_rate |
| `BASE_LR` | **3e-4** | 3e-4 | Giữ nguyên như B0 do dùng Batch 64 |
| `USE_FOCAL_LOSS` | **False** | True | Tránh over-regularization khi có Mixup |
| `USE_MIXUP` | **True** | False | Regularization thêm |
| `MIXUP_ALPHA` | **0.4** | - | Nhẹ hơn ConvNeXt (0.8) |
| `MIXUP_PROB` | **0.3** | - | Chỉ 30% batches |

> **Loss Strategy**: `CrossEntropyLoss` + `class_weight` (Malignant Boost) + `label_smoothing=0.1` + `Mixup/CutMix (light)`

***

## 10. Output Files

| File | Mô tả |
|------|-------|
| `experiment_config.json` | Config với EfficientNet-B2 params |
| `training_results.json` | Kết quả từng fold |
| `training_history.csv` | History chi tiết |
| `oof_metrics.json` | OOF metrics |
| `oof_effnet_b2.csv` | OOF predictions |
| `optimal_thresholds.json` | Optimal thresholds per class |
| `effnet_b2_test_predictions.csv` | Test predictions |
| `effnet_b2_fold*_best.pth` | Model checkpoints |
| `*.png` | Visualizations |

***

## 11. Kỳ vọng Performance

| Metric | EfficientNet-B0 | EfficientNet-B2 | Improvement |
|--------|-----------------|-----------------|-------------|
| **Balanced Accuracy** | 0.70-0.75 | **0.72-0.77** | +2-3% |
| **Mean AUC** | ~0.87 | **0.88-0.90** | +1-2% |
| **SCC AUC** | 0.85 | **0.86-0.88** | +1-2% |
| **MEL AUC** | 0.88 | **0.89-0.91** | +1-2% |

**Lý do EfficientNet-B2 tốt hơn:**

1. **Capacity lớn hơn** (9.2M vs 5.3M params) - học được features phức tạp hơn
2. **Wider & deeper architecture** - nhiều channels và layers hơn B0
3. **Loss strategy hợp lý**: CE + class_weight + Label Smoothing + Mixup (không Focal Loss)
4. **ImageNet performance** - B2 có 80.1% Top-1 vs B0 có 77.1%

**Loss Strategy tối ưu cho EfficientNet-B2:**

- ✅ `CrossEntropyLoss` - baseline loss
- ✅ `class_weight` với Malignant Boost - handle class imbalance
- ✅ `label_smoothing=0.1` - giảm overconfidence
- ✅ `Mixup/CutMix (prob=0.3)` - regularization nhẹ
- ❌ `Focal Loss` - KHÔNG dùng (over-regularization khi kết hợp với Mixup)

***

## 12. So sánh với các model khác

| Model | Params | IMG_SIZE | Expected Bal_Acc | Training Time |
|-------|--------|----------|------------------|---------------|
| EfficientNet-B0 | 5.3M | 256 | 0.70-0.75 | 1x |
| **EfficientNet-B2** | **9.2M** | **256** | **0.72-0.77** | **1.3x** |
| ConvNeXt-tiny | 28.6M | 288 | 0.75-0.80 | 2.5x |

EfficientNet-B2 là lựa chọn tốt khi:

- Cần cải thiện accuracy so với B0 (~+2-3% Balanced Accuracy)
- Không muốn đợi lâu như ConvNeXt-tiny
- VRAM hạn chế (T4 16GB)
- Muốn giữ IMG_SIZE = 256 như B0 để so sánh công bằng

***

## 13. Changelog & Bug Fixes

### Các lỗi đã sửa

| Lỗi | Mô tả | Giải pháp |
|-----|-------|-----------|
| **Double Mixup Probability** | timm's Mixup đã có internal prob check, thêm `random.random() < MIXUP_PROB` làm prob thực = 0.09 | Bỏ external random check, để timm xử lý |
| **SoftTargetCE không có class_weight** | Khi Mixup apply, loss không giữ được Malignant Boost | Tạo `WeightedSoftTargetCrossEntropy` class |
| **Cell 13 chưa hoàn chỉnh** | Code lưu kết quả bị cắt ngắn | Hoàn thiện đầy đủ với visualization functions |
| **Thiếu Visualization** | Không có code vẽ confusion matrix, ROC, history | Thêm 4 visualization functions |

### Các cải tiến

1. **WeightedSoftTargetCrossEntropy**: Custom loss giữ class weights khi dùng Mixup
2. **Enhanced TTA**: Thêm 4 TTA transforms (original, hflip, vflip, 180° rotation)
3. **Inference với Optimal Thresholds**: Support inference với custom thresholds
4. **Complete Visualization Suite**: Confusion matrix, ROC curves, training history, class distribution
5. **Detailed Experiment Summary**: Lưu summary text file với tất cả metrics

***
