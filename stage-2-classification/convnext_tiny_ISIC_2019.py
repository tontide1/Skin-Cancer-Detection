"""
ConvNeXt-tiny for ISIC 2019 Skin Cancer Classification
=======================================================
Optimized implementation with:
- Layer-wise Learning Rate Decay (LLRD)
- Mixup/CutMix (timm.data.Mixup)
- EMA (timm.utils.ModelEmaV2)
- Focal Loss + Class-balanced Sampling
- Malignant Class Boost (MEL, BCC, SCC)
- Gradient Accumulation
- Test Time Augmentation (TTA)
- Threshold Optimization per class

Author: Generated from plan-implementation-ConvNeXt-tiny.md
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import cv2
import json
import math
import random
import glob
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
from timm.utils import ModelEmaV2

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


# ============================================================================
# REPRODUCIBILITY
# ============================================================================
def seed_everything(seed=42):
    """Set seeds for reproducibility"""
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


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    # ==================== PATHS ====================
    CSV_PATH: str = "/kaggle/input/isic-2019-task-1/ISIC_2019_5folds_metadata.csv"
    IMG_ROOT: str = "/kaggle/input/isic-2019-task-1/cropped_lesions/cropped_lesions"
    OUTPUT_DIR: str = "/kaggle/working"
    TEST_CSV_PATH: str = "/kaggle/input/isic-2019-task-1/ISIC_2019_test_metadata.csv"
    TEST_IMG_ROOT: str = "/kaggle/input/isic-2019-task-1/cropped_lesions_testset/cropped_lesions_testset"
    
    # ==================== MODEL ====================
    MODEL_NAME: str = "convnext_tiny.fb_in22k_ft_in1k"
    N_CLASSES: int = 8
    PRETRAINED: bool = True
    DROP_PATH_RATE: float = 0.2
    
    # ==================== TRAINING ====================
    IMG_SIZE: int = 288
    N_FOLDS: int = 5
    BATCH_SIZE: int = 32
    ACCUMULATION_STEPS: int = 2
    EPOCHS: int = 100
    MIN_EPOCHS: int = 10
    PATIENCE: int = 10
    
    # ==================== OPTIMIZER ====================
    BASE_LR: float = 5e-5
    MIN_LR: float = 1e-7
    WEIGHT_DECAY: float = 0.05
    WARMUP_EPOCHS: int = 3
    
    # Layer-wise Learning Rate Decay
    USE_LLRD: bool = True
    LLRD_DECAY: float = 0.75
    
    # ==================== REGULARIZATION ====================
    LABEL_SMOOTHING: float = 0.1
    
    # Mixup & CutMix
    USE_MIXUP: bool = True
    MIXUP_ALPHA: float = 0.8
    CUTMIX_ALPHA: float = 1.0
    MIXUP_PROB: float = 0.5
    CUTMIX_PROB: float = 0.5
    MIXUP_SWITCH_PROB: float = 0.5
    
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
    MALIGNANT_INDICES: tuple = (0, 2, 7)

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


# ============================================================================
# DATA LOADING & CLASS WEIGHTS
# ============================================================================
df = pd.read_csv(cfg.CSV_PATH)
print(f"\nDataset size: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nClass distribution:")
print(df['target'].value_counts().sort_index())

# Tính class weight với Malignant Boost
counts = df["target"].value_counts().sort_index().values
N = len(df)
weights = N / counts
weights = weights / weights.mean()

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

print(f"\nFold distribution:")
print(df['fold'].value_counts().sort_index())


# ============================================================================
# AUGMENTATION
# ============================================================================
def get_train_transforms(img_size):
    """Strong augmentation for ConvNeXt"""
    return A.Compose([
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.6, 1.0),
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
            rotate_limit=180,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.4),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.1, p=1.0),
            A.GridDistortion(distort_limit=0.2, p=1.0),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
        ], p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
        A.OneOf([
            A.CoarseDropout(
                max_holes=12, max_height=img_size//6, max_width=img_size//6,
                min_holes=4, min_height=img_size//12, min_width=img_size//12,
                fill_value=0, p=1.0
            ),
            A.GridDropout(ratio=0.3, unit_size_min=img_size//8, unit_size_max=img_size//4, p=1.0),
        ], p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size):
    """Validation transforms"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
        A.Compose(base_transforms),
        A.Compose([A.HorizontalFlip(p=1.0)] + base_transforms),
        A.Compose([A.VerticalFlip(p=1.0)] + base_transforms),
        A.Compose([A.Transpose(p=1.0)] + base_transforms),
        A.Compose([A.Rotate(limit=(90, 90), p=1.0)] + base_transforms),
        A.Compose([A.Rotate(limit=(180, 180), p=1.0)] + base_transforms),
        A.Compose([A.Rotate(limit=(270, 270), p=1.0)] + base_transforms),
    ]
    
    return tta_list


# ============================================================================
# DATASET
# ============================================================================
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


# ============================================================================
# MODEL
# ============================================================================
def create_model(cfg):
    """Create ConvNeXt-tiny model"""
    model = timm.create_model(
        cfg.MODEL_NAME,
        pretrained=cfg.PRETRAINED,
        num_classes=cfg.N_CLASSES,
        drop_path_rate=cfg.DROP_PATH_RATE,
    )
    return model.to(cfg.DEVICE)


def get_optimizer_params_with_llrd(model, cfg):
    """
    Layer-wise Learning Rate Decay (LLRD).
    LR giảm dần từ head xuống backbone.
    """
    if not cfg.USE_LLRD:
        return [{'params': model.parameters(), 'lr': cfg.BASE_LR, 'weight_decay': cfg.WEIGHT_DECAY}]
    
    layer_names = [
        'head', 'norm', 'stages.3', 'stages.2', 
        'stages.1', 'stages.0', 'stem', 'downsample_layers',
    ]
    
    lr_scales = [cfg.LLRD_DECAY ** i for i in range(len(layer_names))]
    
    param_groups = []
    assigned_params = set()
    
    for layer_name, lr_scale in zip(layer_names, lr_scales):
        group_params = []
        group_params_no_decay = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if layer_name in name and id(param) not in assigned_params:
                assigned_params.add(id(param))
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
    
    # Remaining params
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


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
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


def setup_mixup(cfg):
    """Setup Mixup/CutMix using timm.data.Mixup"""
    if not cfg.USE_MIXUP:
        return None, None
    
    mixup_fn = Mixup(
        mixup_alpha=cfg.MIXUP_ALPHA,
        cutmix_alpha=cfg.CUTMIX_ALPHA,
        cutmix_minmax=None,
        prob=cfg.MIXUP_PROB,
        switch_prob=cfg.MIXUP_SWITCH_PROB,
        mode='batch',
        label_smoothing=cfg.LABEL_SMOOTHING,
        num_classes=cfg.N_CLASSES
    )
    
    mixup_criterion = SoftTargetCrossEntropy()
    
    print("✓ Mixup/CutMix enabled (timm.data.Mixup)")
    print(f"  Mixup α: {cfg.MIXUP_ALPHA}")
    print(f"  CutMix α: {cfg.CUTMIX_ALPHA}")
    print(f"  Apply Prob: {cfg.MIXUP_PROB}")
    print(f"  Switch Prob: {cfg.MIXUP_SWITCH_PROB}")
    
    return mixup_fn, mixup_criterion


# Setup criterion
if cfg.USE_FOCAL_LOSS:
    criterion = FocalLoss(
        alpha=class_weight,
        gamma=cfg.FOCAL_GAMMA,
        label_smoothing=cfg.LABEL_SMOOTHING,
        reduction='mean'
    )
    print(f"✓ Using Focal Loss (γ={cfg.FOCAL_GAMMA})")
else:
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=cfg.LABEL_SMOOTHING)
    print(f"✓ Using CrossEntropyLoss")

mixup_fn, mixup_criterion = setup_mixup(cfg)


# ============================================================================
# SCHEDULER
# ============================================================================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    """Cosine schedule with linear warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
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
    model = create_model(cfg)
    
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    param_groups = get_optimizer_params_with_llrd(base_model, cfg)
    
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.BASE_LR, betas=(0.9, 0.999))
    
    # Scheduler
    num_training_steps = (len(train_loader) // cfg.ACCUMULATION_STEPS) * cfg.EPOCHS
    num_warmup_steps = (len(train_loader) // cfg.ACCUMULATION_STEPS) * cfg.WARMUP_EPOCHS
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps,
        min_lr_ratio=cfg.MIN_LR / cfg.BASE_LR
    )
    
    scaler = GradScaler()
    
    # EMA
    if cfg.USE_EMA:
        ema = ModelEmaV2(model, decay=cfg.EMA_DECAY, device=cfg.DEVICE)
        print(f"✓ Using EMA with decay={cfg.EMA_DECAY}")
    else:
        ema = None
    
    # Training state
    best_bal_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, f"convnext_tiny_fold{fold}_best.pth")
    
    oof_preds = []
    oof_targets = []
    oof_images = []
    
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'bal_acc': [], 'learning_rate': []}
    
    # Training loop
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Train]")
        for step, (images, targets) in enumerate(pbar):
            images = images.to(cfg.DEVICE)
            targets = targets.to(cfg.DEVICE)
            
            if mixup_fn is not None:
                images, targets_mixed = mixup_fn(images, targets)
                current_criterion = mixup_criterion
            else:
                targets_mixed = targets
                current_criterion = criterion
            
            with autocast():
                outputs = model(images)
                loss = current_criterion(outputs, targets_mixed)
                loss = loss / cfg.ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % cfg.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                if ema is not None:
                    ema.update(model)
            
            train_loss += loss.item() * cfg.ACCUMULATION_STEPS
            pbar.set_postfix({'loss': f"{loss.item() * cfg.ACCUMULATION_STEPS:.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
        
        avg_train_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation
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
        
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['bal_acc'].append(bal_acc)
        history['learning_rate'].append(current_lr)
        
        gap = avg_train_loss - avg_val_loss
        overfit_warning = "⚠️ Overfitting!" if gap < -0.3 else ""
        ema_tag = "[EMA]" if ema is not None else ""
        
        print(f"\nEpoch {epoch} {ema_tag}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, bal_acc={bal_acc:.4f} {overfit_warning}")
        
        # Early stopping
        if bal_acc > best_bal_acc + 1e-3:
            best_bal_acc = bal_acc
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
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
                'config': {'model_name': cfg.MODEL_NAME, 'img_size': cfg.IMG_SIZE, 'drop_path_rate': cfg.DROP_PATH_RATE}
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


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================
def find_optimal_thresholds(oof_df, cfg, strategy='f1'):
    """Find optimal threshold per class"""
    print(f"\n{'='*60}")
    print(f"THRESHOLD OPTIMIZATION (Strategy: {strategy})")
    print(f"{'='*60}")
    
    optimal_thresholds = {}
    threshold_metrics = {}
    
    prob_cols = [f'prob_{i}' for i in range(cfg.N_CLASSES)]
    probs = oof_df[prob_cols].values
    targets = oof_df['target'].values
    
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        y_true_binary = (targets == i).astype(int)
        y_prob = probs[:, i]
        
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_prob)
        precision = precision[:-1]
        recall = recall[:-1]
        
        if len(thresholds) == 0:
            optimal_thresholds[class_name] = 0.5
            threshold_metrics[class_name] = {'threshold': 0.5, 'precision': 0, 'recall': 0, 'f1': 0, 'support': int(y_true_binary.sum()), 'is_malignant': i in cfg.MALIGNANT_INDICES}
            continue
        
        if strategy == 'f1':
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            best_precision = precision[best_idx]
            best_recall = recall[best_idx]
        elif strategy == 'recall':
            target_recall = 0.85 if i in cfg.MALIGNANT_INDICES else 0.70
            valid_idx = np.where(recall >= target_recall)[0]
            if len(valid_idx) > 0:
                best_idx = valid_idx[np.argmax(precision[valid_idx])]
            else:
                best_idx = np.argmax(recall)
            best_threshold = thresholds[best_idx]
            best_f1 = 2 * precision[best_idx] * recall[best_idx] / (precision[best_idx] + recall[best_idx] + 1e-8)
            best_precision = precision[best_idx]
            best_recall = recall[best_idx]
        else:
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
    """Apply optimal thresholds for prediction"""
    thresholds_array = np.array([optimal_thresholds[name] for name in class_names])
    normalized_probs = probs / (thresholds_array + 1e-8)
    predictions = np.argmax(normalized_probs, axis=1)
    return predictions


def evaluate_with_optimized_thresholds(oof_df, optimal_thresholds, cfg):
    """Evaluate performance with optimized thresholds"""
    prob_cols = [f'prob_{i}' for i in range(cfg.N_CLASSES)]
    probs = oof_df[prob_cols].values
    targets = oof_df['target'].values
    
    default_preds = np.argmax(probs, axis=1)
    default_bal_acc = balanced_accuracy_score(targets, default_preds)
    
    optimized_preds = apply_optimized_thresholds(probs, optimal_thresholds, cfg.CLASS_NAMES)
    optimized_bal_acc = balanced_accuracy_score(targets, optimized_preds)
    
    print(f"\n{'='*60}")
    print("COMPARISON: Default vs Optimized Thresholds")
    print(f"{'='*60}")
    print(f"Default (argmax) Balanced Accuracy: {default_bal_acc:.4f}")
    print(f"Optimized Thresholds Balanced Accuracy: {optimized_bal_acc:.4f}")
    print(f"Improvement: {(optimized_bal_acc - default_bal_acc)*100:+.2f}%")
    
    return {
        'default_bal_acc': default_bal_acc,
        'optimized_bal_acc': optimized_bal_acc,
        'default_preds': default_preds,
        'optimized_preds': optimized_preds
    }


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Counts)')
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved confusion matrix plot to {save_path}")
    plt.show()


def plot_roc_curves(oof_df, cfg, save_path=None):
    """Plot ROC curves"""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, cfg.N_CLASSES))
    
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        y_true_binary = (oof_df['target'] == i).astype(int)
        y_prob = oof_df[f'prob_{i}'].values
        
        if y_true_binary.sum() > 0:
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
            roc_auc = auc(fpr, tpr)
            marker = '⚠️' if class_name in ['MEL', 'BCC', 'SCC'] else ''
            ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'{class_name} {marker}(AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved ROC curves plot to {save_path}")
    plt.show()


def plot_training_history(all_histories, cfg, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for fold, history in enumerate(all_histories):
        axes[0, 0].plot(history['epoch'], history['train_loss'], label=f'Fold {fold}')
        axes[0, 1].plot(history['epoch'], history['val_loss'], label=f'Fold {fold}')
        axes[1, 0].plot(history['epoch'], history['bal_acc'], label=f'Fold {fold}')
        axes[1, 1].plot(history['epoch'], history['learning_rate'], label=f'Fold {fold}')
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Balanced Accuracy')
    axes[1, 0].set_title('Validation Balanced Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training history plot to {save_path}")
    plt.show()


# ============================================================================
# INFERENCE
# ============================================================================
def inference_with_tta(test_csv_path, test_img_root, cfg, n_tta=7):
    """Inference with 5-fold ensemble + TTA"""
    test_df = pd.read_csv(test_csv_path)
    tta_transforms = get_tta_transforms(cfg.IMG_SIZE)
    
    print(f"\n{'='*60}")
    print(f"TEST SET INFERENCE - ConvNeXt-tiny")
    print(f"{'='*60}")
    print(f"Test samples: {len(test_df)}")
    print(f"TTA augmentations: {n_tta}")
    
    models = []
    for fold in range(cfg.N_FOLDS):
        model = create_model(cfg)
        checkpoint_path = os.path.join(cfg.OUTPUT_DIR, f"convnext_tiny_fold{fold}_best.pth")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        print(f"  ✓ Loaded fold {fold} (bal_acc: {checkpoint['best_bal_acc']:.4f})")
    
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
        
        ensemble_preds = []
        for model in models:
            for tta_transform in tta_transforms[:n_tta]:
                img_tensor = tta_transform(image=image)["image"].unsqueeze(0).to(cfg.DEVICE)
                with torch.no_grad(), autocast():
                    pred = torch.softmax(model(img_tensor), dim=1)
                ensemble_preds.append(pred)
        
        avg_pred = torch.stack(ensemble_preds).mean(dim=0).cpu().numpy()[0]
        all_preds.append(avg_pred)
    
    all_preds = np.array(all_preds)
    pred_classes = all_preds.argmax(axis=1)
    
    submission = pd.DataFrame({
        'image': all_images,
        **{f'prob_{i}': all_preds[:, i] for i in range(cfg.N_CLASSES)},
        'pred': pred_classes,
        'pred_label': [cfg.CLASS_NAMES[p] for p in pred_classes]
    })
    
    output_path = os.path.join(cfg.OUTPUT_DIR, 'convnext_tiny_test_predictions.csv')
    submission.to_csv(output_path, index=False)
    print(f"\n✓ Saved predictions to {output_path}")
    
    print(f"\nPrediction Distribution:")
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        count = (pred_classes == i).sum()
        pct = count / len(pred_classes) * 100
        print(f"  {class_name}: {count} ({pct:.1f}%)")
    
    for model in models:
        del model
    torch.cuda.empty_cache()
    
    return submission


# ============================================================================
# SAVE RESULTS
# ============================================================================
def save_experiment_results(cfg, results, all_histories, oof_df, test_submission, save_dir):
    """Save all experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Config
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
        'min_epochs': cfg.MIN_EPOCHS,
        'patience': cfg.PATIENCE,
        'base_lr': cfg.BASE_LR,
        'min_lr': cfg.MIN_LR,
        'weight_decay': cfg.WEIGHT_DECAY,
        'warmup_epochs': cfg.WARMUP_EPOCHS,
        'drop_path_rate': cfg.DROP_PATH_RATE,
        'label_smoothing': cfg.LABEL_SMOOTHING,
        'use_llrd': cfg.USE_LLRD,
        'llrd_decay': cfg.LLRD_DECAY,
        'use_mixup': cfg.USE_MIXUP,
        'mixup_alpha': cfg.MIXUP_ALPHA,
        'cutmix_alpha': cfg.CUTMIX_ALPHA,
        'use_ema': cfg.USE_EMA,
        'ema_decay': cfg.EMA_DECAY,
        'use_focal_loss': cfg.USE_FOCAL_LOSS,
        'focal_gamma': cfg.FOCAL_GAMMA,
        'use_class_balanced_sampling': cfg.USE_CLASS_BALANCED_SAMPLING,
        'malignant_boost': cfg.MALIGNANT_BOOST,
        'scc_extra_boost': cfg.SCC_EXTRA_BOOST,
        'malignant_indices': list(cfg.MALIGNANT_INDICES),
        'class_names': list(cfg.CLASS_NAMES),
    }
    
    with open(os.path.join(save_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✓ Saved config")
    
    # Training results
    training_results = {
        'timestamp': timestamp,
        'fold_results': [{'fold': r['fold'], 'best_bal_acc': float(r['best_bal_acc']), 'best_val_loss': float(r['best_val_loss']), 'n_epochs_trained': len(r['history']['epoch'])} for r in results],
        'summary': {
            'mean_bal_acc': float(np.mean([r['best_bal_acc'] for r in results])),
            'std_bal_acc': float(np.std([r['best_bal_acc'] for r in results])),
            'best_fold': int(np.argmax([r['best_bal_acc'] for r in results])),
            'best_bal_acc': float(max([r['best_bal_acc'] for r in results])),
        }
    }
    
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump(training_results, f, indent=4)
    print(f"✓ Saved training results")
    
    # Training history
    all_history_dfs = []
    for fold, history in enumerate(all_histories):
        history_df = pd.DataFrame(history)
        history_df['fold'] = fold
        all_history_dfs.append(history_df)
    
    combined_history = pd.concat(all_history_dfs, ignore_index=True)
    combined_history.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
    print(f"✓ Saved training history")
    
    # OOF metrics
    oof_preds_class = oof_df[[f'prob_{i}' for i in range(cfg.N_CLASSES)]].values.argmax(axis=1)
    
    oof_metrics = {
        'timestamp': timestamp,
        'overall': {
            'balanced_accuracy': float(balanced_accuracy_score(oof_df['target'], oof_preds_class)),
            'total_samples': len(oof_df),
        },
        'per_class': {}
    }
    
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        y_true_binary = (oof_df['target'] == i).astype(int)
        y_prob = oof_df[f'prob_{i}'].values
        y_pred_binary = (oof_preds_class == i).astype(int)
        
        tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
        fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
        fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        try:
            auc_score = float(roc_auc_score(y_true_binary, y_prob))
        except:
            auc_score = None
        
        oof_metrics['per_class'][class_name] = {
            'support': int(y_true_binary.sum()),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': auc_score,
            'is_malignant': class_name in ['MEL', 'BCC', 'SCC'],
        }
    
    with open(os.path.join(save_dir, 'oof_metrics.json'), 'w') as f:
        json.dump(oof_metrics, f, indent=4)
    print(f"✓ Saved OOF metrics")
    
    # Test results
    if test_submission is not None:
        test_results = {
            'timestamp': timestamp,
            'total_samples': len(test_submission),
            'prediction_distribution': {}
        }
        
        pred_classes = test_submission['pred'].values
        for i, class_name in enumerate(cfg.CLASS_NAMES):
            count = int((pred_classes == i).sum())
            pct = count / len(pred_classes) * 100
            test_results['prediction_distribution'][class_name] = {'count': count, 'percentage': round(pct, 2)}
        
        with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)
        print(f"✓ Saved test results")
    
    # Summary report
    summary_report = f"""
{'='*60}
EXPERIMENT SUMMARY REPORT - ConvNeXt-tiny
{'='*60}
Timestamp: {timestamp}
Model: {cfg.MODEL_NAME} (~28.6M params)

CONFIGURATION:
- Image Size: {cfg.IMG_SIZE}
- Batch Size: {cfg.BATCH_SIZE} (effective: {cfg.BATCH_SIZE * cfg.ACCUMULATION_STEPS})
- Learning Rate: {cfg.BASE_LR}
- Weight Decay: {cfg.WEIGHT_DECAY}
- Epochs: {cfg.EPOCHS} (min: {cfg.MIN_EPOCHS}, patience: {cfg.PATIENCE})
- Drop Path Rate: {cfg.DROP_PATH_RATE}
- LLRD: {cfg.USE_LLRD} (decay={cfg.LLRD_DECAY})
- Mixup/CutMix: {cfg.USE_MIXUP}
- EMA: {cfg.USE_EMA} (decay={cfg.EMA_DECAY})

TRAINING RESULTS (5-Fold CV):
- Mean Balanced Accuracy: {training_results['summary']['mean_bal_acc']:.4f} ± {training_results['summary']['std_bal_acc']:.4f}
- Best Fold: {training_results['summary']['best_fold']} ({training_results['summary']['best_bal_acc']:.4f})

OOF RESULTS:
- Overall Balanced Accuracy: {oof_metrics['overall']['balanced_accuracy']:.4f}

Per-class Performance:
"""
    
    for class_name, metrics in oof_metrics['per_class'].items():
        marker = "⚠️ MALIGNANT" if metrics['is_malignant'] else ""
        auc_val = metrics['roc_auc'] if metrics['roc_auc'] else 'N/A'
        summary_report += f"  {class_name}: AUC={auc_val}, F1={metrics['f1_score']:.4f}, Support={metrics['support']} {marker}\n"
    
    summary_report += f"\n{'='*60}\n"
    
    with open(os.path.join(save_dir, 'experiment_summary.txt'), 'w') as f:
        f.write(summary_report)
    print(f"✓ Saved experiment summary")
    print(summary_report)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Run 5-Fold Training
    print("\n" + "="*60)
    print("STARTING 5-FOLD CROSS-VALIDATION TRAINING")
    print("="*60)
    
    results = []
    all_oof = []
    all_histories = []
    
    for fold in range(cfg.N_FOLDS):
        fold_result = train_one_fold(fold, df, cfg, criterion, mixup_fn, mixup_criterion)
        results.append(fold_result)
        all_histories.append(fold_result['history'])
        
        fold_oof = pd.DataFrame({
            'image': fold_result['oof_images'],
            'fold': fold,
            'target': fold_result['oof_targets'],
            **{f'prob_{i}': fold_result['oof_preds'][:, i] for i in range(cfg.N_CLASSES)}
        })
        all_oof.append(fold_oof)
    
    # Summary
    bal_accs = [r['best_bal_acc'] for r in results]
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY - ConvNeXt-tiny")
    print(f"{'='*60}")
    for i, acc in enumerate(bal_accs):
        print(f"Fold {i}: {acc:.4f}")
    print(f"Mean: {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}")
    print(f"{'='*60}")
    
    # OOF Evaluation
    oof_df = pd.concat(all_oof, ignore_index=True)
    oof_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'oof_convnext_tiny.csv'), index=False)
    
    oof_preds_class = oof_df[[f'prob_{i}' for i in range(cfg.N_CLASSES)]].values.argmax(axis=1)
    oof_bal_acc = balanced_accuracy_score(oof_df['target'], oof_preds_class)
    
    print(f"\nOOF Balanced Accuracy: {oof_bal_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(oof_df['target'], oof_preds_class, target_names=cfg.CLASS_NAMES))
    
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
    
    cm = confusion_matrix(oof_df['target'], oof_preds_class)
    
    # Threshold Optimization
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)
    
    optimal_thresholds_f1, metrics_f1 = find_optimal_thresholds(oof_df, cfg, strategy='f1')
    optimal_thresholds_recall, metrics_recall = find_optimal_thresholds(oof_df, cfg, strategy='recall')
    
    print("\n>>> Evaluating with F1-optimized thresholds:")
    eval_results_f1 = evaluate_with_optimized_thresholds(oof_df, optimal_thresholds_f1, cfg)
    
    print("\n>>> Evaluating with Recall-optimized thresholds:")
    eval_results_recall = evaluate_with_optimized_thresholds(oof_df, optimal_thresholds_recall, cfg)
    
    # Save thresholds
    thresholds_output = {
        'f1_strategy': {'thresholds': optimal_thresholds_f1, 'metrics': metrics_f1, 'balanced_accuracy': eval_results_f1['optimized_bal_acc']},
        'recall_strategy': {'thresholds': optimal_thresholds_recall, 'metrics': metrics_recall, 'balanced_accuracy': eval_results_recall['optimized_bal_acc']}
    }
    
    with open(os.path.join(cfg.OUTPUT_DIR, 'optimal_thresholds.json'), 'w') as f:
        json.dump(thresholds_output, f, indent=4)
    print(f"\n✓ Saved optimal thresholds")
    
    # Visualizations
    plot_confusion_matrix(cm, cfg.CLASS_NAMES, save_path=os.path.join(cfg.OUTPUT_DIR, 'confusion_matrix.png'))
    plot_roc_curves(oof_df, cfg, save_path=os.path.join(cfg.OUTPUT_DIR, 'roc_curves.png'))
    plot_training_history(all_histories, cfg, save_path=os.path.join(cfg.OUTPUT_DIR, 'training_history.png'))
    
    # Test Set Inference
    print("\n" + "="*60)
    print("TEST SET INFERENCE")
    print("="*60)
    
    test_submission = inference_with_tta(cfg.TEST_CSV_PATH, cfg.TEST_IMG_ROOT, cfg, n_tta=7)
    
    # Save all results
    save_experiment_results(cfg, results, all_histories, oof_df, test_submission, cfg.OUTPUT_DIR)
    
    # List output files
    print("\n" + "="*60)
    print("OUTPUT FILES")
    print("="*60)
    
    output_files = glob.glob(os.path.join(cfg.OUTPUT_DIR, '*'))
    for f in sorted(output_files):
        size = os.path.getsize(f)
        if size > 1024*1024:
            size_str = f"{size/1024/1024:.2f} MB"
        elif size > 1024:
            size_str = f"{size/1024:.2f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {os.path.basename(f):40} {size_str}")
    
    print("\n" + "="*60)
    print("🎉 ALL DONE!")
    print("="*60)

