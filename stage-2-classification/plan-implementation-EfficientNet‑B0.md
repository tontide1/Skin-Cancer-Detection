Plan dưới đây thiết kế riêng cho **EfficientNet‑B0 trên Kaggle (GPU T4x2)**, bao gồm: cấu trúc notebook, hyperparameter, training schedule và early stopping, tối ưu cho ISIC 2019 (8 lớp, ('MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC'))

> **Lưu ý về target mapping:**  
>
> - 0: MEL (Melanoma)
> - 1: NV (Melanocytic nevus)  
> - 2: BCC (Basal cell carcinoma)
> - 3: AK (Actinic keratosis)  
> - 4: BKL (Benign keratosis)  
> - 5: DF (Dermatofibroma)  
> - 6: VASC (Vascular lesion)  
> - 7: SCC (Squamous cell carcinoma)

***

## 1. Cấu trúc notebook & config chung  

**Cell 1 – Imports & config**

```python
import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Reproducibility
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
```

- Đặt đường dẫn:  
  - `CSV_PATH = "/kaggle/input/isic-2019-task-1/ISIC_2019_5folds_metadata.csv"`  
  - `IMG_ROOT = "/kaggle/input/isic-2019-task-1/cropped_lesions"` ✅ (không có duplicate folder)
- Config (dùng dataclass cho type hints):  

```python
from dataclasses import dataclass

@dataclass
class Config:
    # Paths - Training
    CSV_PATH: str = "/kaggle/input/isic-2019-task-1/ISIC_2019_5folds_metadata.csv"
    IMG_ROOT: str = "/kaggle/input/isic-2019-task-1/cropped_lesions"
    OUTPUT_DIR: str = "/kaggle/working"
    
    # Paths - Test set
    TEST_CSV_PATH: str = "/kaggle/input/isic-2019-task-1/ISIC_2019_test_metadata.csv"
    TEST_IMG_ROOT: str = "/kaggle/input/isic-2019-task-1/cropped_lesions_testset/cropped_lesions_testset"
    
    # Model
    MODEL_NAME: str = "efficientnet_b0"
    N_CLASSES: int = 8
    PRETRAINED: bool = True
    DROP_RATE: float = 0.3  # tăng từ 0.2 để regularize tốt hơn
    
    # Training
    IMG_SIZE: int = 256  # tăng từ 224 để capture chi tiết tốt hơn
    N_FOLDS: int = 5
    BATCH_SIZE: int = 32  # giảm xuống 24 hoặc 16 nếu OOM
    EPOCHS: int = 50
    MIN_EPOCHS: int = 15
    PATIENCE: int = 8
    
    # Optimizer
    BASE_LR: float = 3e-4
    MIN_LR: float = 1e-6
    WEIGHT_DECAY: float = 1e-4
    WARMUP_EPOCHS: int = 3
    
    # Hardware
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    
    # Class names for reference
    CLASS_NAMES: tuple = ('MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC')

cfg = Config()
```  

***

## 2. Chuẩn bị dữ liệu, class weight, Dataset  

**Cell 2 – Đọc CSV & tính class weight**

```python
# Đọc CSV
df = pd.read_csv(cfg.CSV_PATH)
print(f"Dataset size: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nClass distribution:")
print(df['target'].value_counts().sort_index())

# Tính class weight (inverse frequency)
counts = df["target"].value_counts().sort_index().values
N = len(df)
weights = N / counts
weights = weights / weights.mean()  # normalize để tránh gradient explosion
class_weight = torch.tensor(weights, dtype=torch.float).to(cfg.DEVICE)

print(f"\nClass weights: {weights}")
```

**Cell 3 – Dataset & augment**

- Dùng Albumentations với augmentation phù hợp cho dermatology:[7][8]

```python
def get_train_transforms(img_size):
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 5)),
        ], p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),  # tăng contrast local cho skin lesion
        A.CoarseDropout(max_holes=1, max_height=img_size//4, max_width=img_size//4, 
                        fill_value=0, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
```

- `ISICDataset`:

```python
class ISICDataset(Dataset):
    def __init__(self, df, transforms=None, img_root=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.img_root = img_root
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Lấy filename từ path trong CSV
        img_path = os.path.join(self.img_root, os.path.basename(row["path"]))
        
        # Đọc ảnh bằng cv2 (BGR -> RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        
        target = torch.tensor(row["target"], dtype=torch.long)
        
        return image, target
```

***

## 3. Model EfficientNet‑B0 + hyperparameter  

**Cell 4 – Model factory**

```python
def create_model(cfg, use_dp=True):
    """Tạo model với optional DataParallel wrapper"""
    model = timm.create_model(
        cfg.MODEL_NAME, 
        pretrained=cfg.PRETRAINED, 
        num_classes=cfg.N_CLASSES, 
        drop_rate=cfg.DROP_RATE
    )
    model = model.to(cfg.DEVICE)
    
    # Wrap với DataParallel nếu có nhiều GPU
    if use_dp and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    return model
```

- Loss với class weight:  

```python
criterion = nn.CrossEntropyLoss(weight=class_weight)
```

- Optimizer (AdamW với weight decay):  

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY)
```

- Scheduler (Warmup + CosineAnnealing):  

```python
def get_scheduler(optimizer, cfg, num_train_steps):
    """Warmup linear rồi CosineAnnealing"""
    warmup_steps = cfg.WARMUP_EPOCHS * (num_train_steps // cfg.EPOCHS)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup từ 1e-6 đến BASE_LR
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, num_train_steps - warmup_steps))
        return max(cfg.MIN_LR / cfg.BASE_LR, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

- AMP (Mixed Precision Training):  

```python
scaler = GradScaler()
```

***

## 4. Training schedule + early stopping (cho từng fold)  

**Cell 5 – Hàm train_one_fold(fold)**  

```python
def train_one_fold(fold, df, cfg):
    print(f"\n{'='*50}")
    print(f"FOLD {fold}")
    print(f"{'='*50}")
    
    # 1. Tạo train/val DataLoader
    train_df = df[df.fold != fold].reset_index(drop=True)
    val_df = df[df.fold == fold].reset_index(drop=True)
    
    train_dataset = ISICDataset(train_df, get_train_transforms(cfg.IMG_SIZE), cfg.IMG_ROOT)
    val_dataset = ISICDataset(val_df, get_val_transforms(cfg.IMG_SIZE), cfg.IMG_ROOT)
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    
    # 2. Khởi tạo model, optimizer, scheduler, scaler
    model = create_model(cfg, use_dp=True)  # Sử dụng DataParallel nếu có nhiều GPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY)
    
    num_train_steps = len(train_loader) * cfg.EPOCHS
    scheduler = get_scheduler(optimizer, cfg, num_train_steps)
    scaler = GradScaler()
    
    # Early stopping state
    best_bal_acc = 0.0
    epochs_no_improve = 0
    best_checkpoint_path = os.path.join(cfg.OUTPUT_DIR, f"effnet_b0_fold{fold}_best.pth")
    
    # Lists để lưu OOF predictions
    oof_preds = []
    oof_targets = []
    oof_images = []
    
    # History để tracking training progress
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'bal_acc': [],
        'learning_rate': []
    }
    
    # 3. Vòng epoch
    for epoch in range(1, cfg.EPOCHS + 1):
        # ---- TRAIN PHASE ----
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Train]")
        for images, targets in pbar:
            images = images.to(cfg.DEVICE)
            targets = targets.to(cfg.DEVICE)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping (optional nhưng recommended)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
        
        avg_train_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # ---- VAL PHASE ----
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
        
        # Tính metrics
        bal_acc = balanced_accuracy_score(all_targets, all_preds)
        
        # Lưu history
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['bal_acc'].append(bal_acc)
        history['learning_rate'].append(current_lr)
        
        print(f"\nEpoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, bal_acc={bal_acc:.4f}")
        
        # ---- EARLY STOPPING ----
        if bal_acc > best_bal_acc + 1e-3:
            best_bal_acc = bal_acc
            epochs_no_improve = 0
            
            # Lưu checkpoint (handle DataParallel)
            # Nếu model được wrap bởi DataParallel, lưu model.module.state_dict()
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_bal_acc': best_bal_acc,
            }, best_checkpoint_path)
            print(f"✓ Saved best model with bal_acc={best_bal_acc:.4f}")
            
            # Lưu OOF predictions cho fold này
            oof_preds = all_probs
            oof_targets = all_targets
            oof_images = val_df['image'].tolist()
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
        
        if epoch >= cfg.MIN_EPOCHS and epochs_no_improve >= cfg.PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # 4. Trả về kết quả
    return {
        'fold': fold,
        'best_bal_acc': best_bal_acc,
        'checkpoint_path': best_checkpoint_path,
        'oof_preds': oof_preds,
        'oof_targets': oof_targets,
        'oof_images': oof_images,
        'history': history,  # Thêm history
    }
```

**Cell 6 – Chạy 5 folds**

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
print(f"\n{'='*50}")
print(f"RESULTS SUMMARY")
print(f"{'='*50}")
for i, acc in enumerate(bal_accs):
    print(f"Fold {i}: {acc:.4f}")
print(f"Mean: {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}")
```

**Cell 6.1 – Visualization Training History**

```python
def plot_training_history(all_histories, cfg, save_path=None):
    """Vẽ biểu đồ training history cho tất cả folds"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, cfg.N_FOLDS))
    
    # 1. Training Loss
    ax1 = axes[0, 0]
    for fold, history in enumerate(all_histories):
        ax1.plot(history['epoch'], history['train_loss'], 
                 label=f'Fold {fold}', color=colors[fold], alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss per Fold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Loss
    ax2 = axes[0, 1]
    for fold, history in enumerate(all_histories):
        ax2.plot(history['epoch'], history['val_loss'], 
                 label=f'Fold {fold}', color=colors[fold], alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss per Fold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Balanced Accuracy
    ax3 = axes[1, 0]
    for fold, history in enumerate(all_histories):
        ax3.plot(history['epoch'], history['bal_acc'], 
                 label=f'Fold {fold}', color=colors[fold], alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Balanced Accuracy')
    ax3.set_title('Balanced Accuracy per Fold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning Rate
    ax4 = axes[1, 1]
    for fold, history in enumerate(all_histories):
        ax4.plot(history['epoch'], history['learning_rate'], 
                 label=f'Fold {fold}', color=colors[fold], alpha=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training history plot to {save_path}")
    
    plt.show()


def plot_fold_comparison(results, cfg, save_path=None):
    """Vẽ biểu đồ so sánh performance giữa các folds"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Bar chart - Balanced Accuracy per fold
    ax1 = axes[0]
    folds = [r['fold'] for r in results]
    bal_accs = [r['best_bal_acc'] for r in results]
    bars = ax1.bar(folds, bal_accs, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(folds))))
    ax1.axhline(y=np.mean(bal_accs), color='red', linestyle='--', 
                label=f'Mean: {np.mean(bal_accs):.4f}')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Balanced Accuracy')
    ax1.set_title('Balanced Accuracy per Fold')
    ax1.set_xticks(folds)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Thêm giá trị lên mỗi bar
    for bar, acc in zip(bars, bal_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Box plot - Distribution of balanced accuracy
    ax2 = axes[1]
    ax2.boxplot(bal_accs, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    ax2.scatter([1]*len(bal_accs), bal_accs, color='blue', alpha=0.6, s=100, zorder=5)
    ax2.set_ylabel('Balanced Accuracy')
    ax2.set_title(f'Distribution of Balanced Accuracy\nMean: {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved fold comparison plot to {save_path}")
    
    plt.show()


# Vẽ biểu đồ training history
plot_training_history(
    all_histories, cfg,
    save_path=os.path.join(cfg.OUTPUT_DIR, 'training_history.png')
)

# Vẽ biểu đồ so sánh folds
plot_fold_comparison(
    results, cfg,
    save_path=os.path.join(cfg.OUTPUT_DIR, 'fold_comparison.png')
)
```  

***

## 5. OOF predictions & inference test  

**Cell 7 – Sinh OOF & Đánh giá chi tiết**

```python
# Concat tất cả OOF predictions
oof_df = pd.concat(all_oof, ignore_index=True)
oof_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'oof_effnet_b0.csv'), index=False)

# Tính OOF balanced accuracy tổng
oof_preds_class = oof_df[[f'prob_{i}' for i in range(cfg.N_CLASSES)]].values.argmax(axis=1)
oof_bal_acc = balanced_accuracy_score(oof_df['target'], oof_preds_class)
print(f"\nOOF Balanced Accuracy: {oof_bal_acc:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(
    oof_df['target'], 
    oof_preds_class, 
    target_names=cfg.CLASS_NAMES
))

# ROC-AUC từng lớp (One-vs-Rest)
print("\nPer-class ROC-AUC:")
for i, class_name in enumerate(cfg.CLASS_NAMES):
    y_true_binary = (oof_df['target'] == i).astype(int)
    y_prob = oof_df[f'prob_{i}'].values
    try:
        auc = roc_auc_score(y_true_binary, y_prob)
        malignant = "⚠️" if class_name in ['MEL', 'BCC', 'SCC'] else ""
        print(f"  {class_name}: {auc:.4f} {malignant}")
    except:
        print(f"  {class_name}: N/A (only one class present)")

# Confusion matrix
cm = confusion_matrix(oof_df['target'], oof_preds_class)
print("\nConfusion Matrix:")
print(cm)
```

**Cell 7.1 – Visualization OOF Results**

```python
def plot_confusion_matrix(cm, class_names, save_path=None):
    """Vẽ confusion matrix với heatmap"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Confusion matrix (counts)
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # 2. Confusion matrix (normalized by true labels)
    ax2 = axes[1]
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (Normalized by True Labels)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved confusion matrix plot to {save_path}")
    
    plt.show()


def plot_roc_curves(oof_df, cfg, save_path=None):
    """Vẽ ROC curves cho từng class"""
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, cfg.N_CLASSES))
    
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        y_true_binary = (oof_df['target'] == i).astype(int)
        y_prob = oof_df[f'prob_{i}'].values
        
        if y_true_binary.sum() > 0:
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Đánh dấu malignant classes
            marker = '⚠️' if class_name in ['MEL', 'BCC', 'SCC'] else ''
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{class_name} {marker}(AUC = {roc_auc:.3f})')
    
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


def plot_class_distribution(oof_df, cfg, save_path=None):
    """Vẽ phân phối class trong predictions vs ground truth"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    pred_classes = oof_df[[f'prob_{i}' for i in range(cfg.N_CLASSES)]].values.argmax(axis=1)
    
    # 1. Ground truth distribution
    ax1 = axes[0]
    true_counts = oof_df['target'].value_counts().sort_index()
    bars1 = ax1.bar(cfg.CLASS_NAMES, true_counts.values, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_title('Ground Truth Distribution')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Prediction distribution
    ax2 = axes[1]
    pred_counts = pd.Series(pred_classes).value_counts().sort_index()
    # Đảm bảo có đủ 8 classes
    pred_counts_full = [pred_counts.get(i, 0) for i in range(cfg.N_CLASSES)]
    bars2 = ax2.bar(cfg.CLASS_NAMES, pred_counts_full, color='coral', alpha=0.8)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Distribution')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved class distribution plot to {save_path}")
    
    plt.show()


# Vẽ confusion matrix
cm = confusion_matrix(oof_df['target'], oof_preds_class)
plot_confusion_matrix(
    cm, cfg.CLASS_NAMES,
    save_path=os.path.join(cfg.OUTPUT_DIR, 'confusion_matrix.png')
)

# Vẽ ROC curves
plot_roc_curves(
    oof_df, cfg,
    save_path=os.path.join(cfg.OUTPUT_DIR, 'roc_curves.png')
)

# Vẽ class distribution
plot_class_distribution(
    oof_df, cfg,
    save_path=os.path.join(cfg.OUTPUT_DIR, 'class_distribution.png')
)
```

**Cell 8 – Inference test (ensemble 5 folds) + TTA**  

```python
def tta_predict(model, image, val_transforms):
    """Test Time Augmentation với flip"""
    model.eval()
    preds = []
    
    with torch.no_grad():
        # Original
        img = val_transforms(image=image)["image"].unsqueeze(0).to(cfg.DEVICE)
        with autocast():
            preds.append(torch.softmax(model(img), dim=1))
        
        # Horizontal flip
        img_hflip = val_transforms(image=cv2.flip(image, 1))["image"].unsqueeze(0).to(cfg.DEVICE)
        with autocast():
            preds.append(torch.softmax(model(img_hflip), dim=1))
        
        # Vertical flip
        img_vflip = val_transforms(image=cv2.flip(image, 0))["image"].unsqueeze(0).to(cfg.DEVICE)
        with autocast():
            preds.append(torch.softmax(model(img_vflip), dim=1))
    
    return torch.stack(preds).mean(dim=0)


def inference_test(test_csv_path, test_img_root, cfg, use_tta=True):
    """Inference với ensemble 5 folds + optional TTA"""
    test_df = pd.read_csv(test_csv_path)
    val_transforms = get_val_transforms(cfg.IMG_SIZE)
    
    print(f"\n{'='*50}")
    print(f"TEST SET INFERENCE")
    print(f"{'='*50}")
    print(f"Test samples: {len(test_df)}")
    print(f"Test images folder: {test_img_root}")
    print(f"Using TTA: {use_tta}")
    
    # Load 5 models với DataParallel support
    models = []
    for fold in range(cfg.N_FOLDS):
        # Tạo model với DataParallel nếu có nhiều GPU
        model = create_model(cfg, use_dp=True)
        
        # Load checkpoint với weights_only=False để tránh lỗi
        checkpoint_path = os.path.join(cfg.OUTPUT_DIR, f"effnet_b0_fold{fold}_best.pth")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
        
        # Handle DataParallel state_dict
        state_dict = checkpoint['model_state_dict']
        if isinstance(model, nn.DataParallel):
            # Nếu model hiện tại là DataParallel nhưng checkpoint không có prefix 'module.'
            if not any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        else:
            # Nếu model không phải DataParallel nhưng checkpoint có prefix 'module.'
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        print(f"  ✓ Loaded fold {fold} (best_bal_acc: {checkpoint['best_bal_acc']:.4f})")
    
    print(f"Loaded {len(models)} fold models")
    
    # Predict
    all_preds = []
    all_images = []
    
    for idx in tqdm(range(len(test_df)), desc="Inference on Test Set"):
        row = test_df.iloc[idx]
        
        # Sử dụng test_img_root cho test set
        img_path = os.path.join(test_img_root, os.path.basename(row["path"]))
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Cannot read image {img_path}")
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_images.append(row['image'])
        
        # Ensemble predictions từ 5 folds
        fold_preds = []
        for model in models:
            if use_tta:
                pred = tta_predict(model, image, val_transforms)
            else:
                img_tensor = val_transforms(image=image)["image"].unsqueeze(0).to(cfg.DEVICE)
                with torch.no_grad(), autocast():
                    pred = torch.softmax(model(img_tensor), dim=1)
            fold_preds.append(pred)
        
        # Average across folds
        avg_pred = torch.stack(fold_preds).mean(dim=0).cpu().numpy()[0]
        all_preds.append(avg_pred)
    
    all_preds = np.array(all_preds)
    pred_classes = all_preds.argmax(axis=1)
    
    # Tạo submission DataFrame
    submission = pd.DataFrame({
        'image': all_images,
        **{f'prob_{i}': all_preds[:, i] for i in range(cfg.N_CLASSES)},
        'pred': pred_classes,
        'pred_label': [cfg.CLASS_NAMES[p] for p in pred_classes]
    })
    
    # Lưu predictions
    output_path = os.path.join(cfg.OUTPUT_DIR, 'effnet_b0_test_predictions.csv')
    submission.to_csv(output_path, index=False)
    print(f"\n✓ Saved test predictions to {output_path}")
    
    # Thống kê predictions
    print(f"\nTest Set Prediction Distribution:")
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        count = (pred_classes == i).sum()
        pct = count / len(pred_classes) * 100
        print(f"  {class_name}: {count} ({pct:.1f}%)")
    
    return submission
```

**Cell 9 – Chạy inference trên Test Set**

```python
# ====== CHẠY INFERENCE TRÊN TEST SET ======
print("\n" + "="*60)
print("STEP: INFERENCE ON TEST SET")
print("="*60)

# Chạy inference với TTA
test_submission = inference_test(
    test_csv_path=cfg.TEST_CSV_PATH,
    test_img_root=cfg.TEST_IMG_ROOT,
    cfg=cfg,
    use_tta=True  # Bật TTA để tăng accuracy
)

print("\n" + "="*60)
print("TRAINING & INFERENCE COMPLETED!")
print("="*60)
print(f"\nOutput files:")
print(f"  - OOF predictions: {cfg.OUTPUT_DIR}/oof_effnet_b0.csv")
print(f"  - Test predictions: {cfg.OUTPUT_DIR}/effnet_b0_test_predictions.csv")
print(f"  - Model checkpoints: {cfg.OUTPUT_DIR}/effnet_b0_fold*_best.pth")
```

**Cell 10 – Lưu toàn bộ Config và Results**

```python
def save_experiment_results(cfg, results, all_histories, oof_df, test_submission, save_dir):
    """Lưu toàn bộ config, kết quả training/val và test"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Lưu Config
    config_dict = {
        'experiment_timestamp': timestamp,
        'model_name': cfg.MODEL_NAME,
        'n_classes': cfg.N_CLASSES,
        'img_size': cfg.IMG_SIZE,
        'n_folds': cfg.N_FOLDS,
        'batch_size': cfg.BATCH_SIZE,
        'epochs': cfg.EPOCHS,
        'min_epochs': cfg.MIN_EPOCHS,
        'patience': cfg.PATIENCE,
        'base_lr': cfg.BASE_LR,
        'min_lr': cfg.MIN_LR,
        'weight_decay': cfg.WEIGHT_DECAY,
        'warmup_epochs': cfg.WARMUP_EPOCHS,
        'drop_rate': cfg.DROP_RATE,
        'pretrained': cfg.PRETRAINED,
        'class_names': list(cfg.CLASS_NAMES),
        'paths': {
            'csv_path': cfg.CSV_PATH,
            'img_root': cfg.IMG_ROOT,
            'test_csv_path': cfg.TEST_CSV_PATH,
            'test_img_root': cfg.TEST_IMG_ROOT,
            'output_dir': cfg.OUTPUT_DIR,
        }
    }
    
    config_path = os.path.join(save_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✓ Saved config to {config_path}")
    
    # 2. Lưu Training Results
    training_results = {
        'timestamp': timestamp,
        'fold_results': [],
        'summary': {
            'mean_bal_acc': float(np.mean([r['best_bal_acc'] for r in results])),
            'std_bal_acc': float(np.std([r['best_bal_acc'] for r in results])),
            'best_fold': int(np.argmax([r['best_bal_acc'] for r in results])),
            'best_bal_acc': float(max([r['best_bal_acc'] for r in results])),
        }
    }
    
    for r in results:
        training_results['fold_results'].append({
            'fold': r['fold'],
            'best_bal_acc': float(r['best_bal_acc']),
            'checkpoint_path': r['checkpoint_path'],
            'n_epochs_trained': len(r['history']['epoch']),
        })
    
    results_path = os.path.join(save_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=4)
    print(f"✓ Saved training results to {results_path}")
    
    # 3. Lưu Training History (CSV format cho dễ phân tích)
    all_history_dfs = []
    for fold, history in enumerate(all_histories):
        history_df = pd.DataFrame(history)
        history_df['fold'] = fold
        all_history_dfs.append(history_df)
    
    combined_history = pd.concat(all_history_dfs, ignore_index=True)
    history_path = os.path.join(save_dir, 'training_history.csv')
    combined_history.to_csv(history_path, index=False)
    print(f"✓ Saved training history to {history_path}")
    
    # 4. Lưu OOF Metrics
    oof_preds_class = oof_df[[f'prob_{i}' for i in range(cfg.N_CLASSES)]].values.argmax(axis=1)
    
    oof_metrics = {
        'timestamp': timestamp,
        'overall': {
            'balanced_accuracy': float(balanced_accuracy_score(oof_df['target'], oof_preds_class)),
            'total_samples': len(oof_df),
        },
        'per_class': {}
    }
    
    # Per-class metrics
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        y_true_binary = (oof_df['target'] == i).astype(int)
        y_prob = oof_df[f'prob_{i}'].values
        y_pred_binary = (oof_preds_class == i).astype(int)
        
        # Confusion matrix components
        tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
        tn = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
        fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
        fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        try:
            auc = float(roc_auc_score(y_true_binary, y_prob))
        except:
            auc = None
        
        oof_metrics['per_class'][class_name] = {
            'support': int(y_true_binary.sum()),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': auc,
            'is_malignant': class_name in ['MEL', 'BCC', 'SCC'],
        }
    
    oof_metrics_path = os.path.join(save_dir, 'oof_metrics.json')
    with open(oof_metrics_path, 'w') as f:
        json.dump(oof_metrics, f, indent=4)
    print(f"✓ Saved OOF metrics to {oof_metrics_path}")
    
    # 5. Lưu Test Set Results
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
            test_results['prediction_distribution'][class_name] = {
                'count': count,
                'percentage': round(pct, 2)
            }
        
        test_results_path = os.path.join(save_dir, 'test_results.json')
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=4)
        print(f"✓ Saved test results to {test_results_path}")
    
    # 6. Tạo summary report
    summary_report = f"""
{'='*60}
EXPERIMENT SUMMARY REPORT
{'='*60}
Timestamp: {timestamp}
Model: {cfg.MODEL_NAME}

CONFIGURATION:
- Image Size: {cfg.IMG_SIZE}
- Batch Size: {cfg.BATCH_SIZE}
- Learning Rate: {cfg.BASE_LR}
- Weight Decay: {cfg.WEIGHT_DECAY}
- Epochs: {cfg.EPOCHS} (min: {cfg.MIN_EPOCHS}, patience: {cfg.PATIENCE})
- Dropout Rate: {cfg.DROP_RATE}

TRAINING RESULTS (5-Fold CV):
- Mean Balanced Accuracy: {training_results['summary']['mean_bal_acc']:.4f} ± {training_results['summary']['std_bal_acc']:.4f}
- Best Fold: {training_results['summary']['best_fold']} ({training_results['summary']['best_bal_acc']:.4f})

OOF RESULTS:
- Overall Balanced Accuracy: {oof_metrics['overall']['balanced_accuracy']:.4f}

Per-class Performance (sorted by AUC):
"""
    
    # Sort by AUC
    class_aucs = [(name, metrics['roc_auc'], metrics['is_malignant']) 
                  for name, metrics in oof_metrics['per_class'].items() 
                  if metrics['roc_auc'] is not None]
    class_aucs.sort(key=lambda x: x[1], reverse=True)
    
    for class_name, auc, is_malignant in class_aucs:
        marker = "⚠️ MALIGNANT" if is_malignant else ""
        metrics = oof_metrics['per_class'][class_name]
        summary_report += f"  {class_name}: AUC={auc:.4f}, F1={metrics['f1_score']:.4f}, Support={metrics['support']} {marker}\n"
    
    if test_submission is not None:
        summary_report += f"""
TEST SET PREDICTIONS:
- Total Samples: {test_results['total_samples']}
- Distribution:
"""
        for class_name, stats in test_results['prediction_distribution'].items():
            summary_report += f"  {class_name}: {stats['count']} ({stats['percentage']}%)\n"
    
    summary_report += f"""
OUTPUT FILES:
- experiment_config.json
- training_results.json
- training_history.csv
- oof_metrics.json
- oof_effnet_b0.csv
- effnet_b0_test_predictions.csv
- effnet_b0_fold*_best.pth
- training_history.png
- fold_comparison.png
- confusion_matrix.png
- roc_curves.png
- class_distribution.png
{'='*60}
"""
    
    report_path = os.path.join(save_dir, 'experiment_summary.txt')
    with open(report_path, 'w') as f:
        f.write(summary_report)
    print(f"✓ Saved experiment summary to {report_path}")
    
    print(summary_report)
    
    return {
        'config_path': config_path,
        'results_path': results_path,
        'history_path': history_path,
        'oof_metrics_path': oof_metrics_path,
        'report_path': report_path,
    }


# Lưu toàn bộ kết quả
saved_paths = save_experiment_results(
    cfg=cfg,
    results=results,
    all_histories=all_histories,
    oof_df=oof_df,
    test_submission=test_submission,
    save_dir=cfg.OUTPUT_DIR
)

print("\n" + "="*60)
print("ALL RESULTS SAVED SUCCESSFULLY!")
print("="*60)
```

***

## 6. Tổng kết các điểm quan trọng

| Config | Giá trị | Lý do |
|--------|---------|-------|
| `IMG_SIZE` | 256 | Capture chi tiết texture da tốt hơn 224 |
| `DROP_RATE` | 0.3 | Regularization mạnh hơn cho imbalanced data |
| `BATCH_SIZE` | 32 | Cân bằng giữa speed và memory |
| `BASE_LR` | 3e-4 | Learning rate chuẩn cho fine-tuning |
| `WARMUP_EPOCHS` | 3 | Ổn định training ban đầu |
| `PATIENCE` | 8 | Đủ thời gian cho model converge |
| `MIN_EPOCHS` | 15 | Đảm bảo train đủ lâu trước khi early stop |

**Augmentation đặc biệt cho dermatology:**

- `CLAHE`: Tăng contrast local, giúp nhìn rõ cấu trúc da
- `HueSaturationValue`: Xử lý skin tone variation
- `GaussNoise/GaussianBlur`: Simulate ảnh chất lượng khác nhau
- TTA (flip horizontal/vertical): Tăng ~1-2% accuracy

**Output Files được tạo ra:**

| File | Mô tả |
|------|-------|
| `experiment_config.json` | Toàn bộ config của experiment |
| `training_results.json` | Kết quả training từng fold + summary |
| `training_history.csv` | Loss, accuracy, LR theo từng epoch |
| `oof_metrics.json` | Metrics chi tiết cho OOF predictions |
| `oof_effnet_b0.csv` | OOF predictions với probabilities |
| `effnet_b0_test_predictions.csv` | Predictions trên test set |
| `effnet_b0_fold*_best.pth` | 5 model checkpoints |
| `training_history.png` | Biểu đồ training curves |
| `fold_comparison.png` | So sánh performance giữa folds |
| `confusion_matrix.png` | Confusion matrix visualization |
| `roc_curves.png` | ROC curves cho từng class |
| `class_distribution.png` | Phân phối class predictions |
| `experiment_summary.txt` | Summary report dạng text |

***
