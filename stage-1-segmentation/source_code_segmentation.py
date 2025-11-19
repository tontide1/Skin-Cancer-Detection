# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings
import time
import os
import gc
warnings.filterwarnings('ignore')

# Clear CUDA cache trước khi bắt đầu
torch.cuda.empty_cache()
gc.collect()

# Set các biến môi trường để tối ưu PyTorch và fix memory fragmentation
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Fix memory fragmentation

# Kiểm tra GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

print("=" * 60)
print("GPU INFORMATION")
print("=" * 60)
print(f"Device: {device}")
print(f"Number of GPUs available: {num_gpus}")
if torch.cuda.is_available():
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        # Hiển thị memory info
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  - Total Memory: {mem_total:.2f} GB")
        print(f"  - Allocated: {mem_allocated:.2f} GB")
        print(f"  - Reserved: {mem_reserved:.2f} GB")
        print(f"  - Free: {mem_total - mem_allocated:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
print("=" * 60)

# Enable các tối ưu backend
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("✓ Backend optimization enabled!")
print("✓ Memory fragmentation fix enabled!")

# %%
# Configuration - Optimized based on training history analysis
class Config:
    # Paths kaggle
    DATA_ROOT = Path("/kaggle/input/isic-2018-task-1-segmentation")
    IMG_DIRS = {
        "train": DATA_ROOT / "ISIC2018_Task1-2_Training_Input",
        "val": DATA_ROOT / "ISIC2018_Task1-2_Validation_Input",
        "test": DATA_ROOT / "ISIC2018_Task1-2_Test_Input",
    }
    MASK_DIRS = {
        "train": DATA_ROOT / "ISIC2018_Task1_Training_GroundTruth",
        "val": DATA_ROOT / "ISIC2018_Task1_Validation_GroundTruth",
        "test": DATA_ROOT / "ISIC2018_Task1_Test_GroundTruth",
    }
    
    #path local
    
    # DATA_ROOT = Path("./data")  # Relative to notebook location
    # IMG_DIRS = {
    #     "train": DATA_ROOT / "ISIC2018_Task1-2_Training_Input",
    #     "val": DATA_ROOT / "ISIC2018_Task1-2_Validation_Input",
    #     "test": DATA_ROOT / "ISIC2018_Task1-2_Test_Input",
    # }
    # MASK_DIRS = {
    #     "train": DATA_ROOT / "ISIC2018_Task1_Training_GroundTruth",
    #     "val": DATA_ROOT / "ISIC2018_Task1_Validation_GroundTruth",
    #     "test": DATA_ROOT / "ISIC2018_Task1_Test_GroundTruth",
    # }
    
    
    
    # Model
    IMG_SIZE = 256
    IN_CHANNELS = 3
    OUT_CHANNELS = 1
    BASE_FILTERS = 64
    DROPOUT_RATE = 0.2  # NEW: Add dropout for regularization
    
    # Multi-GPU settings
    NUM_GPUS = torch.cuda.device_count()
    USE_MULTI_GPU = NUM_GPUS > 1
    
    # Training - Optimized based on results
    if USE_MULTI_GPU:
        BATCH_SIZE = 64  # 32 per GPU
        LR_INITIAL = 2.5e-4
    else:
        BATCH_SIZE = 32
        LR_INITIAL = 2e-4
    
    MAX_EPOCHS = 100
    LR_MIN = 1e-6
    
    # Optimizer
    BETA_1 = 0.9
    BETA_2 = 0.999
    EPSILON = 1e-7
    WEIGHT_DECAY = 1e-4  # NEW: Add L2 regularization
    
    # Callbacks - Optimized based on training history
    EARLY_STOPPING_PATIENCE = 20  # Increased from 15
    LR_REDUCE_PATIENCE = 7  # Increased from 5
    LR_REDUCE_FACTOR = 0.5
    
    # Loss
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5
    
    # Mixed Precision
    USE_AMP = True
    
    # Memory management
    GRADIENT_ACCUMULATION_STEPS = 1
    
    # Save
    CHECKPOINT_DIR = Path("./checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    BEST_MODEL_PATH = CHECKPOINT_DIR / "best_unet_model.pth"
    
    # NEW: Monitoring
    SAVE_TOP_K = 3  # Save top 3 best models
    MONITOR_METRIC = "val_dice"  # Monitor Dice instead of loss

cfg = Config()

# Clear memory lần nữa sau khi load config
torch.cuda.empty_cache()
gc.collect()

print("\n" + "=" * 60)
print("CONFIGURATION (OPTIMIZED v2 - Based on Training Analysis)")
print("=" * 60)
print(f"Number of GPUs: {cfg.NUM_GPUS}")
print(f"Multi-GPU mode: {cfg.USE_MULTI_GPU}")
print(f"Total Batch size: {cfg.BATCH_SIZE}")
if cfg.USE_MULTI_GPU:
    print(f"Batch per GPU: {cfg.BATCH_SIZE // cfg.NUM_GPUS}")
print(f"Initial LR: {cfg.LR_INITIAL}")
print(f"Weight Decay: {cfg.WEIGHT_DECAY}")
print(f"Dropout Rate: {cfg.DROPOUT_RATE}")
print(f"Max epochs: {cfg.MAX_EPOCHS}")
print(f"Early stopping patience: {cfg.EARLY_STOPPING_PATIENCE}")
print(f"LR reduce patience: {cfg.LR_REDUCE_PATIENCE}")
print(f"Mixed Precision: {cfg.USE_AMP}")
print("=" * 60)

# %%
def get_train_transforms():
    """Enhanced augmentation to reduce overfitting"""
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5
        ),
        
        # NEW: Elastic transform for better generalization
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.3
        ),
        
        # NEW: Color augmentations (không ảnh hưởng mask)
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        ], p=0.5),
        
        # NEW: Blur and noise for robustness
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),
        
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        
        # Normalize
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])

def get_val_transforms():
    """Augmentation cho validation/test (chỉ normalize)"""
    return A.Compose([
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])

print("✓ Enhanced augmentation pipelines created!")

# %%
class ISICSegmentationDataset(Dataset):
    """Dataset cho ISIC 2018 Task 1 Segmentation"""
    
    def __init__(self, img_dir, mask_dir, img_size=256, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.transform = transform
        
        # Load danh sách ảnh
        self.img_paths = sorted(list(self.img_dir.glob("*.jpg")))
        if len(self.img_paths) == 0:
            self.img_paths = sorted(list(self.img_dir.glob("*.png")))
        
        print(f"Found {len(self.img_paths)} images in {img_dir}")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        img_name = img_path.stem
        mask_path = self.mask_dir / f"{img_name}_segmentation.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Binarize mask
        mask = (mask > 127).astype(np.uint8)
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Đảm bảo mask có đúng shape (1, H, W)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        mask = mask.float()
        
        return image, mask

# Create datasets
train_dataset = ISICSegmentationDataset(
    cfg.IMG_DIRS['train'], 
    cfg.MASK_DIRS['train'],
    img_size=cfg.IMG_SIZE,
    transform=get_train_transforms()
)

val_dataset = ISICSegmentationDataset(
    cfg.IMG_DIRS['val'], 
    cfg.MASK_DIRS['val'],
    img_size=cfg.IMG_SIZE,
    transform=get_val_transforms()
)

test_dataset = ISICSegmentationDataset(
    cfg.IMG_DIRS['test'], 
    cfg.MASK_DIRS['test'],
    img_size=cfg.IMG_SIZE,
    transform=get_val_transforms()
)

print(f"\nDataset Summary:")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# %%
# Optimized DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=cfg.BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,  # Tăng workers
    pin_memory=True,
    persistent_workers=True,  # Giữ workers alive
    prefetch_factor=2  # Prefetch batches
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=cfg.BATCH_SIZE, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=cfg.BATCH_SIZE, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True
)

print(f"✓ DataLoaders created with optimization!")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

# %%
class ConvBlock(nn.Module):
    """Conv2D + BatchNorm + ReLU + Dropout block - Enhanced"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class UNet(nn.Module):
    """
    U-Net with BatchNormalization + Dropout - Enhanced
    Theo spec: 5 level encoding + 5 level decoding
    Base filters: 64, doubling at each downsampling
    Added dropout for regularization to reduce overfitting
    """
    def __init__(self, in_channels=3, out_channels=1, base_filters=64, dropout_rate=0.2):
        super().__init__()
        
        # Encoder - Add dropout to deeper layers
        self.enc1 = ConvBlock(in_channels, base_filters, dropout_rate=0)  # No dropout in first layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = ConvBlock(base_filters, base_filters * 2, dropout_rate=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4, dropout_rate=dropout_rate * 0.5)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8, dropout_rate=dropout_rate)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck - Highest dropout
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16, dropout_rate=dropout_rate)
        
        # Decoder - Gradually reduce dropout
        self.upconv4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_filters * 16, base_filters * 8, dropout_rate=dropout_rate)
        
        self.upconv3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4, dropout_rate=dropout_rate * 0.5)
        
        self.upconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2, dropout_rate=0)
        
        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters, dropout_rate=0)
        
        # Output
        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        x = self.bottleneck(self.pool4(enc4))
        
        # Decoder với skip connections
        x = self.dec4(torch.cat([self.upconv4(x), enc4], dim=1))
        x = self.dec3(torch.cat([self.upconv3(x), enc3], dim=1))
        x = self.dec2(torch.cat([self.upconv2(x), enc2], dim=1))
        x = self.dec1(torch.cat([self.upconv1(x), enc1], dim=1))
        
        return self.out_conv(x)


# Khởi tạo model với dropout
model = UNet(
    in_channels=cfg.IN_CHANNELS,
    out_channels=cfg.OUT_CHANNELS,
    base_filters=cfg.BASE_FILTERS,
    dropout_rate=cfg.DROPOUT_RATE
).to(device)

# Compile model với torch.compile (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print("✓ Model compiled with torch.compile!")
    except Exception as e:
        print(f"⚠ Could not compile model: {e}")

print(f"✓ Enhanced U-Net created with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"✓ Dropout rate: {cfg.DROPOUT_RATE} (applied to deeper layers)")

# %%
class DiceLoss(nn.Module):
    """Dice Loss cho segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Loss = α × BCE + β × Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


criterion = CombinedLoss(bce_weight=cfg.BCE_WEIGHT, dice_weight=cfg.DICE_WEIGHT)
print("✓ Combined Loss (BCE + Dice) initialized!")

# %%
def dice_coefficient(pred, target, threshold=0.5, smooth=1.0):
    """Tính Dice Similarity Coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def iou_score(pred, target, threshold=0.5, smooth=1.0):
    """Tính IoU (Jaccard Index)"""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def pixel_accuracy(pred, target, threshold=0.5):
    """Tính Pixel Accuracy"""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    correct = (pred == target).sum()
    total = target.numel()
    
    return (correct / total).item()


print("✓ Metrics (DSC, IoU, Accuracy) defined!")

# %%
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp=True):
    """Train cho 1 epoch - Optimized"""
    model.train()
    
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Mixed precision training
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        with torch.no_grad():
            batch_dice = dice_coefficient(outputs, masks)
            batch_iou = iou_score(outputs, masks)
        
        running_loss += loss.item()
        running_dice += batch_dice
        running_iou += batch_iou
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{batch_dice:.4f}'
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_dice = running_dice / len(loader)
    epoch_iou = running_iou / len(loader)
    
    return epoch_loss, epoch_dice, epoch_iou


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validation - Optimized"""
    model.eval()
    
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    pbar = tqdm(loader, desc="Validation", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        batch_dice = dice_coefficient(outputs, masks)
        batch_iou = iou_score(outputs, masks)
        
        running_loss += loss.item()
        running_dice += batch_dice
        running_iou += batch_iou
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(loader)
    epoch_dice = running_dice / len(loader)
    epoch_iou = running_iou / len(loader)
    
    return epoch_loss, epoch_dice, epoch_iou


print("✓ Training and validation functions ready (optimized)!")

# %%
# Optimizer with weight decay (L2 regularization)
try:
    optimizer = Adam(
        model.parameters(),
        lr=cfg.LR_INITIAL,
        betas=(cfg.BETA_1, cfg.BETA_2),
        eps=cfg.EPSILON,
        weight_decay=cfg.WEIGHT_DECAY,  # NEW: Add L2 regularization
        fused=True
    )
    print("✓ Using Fused Adam optimizer with weight decay")
except:
    optimizer = Adam(
        model.parameters(),
        lr=cfg.LR_INITIAL,
        betas=(cfg.BETA_1, cfg.BETA_2),
        eps=cfg.EPSILON,
        weight_decay=cfg.WEIGHT_DECAY
    )
    print("✓ Using standard Adam optimizer with weight decay")

# LR Scheduler - Improved based on training analysis
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',  # Changed to max for Dice monitoring
    factor=cfg.LR_REDUCE_FACTOR,
    patience=cfg.LR_REDUCE_PATIENCE,
    min_lr=cfg.LR_MIN,
    verbose=True
)

# Mixed Precision Scaler
scaler = GradScaler(growth_interval=100) if cfg.USE_AMP else None

# Training history - Enhanced tracking
history = {
    'train_loss': [],
    'train_dice': [],
    'train_iou': [],
    'val_loss': [],
    'val_dice': [],
    'val_iou': [],
    'lr': [],
    'epoch_time': []  # NEW: Track training time
}

# Early stopping - Enhanced
best_val_dice = 0.0  # Changed to Dice (higher is better)
best_val_loss = float('inf')
patience_counter = 0
best_epoch = 0

# NEW: Save top K models
saved_models = []  # Track saved model paths

print("\n" + "=" * 60)
print("STARTING TRAINING (OPTIMIZED v2)")
print("=" * 60)
print(f"Max epochs: {cfg.MAX_EPOCHS}")
print(f"Batch size: {cfg.BATCH_SIZE}")
print(f"Initial LR: {cfg.LR_INITIAL}")
print(f"Weight Decay: {cfg.WEIGHT_DECAY}")
print(f"Dropout Rate: {cfg.DROPOUT_RATE}")
print(f"Early stopping patience: {cfg.EARLY_STOPPING_PATIENCE}")
print(f"LR reduce patience: {cfg.LR_REDUCE_PATIENCE}")
print(f"Mixed Precision: {cfg.USE_AMP}")
print(f"Monitor Metric: {cfg.MONITOR_METRIC}")
print("=" * 60 + "\n")

start_time = time.time()

for epoch in range(cfg.MAX_EPOCHS):
    epoch_start = time.time()
    print(f"\nEpoch {epoch+1}/{cfg.MAX_EPOCHS}")
    
    # Train
    train_loss, train_dice, train_iou = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler, device, cfg.USE_AMP
    )
    
    # Validate
    val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)
    
    # Get current LR
    current_lr = optimizer.param_groups[0]['lr']
    
    epoch_time = time.time() - epoch_start
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_dice'].append(train_dice)
    history['train_iou'].append(train_iou)
    history['val_loss'].append(val_loss)
    history['val_dice'].append(val_dice)
    history['val_iou'].append(val_iou)
    history['lr'].append(current_lr)
    history['epoch_time'].append(epoch_time)
    
    # Print epoch results with overfitting indicator
    train_val_gap = train_dice - val_dice
    overfit_indicator = "⚠ Overfitting" if train_val_gap > 0.05 else "✓"
    
    print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
    print(f"Train-Val Gap: {train_val_gap:.4f} {overfit_indicator}")
    print(f"Learning Rate: {current_lr:.2e} | Epoch Time: {epoch_time:.1f}s")
    
    # LR Scheduler step - Monitor Dice score
    scheduler.step(val_dice)
    
    # Model Checkpoint - Save based on Dice score
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        
        # Save checkpoint
        checkpoint_path = cfg.BEST_MODEL_PATH
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_dice': best_val_dice,
            'best_val_loss': best_val_loss,
            'history': history
        }, checkpoint_path)
        print(f"✓ Best model saved! Val Dice: {val_dice:.4f} | Val Loss: {val_loss:.4f}")
        
        # NEW: Save top K models
        model_name = cfg.CHECKPOINT_DIR / f"model_epoch{epoch+1}_dice{val_dice:.4f}.pth"
        torch.save(model.state_dict(), model_name)
        saved_models.append((val_dice, model_name))
        saved_models.sort(reverse=True, key=lambda x: x[0])
        
        # Keep only top K
        if len(saved_models) > cfg.SAVE_TOP_K:
            _, remove_path = saved_models.pop()
            if remove_path.exists():
                remove_path.unlink()
    else:
        patience_counter += 1
        print(f"✗ No improvement. Patience: {patience_counter}/{cfg.EARLY_STOPPING_PATIENCE}")
    
    # Early Stopping
    if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
        print(f"\n⚠ Early stopping triggered after {epoch+1} epochs!")
        break

total_time = time.time() - start_time

print("\n" + "=" * 60)
print("TRAINING COMPLETED!")
print("=" * 60)
print(f"Best Epoch: {best_epoch+1}")
print(f"Best Val Dice: {best_val_dice:.4f}")
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Best Val IoU: {max(history['val_iou']):.4f}")
print(f"Total Training Time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
print(f"Average Time per Epoch: {total_time/(epoch+1):.1f}s")
print(f"Total Epochs Trained: {epoch+1}")
print(f"\nTop {len(saved_models)} models saved:")
for dice, path in saved_models:
    print(f"  - {path.name} (Dice: {dice:.4f})")
print("=" * 60)

# %%
# Load best model
checkpoint = torch.load(cfg.BEST_MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
history = checkpoint['history']

print("=" * 60)
print("BEST MODEL LOADED")
print("=" * 60)
print(f"✓ Loaded from epoch: {checkpoint['epoch']+1}")
print(f"✓ Best Val Loss: {checkpoint['best_val_loss']:.4f}")
print(f"✓ Best Val Dice: {max(history['val_dice']):.4f}")
print(f"✓ Best Val IoU: {max(history['val_iou']):.4f}")
print("=" * 60)

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2, color='#2E86DE')
axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2, color='#EE5A6F')
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Loss Curve', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# Dice
axes[0, 1].plot(history['train_dice'], label='Train Dice', linewidth=2, color='#2E86DE')
axes[0, 1].plot(history['val_dice'], label='Val Dice', linewidth=2, color='#EE5A6F')
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Dice Coefficient', fontsize=12)
axes[0, 1].set_title('Dice Coefficient Curve', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# IoU
axes[1, 0].plot(history['train_iou'], label='Train IoU', linewidth=2, color='#2E86DE')
axes[1, 0].plot(history['val_iou'], label='Val IoU', linewidth=2, color='#EE5A6F')
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('IoU Score', fontsize=12)
axes[1, 0].set_title('IoU Score Curve', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# Learning Rate
axes[1, 1].plot(history['lr'], label='Learning Rate', linewidth=2, color='#F39C12')
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[1, 1].set_yscale('log')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Training history plotted and saved!")

# %%
@torch.no_grad()
def evaluate_on_test(model, loader, device):
    """Đánh giá model trên test set"""
    model.eval()
    
    all_dice = []
    all_iou = []
    all_acc = []
    
    pbar = tqdm(loader, desc="Evaluating on Test Set")
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        outputs = model(images)
        
        batch_dice = dice_coefficient(outputs, masks)
        batch_iou = iou_score(outputs, masks)
        batch_acc = pixel_accuracy(outputs, masks)
        
        all_dice.append(batch_dice)
        all_iou.append(batch_iou)
        all_acc.append(batch_acc)
    
    mean_dice = np.mean(all_dice)
    mean_iou = np.mean(all_iou)
    mean_acc = np.mean(all_acc)
    
    std_dice = np.std(all_dice)
    std_iou = np.std(all_iou)
    std_acc = np.std(all_acc)
    
    return {
        'dice': (mean_dice, std_dice),
        'iou': (mean_iou, std_iou),
        'accuracy': (mean_acc, std_acc)
    }

# Evaluate
print("Evaluating on test set...")
test_results = evaluate_on_test(model, test_loader, device)

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"Dice Coefficient: {test_results['dice'][0]:.4f} ± {test_results['dice'][1]:.4f}")
print(f"IoU Score:        {test_results['iou'][0]:.4f} ± {test_results['iou'][1]:.4f}")
print(f"Pixel Accuracy:   {test_results['accuracy'][0]:.4f} ± {test_results['accuracy'][1]:.4f}")
print("=" * 60)

# %%
def visualize_predictions(model, dataset, device, num_samples=5):
    """Visualize predictions"""
    model.eval()
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 4))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            
            # Prediction
            image_input = image.unsqueeze(0).to(device)
            output = model(image_input)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_binary = (pred_mask > 0.5).astype(np.uint8)
            
            # Convert to numpy for visualization
            image_np = image.permute(1, 2, 0).cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()
            
            # Calculate metrics for this sample
            dice = dice_coefficient(output, mask.unsqueeze(0).to(device))
            iou = iou_score(output, mask.unsqueeze(0).to(device))
            
            # Plot
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title('Input Image', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask, cmap='jet')
            axes[i, 2].set_title('Prediction (Probability)', fontsize=12, fontweight='bold')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(pred_binary, cmap='gray')
            axes[i, 3].set_title(f'Binary (Dice: {dice:.3f}, IoU: {iou:.3f})', 
                                fontsize=12, fontweight='bold')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

# Visualize
print("Generating prediction visualizations...")
visualize_predictions(model, test_dataset, device, num_samples=5)
print("✓ Predictions visualized and saved!")

# %%
def predict_single_image(model, image_path, device, threshold=0.5):
    """
    Inference pipeline cho 1 ảnh theo spec:
    - Resize to (256, 256)
    - Normalize [0, 1]
    - Model prediction
    - Sigmoid
    - Threshold
    - Resize về kích thước gốc
    """
    model.eval()
    
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = (image.shape[1], image.shape[0])  # (W, H)
    
    # Resize to 256x256
    image_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # To tensor
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Threshold
    pred_binary = (pred_prob > threshold).astype(np.uint8) * 255
    
    # Resize về kích thước gốc
    pred_resized = cv2.resize(pred_binary, original_size, interpolation=cv2.INTER_NEAREST)
    
    return pred_resized, pred_prob

print("=" * 60)
print("✓ Inference function ready!")
print("=" * 60)
print("\nAll cells completed successfully!")
print("You can now use predict_single_image() to make predictions on new images.")
print("=" * 60)

# %%
# predict_single_image()


