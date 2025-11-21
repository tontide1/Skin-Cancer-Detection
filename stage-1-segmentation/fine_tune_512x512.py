# %%
"""
PROGRESSIVE RESIZING: Fine-tuning from 256x256 to 512x512
==========================================================
Tiếp tục training với resolution cao hơn để cải thiện edge detection và IoU
"""

# %%
"""
Cell 1: Fine-tuning Configuration (512x512)
"""

import copy

# Create fine-tuning config based on original
FINE_TUNE_CONFIG = copy.deepcopy(CONFIG)

# Update for progressive resizing
FINE_TUNE_CONFIG['data']['input_size'] = (512, 512)  # Double the resolution
FINE_TUNE_CONFIG['training']['batch_size'] = 4  # Reduce for memory (conservative)
FINE_TUNE_CONFIG['training']['lr'] = 1e-5  # 10x smaller to preserve learned features
FINE_TUNE_CONFIG['training']['max_epochs'] = 20  # Fewer epochs for fine-tuning
FINE_TUNE_CONFIG['lr_schedule']['patience'] = 3  # Reduce patience for faster adaptation
FINE_TUNE_CONFIG['early_stopping']['patience'] = 5  # Early stop if no improvement

# Paths for fine-tuned model
FINE_TUNE_CONFIG['paths']['pretrained_model'] = CONFIG['paths']['output_dir'] / 'best_model.pth'
FINE_TUNE_CONFIG['paths']['finetuned_model'] = CONFIG['paths']['output_dir'] / 'best_model_512.pth'

print("="*70)
print("🔥 PROGRESSIVE RESIZING: FINE-TUNING CONFIGURATION")
print("="*70)
print(f"\n📊 Resolution Upgrade:")
print(f"  Original:    {CONFIG['data']['input_size']}")
print(f"  Fine-tuning: {FINE_TUNE_CONFIG['data']['input_size']}")
print(f"\n⚙️  Fine-tuning Settings:")
print(f"  Batch size:   {FINE_TUNE_CONFIG['training']['batch_size']} (reduced from {CONFIG['training']['batch_size']})")
print(f"  Learning rate: {FINE_TUNE_CONFIG['training']['lr']:.2e} (10x lower)")
print(f"  Max epochs:   {FINE_TUNE_CONFIG['training']['max_epochs']}")
print(f"  Device:       {FINE_TUNE_CONFIG['hardware']['device']}")
print(f"\n📁 Model Paths:")
print(f"  Load from:  {FINE_TUNE_CONFIG['paths']['pretrained_model']}")
print(f"  Save to:    {FINE_TUNE_CONFIG['paths']['finetuned_model']}")
print("="*70)

# %%
"""
Cell 2: Re-create datasets and dataloaders with 512x512
"""

print("\n" + "="*70)
print("📦 CREATING DATASETS FOR 512x512 RESOLUTION")
print("="*70)

# Create datasets with new resolution
train_dataset_512 = ISICDataset(
    IMG_DIRS['train'], 
    MASK_DIRS['train'], 
    transform=get_transforms('train', FINE_TUNE_CONFIG['data']['input_size'])
)
val_dataset_512 = ISICDataset(
    IMG_DIRS['val'], 
    MASK_DIRS['val'], 
    transform=get_transforms('val', FINE_TUNE_CONFIG['data']['input_size'])
)

print(f"\n📊 Dataset sizes:")
print(f"  Train: {len(train_dataset_512):,} images at {FINE_TUNE_CONFIG['data']['input_size']}")
print(f"  Val:   {len(val_dataset_512):,} images at {FINE_TUNE_CONFIG['data']['input_size']}")

# Create dataloaders with reduced batch size
train_loader_512 = DataLoader(
    train_dataset_512,
    batch_size=FINE_TUNE_CONFIG['training']['batch_size'],
    shuffle=True,
    num_workers=FINE_TUNE_CONFIG['data']['num_workers'],
    pin_memory=FINE_TUNE_CONFIG['data']['pin_memory'],
    persistent_workers=FINE_TUNE_CONFIG['data']['persistent_workers'],
)
val_loader_512 = DataLoader(
    val_dataset_512,
    batch_size=FINE_TUNE_CONFIG['training']['batch_size'],
    shuffle=False,
    num_workers=FINE_TUNE_CONFIG['data']['num_workers'],
    pin_memory=FINE_TUNE_CONFIG['data']['pin_memory'],
    persistent_workers=FINE_TUNE_CONFIG['data']['persistent_workers'],
)

print(f"\n🔄 DataLoaders created:")
print(f"  Train batches: {len(train_loader_512)} (batch_size={FINE_TUNE_CONFIG['training']['batch_size']})")
print(f"  Val batches:   {len(val_loader_512)}")

# Verify sample shape
sample_img_512, sample_mask_512 = train_dataset_512[0]
print(f"\n🖼️  Sample shapes:")
print(f"  Image: {sample_img_512.shape} ← 4x larger than 256x256")
print(f"  Mask:  {sample_mask_512.shape}")

# Estimate memory usage
images_per_gpu = FINE_TUNE_CONFIG['training']['batch_size'] // FINE_TUNE_CONFIG['hardware']['num_gpus']
estimated_memory_gb = (images_per_gpu * 3 * 512 * 512 * 4 * 2) / (1024**3)  # Forward + backward
print(f"\n💾 Estimated memory per GPU: ~{estimated_memory_gb:.2f} GB (should fit in T4 15GB)")

print("✅ Datasets and loaders ready for 512x512 training!")

# %%
"""
Cell 3: Load pretrained model and weights from 256x256
"""

print("\n" + "="*70)
print("🔄 LOADING PRETRAINED MODEL (256x256 → 512x512)")
print("="*70)

# Create fresh model (same architecture, no weights yet)
model_512 = create_model(FINE_TUNE_CONFIG)

# Load checkpoint from 256x256 training
checkpoint_path = FINE_TUNE_CONFIG['paths']['pretrained_model']
print(f"\n📂 Loading checkpoint from: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(f"✅ Checkpoint loaded successfully")
print(f"   Trained epoch: {checkpoint['epoch']}")
print(f"   Val Dice: {checkpoint['val_dice']:.4f}")
print(f"   Val IoU: {checkpoint.get('val_iou', 'N/A')}")

# Handle state dict (remove 'module.' prefix if present)
state_dict = checkpoint['model_state_dict']
if list(state_dict.keys())[0].startswith('module.'):
    # Remove 'module.' prefix from DataParallel
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    print("   ⚙️  Removed 'module.' prefix from state dict")

# Load weights into model
model_512.load_state_dict(state_dict)
print("✅ Pretrained weights loaded successfully!")
print("   Note: U-Net is fully convolutional → can handle different input sizes")

# Move to device and wrap with DataParallel
device = torch.device(FINE_TUNE_CONFIG['hardware']['device'])
if FINE_TUNE_CONFIG['hardware']['num_gpus'] > 1:
    model_512 = nn.DataParallel(model_512)
    print(f"🚀 Using {FINE_TUNE_CONFIG['hardware']['num_gpus']} GPUs with DataParallel")

model_512 = model_512.to(device)

# Count parameters
total_params = sum(p.numel() for p in model_512.parameters())
trainable_params = sum(p.numel() for p in model_512.parameters() if p.requires_grad)

print(f"\n🏗️  Model Architecture:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size: ~{total_params * 4 / 1024**2:.1f} MB (FP32)")

print("\n✅ Model ready for fine-tuning at 512x512 resolution!")

# %%
"""
Cell 4: Helper functions to freeze/unfreeze encoder
"""

def freeze_encoder(model):
    """
    Freeze ResNet34 encoder to preserve learned features
    Only train decoder and segmentation head
    """
    # Get the actual model (handle DataParallel wrapper)
    model_to_freeze = model.module if isinstance(model, nn.DataParallel) else model
    
    # Freeze encoder
    if hasattr(model_to_freeze, 'encoder'):
        for param in model_to_freeze.encoder.parameters():
            param.requires_grad = False
        frozen_params = sum(p.numel() for p in model_to_freeze.encoder.parameters())
        print(f"🔒 Encoder frozen: {frozen_params:,} parameters")
    else:
        print("⚠️  Warning: Could not find encoder attribute")
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def unfreeze_all(model):
    """
    Unfreeze all parameters for full fine-tuning
    """
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔓 All parameters unfrozen: {trainable:,} parameters trainable")


print("✅ Freeze/unfreeze helper functions defined!")

# %%
"""
Cell 5: Setup optimizer and scheduler for fine-tuning
"""

# Initialize optimizer with lower learning rate
optimizer_512 = torch.optim.AdamW(
    model_512.parameters(),
    lr=FINE_TUNE_CONFIG['training']['lr'],
    weight_decay=FINE_TUNE_CONFIG['training']['weight_decay']
)

# Initialize scheduler
scheduler_512 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_512,
    mode=FINE_TUNE_CONFIG['lr_schedule']['mode'],
    factor=FINE_TUNE_CONFIG['lr_schedule']['factor'],
    patience=FINE_TUNE_CONFIG['lr_schedule']['patience'],
    min_lr=FINE_TUNE_CONFIG['lr_schedule']['min_lr']
)

print("✅ Optimizer and scheduler created for fine-tuning!")
print(f"  Optimizer: {FINE_TUNE_CONFIG['training']['optimizer']}")
print(f"  Fine-tune LR: {FINE_TUNE_CONFIG['training']['lr']:.2e}")
print(f"  Weight decay: {FINE_TUNE_CONFIG['training']['weight_decay']}")
print(f"  Scheduler patience: {FINE_TUNE_CONFIG['lr_schedule']['patience']} epochs")

# %%
"""
Cell 6: Fine-tuning training loop with progressive freezing
"""

print("\n" + "="*70)
print("🔥 STARTING PROGRESSIVE RESIZING FINE-TUNING (512x512)")
print("="*70)

# Training strategy
FREEZE_EPOCHS = 3  # Freeze encoder for first 3 epochs
TOTAL_EPOCHS = FINE_TUNE_CONFIG['training']['max_epochs']

print(f"\n📋 Training Strategy:")
print(f"  Epochs 1-{FREEZE_EPOCHS}: Encoder frozen (train decoder only)")
print(f"  Epochs {FREEZE_EPOCHS+1}-{TOTAL_EPOCHS}: Full model training (all layers)")
print("="*70)

# Initialize training components
scaler_512 = GradScaler(enabled=FINE_TUNE_CONFIG['training']['mixed_precision'])
early_stopping_512 = EarlyStopping(
    patience=FINE_TUNE_CONFIG['early_stopping']['patience'],
    min_delta=FINE_TUNE_CONFIG['early_stopping']['min_delta'],
    mode=FINE_TUNE_CONFIG['early_stopping']['mode']
)

# History tracking
history_512 = {
    'train_loss': [],
    'train_dice': [],
    'train_iou': [],
    'val_loss': [],
    'val_dice': [],
    'val_iou': [],
    'lr': [],
}

# Initialize best metric from pretrained model
best_val_dice_512 = checkpoint['val_dice']  # Start from 256x256 performance
print(f"\n🎯 Baseline Val Dice from 256x256: {best_val_dice_512:.4f}")
print("   Goal: Improve this with 512x512 resolution!")

best_model_path_512 = FINE_TUNE_CONFIG['paths']['finetuned_model']

# Training loop
for epoch in range(TOTAL_EPOCHS):
    print(f"\n{'='*70}")
    print(f"📍 Fine-tuning Epoch {epoch+1}/{TOTAL_EPOCHS} at 512x512")
    print(f"{'='*70}")
    
    # Freeze/unfreeze strategy
    if epoch == 0:
        print("\n🔒 Phase 1: Freezing encoder (training decoder only)")
        freeze_encoder(model_512)
    elif epoch == FREEZE_EPOCHS:
        print(f"\n🔓 Phase 2: Unfreezing all layers (full fine-tuning)")
        unfreeze_all(model_512)
        # Lower LR further when unfreezing
        for param_group in optimizer_512.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
        print(f"   Learning rate adjusted to: {optimizer_512.param_groups[0]['lr']:.2e}")
    
    print("-" * 70)
    
    # Train
    train_loss, train_dice, train_iou = train_one_epoch(
        model_512, train_loader_512, optimizer_512, scaler_512, device,
        FINE_TUNE_CONFIG['training']['loss_weights']
    )
    
    # Validate
    val_loss, val_dice, val_iou = validate(
        model_512, val_loader_512, device,
        FINE_TUNE_CONFIG['training']['loss_weights']
    )
    
    # Learning rate
    current_lr = optimizer_512.param_groups[0]['lr']
    
    # Update history
    history_512['train_loss'].append(train_loss)
    history_512['train_dice'].append(train_dice)
    history_512['train_iou'].append(train_iou)
    history_512['val_loss'].append(val_loss)
    history_512['val_dice'].append(val_dice)
    history_512['val_iou'].append(val_iou)
    history_512['lr'].append(current_lr)
    
    # Print epoch summary
    print(f"\n📊 Epoch {epoch+1} Summary:")
    print(f"  Train → Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}")
    print(f"  Val   → Loss: {val_loss:.4f} | Dice: {val_dice:.4f} ⭐ | IoU: {val_iou:.4f}")
    print(f"  LR: {current_lr:.2e}")
    
    # Calculate improvement over baseline
    dice_improvement = val_dice - best_val_dice_512
    if dice_improvement > 0:
        print(f"  📈 Improvement: +{dice_improvement:.4f} over baseline!")
    
    # Save best model
    if val_dice > best_val_dice_512:
        best_val_dice_512 = val_dice
        
        # Get state dict (handle DataParallel)
        save_state_dict = model_512.module.state_dict() if isinstance(model_512, nn.DataParallel) else model_512.state_dict()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': save_state_dict,
            'optimizer_state_dict': optimizer_512.state_dict(),
            'val_dice': val_dice,
            'val_iou': val_iou,
            'config': FINE_TUNE_CONFIG,
            'history': history_512,
            'pretrained_from': str(checkpoint_path),
            'input_size': FINE_TUNE_CONFIG['data']['input_size'],
        }, best_model_path_512)
        print(f"  ✅ New best model saved! (Val Dice: {val_dice:.4f})")
    
    # Learning rate scheduling
    scheduler_512.step(val_loss)
    
    # Early stopping
    if early_stopping_512(val_dice):
        print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
        print(f"  Best Val Dice: {best_val_dice_512:.4f}")
        break

print("\n" + "="*70)
print("✅ FINE-TUNING COMPLETED!")
print("="*70)
print(f"\n🏆 Results:")
print(f"  Baseline (256x256): {checkpoint['val_dice']:.4f}")
print(f"  Fine-tuned (512x512): {best_val_dice_512:.4f}")
print(f"  Improvement: {best_val_dice_512 - checkpoint['val_dice']:+.4f}")
print(f"\n📁 Fine-tuned model saved to: {best_model_path_512}")

# %%
"""
Cell 7: Plot fine-tuning curves and comparison
"""

def plot_finetuning_curves(history_512, baseline_dice, save_path):
    """Plot fine-tuning curves with baseline comparison"""
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    epochs = range(1, len(history_512['train_loss']) + 1)
    
    # 1. Dice Score with baseline
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, history_512['train_dice'], 'b-o', label='Train Dice (512x512)', linewidth=2.5, markersize=5)
    ax1.plot(epochs, history_512['val_dice'], 'r-s', label='Val Dice (512x512)', linewidth=2.5, markersize=5)
    ax1.axhline(y=baseline_dice, color='orange', linestyle='--', linewidth=2, label=f'Baseline (256x256): {baseline_dice:.4f}')
    best_val_idx = np.argmax(history_512['val_dice'])
    ax1.scatter([best_val_idx+1], [history_512['val_dice'][best_val_idx]], 
                color='gold', s=200, zorder=5, edgecolors='darkgreen', linewidth=2,
                label=f'Best: {history_512["val_dice"][best_val_idx]:.4f}')
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Dice Coefficient', fontsize=14, fontweight='bold')
    ax1.set_title('🔥 DICE SCORE - Progressive Resizing (512x512)', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([max(0.7, min(history_512['val_dice'])-0.05), 1.0])
    
    # 2. IoU Score
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(epochs, history_512['train_iou'], 'b-o', label='Train IoU', linewidth=2, markersize=4)
    ax2.plot(epochs, history_512['val_iou'], 'r-s', label='Val IoU', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('IoU Score', fontsize=12)
    ax2.set_title('IoU at 512x512', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, history_512['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax3.plot(epochs, history_512['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Loss Curve', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning Rate
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, history_512['lr'], 'g-^', linewidth=2.5, markersize=5)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Learning Rate', fontsize=12)
    ax4.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, which='both')
    
    # 5. Improvement metrics
    ax5 = fig.add_subplot(gs[1, 2])
    final_dice = history_512['val_dice'][-1]
    best_dice = max(history_512['val_dice'])
    final_iou = history_512['val_iou'][-1]
    
    metrics = ['Baseline\n256x256', 'Final\n512x512', 'Best\n512x512']
    values = [baseline_dice, final_dice, best_dice]
    colors = ['#FFA500', '#3498db', '#2ecc71']
    bars = ax5.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Dice Score', fontsize=12)
    ax5.set_title('Dice Comparison', fontsize=13, fontweight='bold')
    ax5.set_ylim([max(0.7, min(values)-0.05), 1.0])
    ax5.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('🏥 Progressive Resizing: 256→512 Fine-tuning Results', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Fine-tuning curves saved to: {save_path}")
    plt.show()


# Plot curves
plot_finetuning_curves(
    history_512, 
    checkpoint['val_dice'],
    FINE_TUNE_CONFIG['paths']['output_dir'] / 'finetuning_curves_512.png'
)

# %%
"""
Cell 8: Save fine-tuning history and summary
"""

# Save history as JSON
history_512_path = FINE_TUNE_CONFIG['paths']['output_dir'] / 'finetuning_history_512.json'
with open(history_512_path, 'w') as f:
    json.dump(history_512, f, indent=4)
print(f"✅ Fine-tuning history (JSON) saved to: {history_512_path}")

# Save history as CSV
history_512_df = pd.DataFrame(history_512)
history_512_df['epoch'] = range(1, len(history_512_df) + 1)
history_512_csv_path = FINE_TUNE_CONFIG['paths']['output_dir'] / 'finetuning_history_512.csv'
history_512_df.to_csv(history_512_csv_path, index=False)
print(f"✅ Fine-tuning history (CSV) saved to: {history_512_csv_path}")

# Print comprehensive summary
print("\n" + "="*70)
print("📊 PROGRESSIVE RESIZING SUMMARY")
print("="*70)

print(f"\n🎯 Performance Comparison:")
print(f"  {'Metric':<20} {'256x256 (Baseline)':<20} {'512x512 (Fine-tuned)':<20} {'Improvement':<15}")
print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*15}")
print(f"  {'Val Dice':<20} {checkpoint['val_dice']:<20.4f} {best_val_dice_512:<20.4f} {best_val_dice_512 - checkpoint['val_dice']:+.4f}")
print(f"  {'Val IoU':<20} {checkpoint.get('val_iou', 0):<20.4f} {history_512['val_iou'][np.argmax(history_512['val_dice'])]:<20.4f} {history_512['val_iou'][np.argmax(history_512['val_dice'])] - checkpoint.get('val_iou', 0):+.4f}")

print(f"\n🏆 Best Results at 512x512:")
best_epoch_512 = np.argmax(history_512['val_dice']) + 1
print(f"  Best Epoch: {best_epoch_512}")
print(f"  Best Val Dice: {max(history_512['val_dice']):.4f}")
print(f"  Best Val IoU:  {history_512['val_iou'][best_epoch_512-1]:.4f}")

print(f"\n📈 Training Statistics:")
print(f"  Total fine-tuning epochs: {len(history_512['train_loss'])}")
print(f"  Frozen encoder epochs: {FREEZE_EPOCHS}")
print(f"  Full training epochs: {len(history_512['train_loss']) - FREEZE_EPOCHS}")
print(f"  Final learning rate: {history_512['lr'][-1]:.2e}")

print(f"\n📁 Saved Files:")
print(f"  Model:   {best_model_path_512}")
print(f"  Curves:  {FINE_TUNE_CONFIG['paths']['output_dir'] / 'finetuning_curves_512.png'}")
print(f"  History: {history_512_path}")
print(f"  CSV:     {history_512_csv_path}")

print("\n" + "="*70)
print("✅ PROGRESSIVE RESIZING COMPLETED SUCCESSFULLY!")
print("="*70)

improvement_pct = ((best_val_dice_512 - checkpoint['val_dice']) / checkpoint['val_dice']) * 100
if improvement_pct > 0:
    print(f"\n🎉 Success! Improved Dice score by {improvement_pct:.2f}%")
    print(f"   Higher resolution → Better edge detection & IoU!")
else:
    print(f"\n⚠️  No improvement detected. Consider:")
    print(f"   - Training for more epochs")
    print(f"   - Adjusting learning rate")
    print(f"   - Different freezing strategy")

# %%
"""
Cell 9: Test fine-tuned model on test set (512x512)
"""

print("\n" + "="*70)
print("🧪 TESTING FINE-TUNED MODEL (512x512) ON TEST SET")
print("="*70)

# Create test dataset at 512x512
test_dataset_512 = ISICDataset(
    IMG_DIRS['test'], 
    MASK_DIRS['test'], 
    transform=get_transforms('test', FINE_TUNE_CONFIG['data']['input_size'])
)

test_loader_512 = DataLoader(
    test_dataset_512,
    batch_size=FINE_TUNE_CONFIG['training']['batch_size'],
    shuffle=False,
    num_workers=FINE_TUNE_CONFIG['data']['num_workers'],
    pin_memory=FINE_TUNE_CONFIG['data']['pin_memory'],
)

print(f"\n📦 Test dataset:")
print(f"   Samples: {len(test_dataset_512):,} at {FINE_TUNE_CONFIG['data']['input_size']}")
print(f"   Batches: {len(test_loader_512)}")

# Load best fine-tuned model
print(f"\n📂 Loading best fine-tuned model from: {best_model_path_512}")
checkpoint_512 = torch.load(best_model_path_512, map_location=device)

# Create model and load weights
test_model_512 = create_model(FINE_TUNE_CONFIG)
test_model_512.load_state_dict(checkpoint_512['model_state_dict'])
test_model_512 = test_model_512.to(device)
test_model_512.eval()

print(f"✅ Model loaded (Epoch {checkpoint_512['epoch']}, Val Dice: {checkpoint_512['val_dice']:.4f})")

# Evaluate on test set
print(f"\n⏳ Evaluating on test set...")
test_metrics_512 = test_model(test_model_512, test_loader_512, device)

# Print results
print("\n" + "="*70)
print("📊 TEST SET RESULTS (512x512 Fine-tuned Model)")
print("="*70)
print(f"\n🎯 Overall Metrics:")
print(f"   Test Dice Score: {test_metrics_512['mean_dice']:.4f} ± {test_metrics_512['std_dice']:.4f}")
print(f"   Test IoU Score:  {test_metrics_512['mean_iou']:.4f} ± {test_metrics_512['std_iou']:.4f}")
print(f"   Test Loss:       {test_metrics_512['mean_loss']:.4f}")

print(f"\n📈 Distribution Statistics:")
print(f"   Min Dice: {min(test_metrics_512['dice_scores']):.4f}")
print(f"   Max Dice: {max(test_metrics_512['dice_scores']):.4f}")
print(f"   Median Dice: {np.median(test_metrics_512['dice_scores']):.4f}")

# Save test results
test_results_512_path = FINE_TUNE_CONFIG['paths']['output_dir'] / 'test_results_512.json'
with open(test_results_512_path, 'w') as f:
    json.dump({
        'test_metrics': {
            'mean_dice': float(test_metrics_512['mean_dice']),
            'std_dice': float(test_metrics_512['std_dice']),
            'mean_iou': float(test_metrics_512['mean_iou']),
            'std_iou': float(test_metrics_512['std_iou']),
            'mean_loss': float(test_metrics_512['mean_loss']),
            'min_dice': float(min(test_metrics_512['dice_scores'])),
            'max_dice': float(max(test_metrics_512['dice_scores'])),
            'median_dice': float(np.median(test_metrics_512['dice_scores'])),
        },
        'num_test_samples': len(test_dataset_512),
        'model_checkpoint_epoch': checkpoint_512['epoch'],
        'input_size': FINE_TUNE_CONFIG['data']['input_size'],
        'pretrained_from': checkpoint_512.get('pretrained_from', 'Unknown'),
    }, f, indent=2)

print(f"\n✅ Test results saved to: {test_results_512_path}")
print("\n" + "="*70)
print("✅ TESTING COMPLETED!")
print("="*70)

