"""
Script để trực quan hóa training history từ file CSV
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_training_curves(csv_path: Path, save_path: Path):
    """
    Vẽ biểu đồ training curves với layout 2x2 gọn gàng
    
    Args:
        csv_path: Đường dẫn đến file training_history.csv
        save_path: Đường dẫn để lưu biểu đồ
    """
    # Đọc dữ liệu từ CSV
    df = pd.read_csv(csv_path)
    
    # Tạo dictionary history từ DataFrame
    history = {
        'train_loss': df['train_loss'].values,
        'train_dice': df['train_dice'].values,
        'train_iou': df['train_iou'].values,
        'val_loss': df['val_loss'].values,
        'val_dice': df['val_dice'].values,
        'val_iou': df['val_iou'].values,
        'lr': df['lr'].values,
        'epoch': df['epoch'].values
    }
    
    # Tạo figure với 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Loss Curve
    axes[0, 0].plot(history['epoch'], history['train_loss'], 
                    label='Train Loss', linewidth=2, color='#2E86DE')
    axes[0, 0].plot(history['epoch'], history['val_loss'], 
                    label='Val Loss', linewidth=2, color='#EE5A6F')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss Curve', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice Coefficient
    axes[0, 1].plot(history['epoch'], history['train_dice'], 
                    label='Train Dice', linewidth=2, color='#2E86DE')
    axes[0, 1].plot(history['epoch'], history['val_dice'], 
                    label='Val Dice', linewidth=2, color='#EE5A6F')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Dice Coefficient', fontsize=12)
    axes[0, 1].set_title('Dice Coefficient Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU Score
    axes[1, 0].plot(history['epoch'], history['train_iou'], 
                    label='Train IoU', linewidth=2, color='#2E86DE')
    axes[1, 0].plot(history['epoch'], history['val_iou'], 
                    label='Val IoU', linewidth=2, color='#EE5A6F')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('IoU Score', fontsize=12)
    axes[1, 0].set_title('IoU Score Curve', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate Schedule
    axes[1, 1].plot(history['epoch'], history['lr'], 
                    label='Learning Rate', linewidth=2, color='#F39C12')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Biểu đồ training curves đã được lưu tại: {save_path}")
    
    # In ra thống kê tóm tắt
    print(f"\n📊 Tóm tắt kết quả Training:")
    print(f"  Tổng số epochs: {len(history['epoch'])}")
    print(f"  Best Val Dice: {history['val_dice'].max():.4f} (Epoch {history['epoch'][history['val_dice'].argmax()]})")
    print(f"  Best Val IoU: {history['val_iou'].max():.4f} (Epoch {history['epoch'][history['val_iou'].argmax()]})")
    print(f"  Best Val Loss: {history['val_loss'].min():.4f} (Epoch {history['epoch'][history['val_loss'].argmin()]})")
    print(f"  Final Val Dice: {history['val_dice'][-1]:.4f}")
    print(f"  Final Val IoU: {history['val_iou'][-1]:.4f}")
    print(f"  Final Learning Rate: {history['lr'][-1]}")
    
    plt.show()
    
    return history


if __name__ == "__main__":
    # Định nghĩa đường dẫn
    results_dir = Path("results/")
    csv_path = results_dir / "training_history.csv"
    save_path = results_dir / "training_curves_visualization.png"
    
    # Kiểm tra file tồn tại
    if not csv_path.exists():
        print(f" Không tìm thấy file: {csv_path}")
        print("   Vui lòng kiểm tra đường dẫn!")
    else:
        print(f" Đọc file: {csv_path}")
        history = plot_training_curves(csv_path, save_path)

