"""Các utility functions dùng chung."""

from __future__ import annotations

import random
import numpy as np
from pathlib import Path

import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """Set random seed để reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Trả về device khả dụng (CUDA > CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> dict:
    """Đếm số parameters của model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "size_mb": total * 4 / 1024**2,  # FP32
    }


def plot_training_curves(history: list[dict], save_path: Path) -> None:
    """
    Vẽ 4 đồ thị: Loss, Dice, IoU, LR.

    Args:
        history: List các dict metric theo từng epoch
        save_path: Nơi lưu ảnh PNG
    """
    epochs = list(range(1, len(history) + 1))

    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_dice = [h["train_dice"] for h in history]
    val_dice = [h["val_dice"] for h in history]
    train_iou = [h["train_iou"] for h in history]
    val_iou = [h["val_iou"] for h in history]
    lr = [h["lr"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, val_loss, "r-", label="Val Loss", linewidth=2)
    ax.set_title("Combined Loss (Focal + Dice)", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # Dice
    ax = axes[0, 1]
    ax.plot(epochs, train_dice, "b-", label="Train Dice", linewidth=2)
    ax.plot(epochs, val_dice, "r-", label="Val Dice", linewidth=2)
    if val_dice:
        best_ep = int(np.argmax(val_dice))
        ax.plot(
            best_ep + 1,
            val_dice[best_ep],
            "g*",
            markersize=14,
            label=f"Best: {val_dice[best_ep]:.4f}",
        )
    ax.set_title("Dice Coefficient", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # IoU
    ax = axes[1, 0]
    ax.plot(epochs, train_iou, "b-", label="Train IoU", linewidth=2)
    ax.plot(epochs, val_iou, "r-", label="Val IoU", linewidth=2)
    ax.set_title("IoU Score (Jaccard)", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # LR
    ax = axes[1, 1]
    ax.plot(epochs, lr, "g-", label="Learning Rate", linewidth=2)
    ax.set_title("Learning Rate Schedule", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def denormalize(tensor: "torch.Tensor") -> np.ndarray:
    """
    Denormalize ảnh từ ImageNet normalization về [0, 1].
    tensor: (C, H, W) float tensor
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * std + mean, 0, 1)
