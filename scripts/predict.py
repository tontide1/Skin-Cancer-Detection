#!/usr/bin/env python3
"""
Inference script — chạy model trên ảnh mới, lưu predicted masks.

Usage:
    # Single image
    python scripts/predict.py \
        --config configs/experiments/resnet34_unet_v1.yaml \
        --checkpoint outputs/resnet34_unet_v1/best_model.pth \
        --input path/to/image.jpg \
        --output outputs/predictions

    # Directory of images
    python scripts/predict.py \
        --config configs/experiments/resnet34_unet_v1.yaml \
        --checkpoint outputs/resnet34_unet_v1/best_model.pth \
        --input path/to/images/ \
        --output outputs/predictions \
        --threshold 0.45 \
        --tta
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.transforms import get_transforms
from src.models.segmentation import create_model
from src.utils.config import load_config, override_config
from src.utils.misc import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def preprocess(image_path: Path, config) -> torch.Tensor:
    """Load + preprocess một ảnh → (1, C, H, W) tensor."""
    image = np.array(Image.open(image_path).convert("RGB"))
    transform = get_transforms("val", config)  # resize + normalize, no augment
    dummy_mask = np.zeros(image.shape[:2], dtype=np.float32)
    out = transform(image=image, mask=dummy_mask)
    tensor = out["image"]                      # (C, H, W)
    return tensor.unsqueeze(0)                 # (1, C, H, W)


@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
    use_tta: bool = False,
) -> np.ndarray:
    """
    Predict mask for a single image tensor.

    Returns:
        Binary mask as uint8 numpy array (H, W) with values 0/255.
    """
    model.eval()
    x = image_tensor.to(device)

    if use_tta:
        probs  = torch.sigmoid(model(x))
        probs += torch.sigmoid(model(torch.flip(x, dims=[3]))).flip(dims=[3])
        probs += torch.sigmoid(model(torch.flip(x, dims=[2]))).flip(dims=[2])
        probs /= 3.0
    else:
        probs = torch.sigmoid(model(x))

    mask = (probs[0, 0] > threshold).cpu().numpy().astype(np.uint8) * 255
    return mask


def save_overlay(
    image_path: Path,
    mask: np.ndarray,
    save_path: Path,
    alpha: float = 0.4,
) -> None:
    """Save side-by-side: original | mask overlay."""
    import matplotlib.pyplot as plt

    orig = np.array(Image.open(image_path).convert("RGB"))
    # Prediction mask được tạo ở input_size của model; resize về ảnh gốc để tránh shape mismatch.
    if mask.shape != orig.shape[:2]:
        orig_h, orig_w = orig.shape[:2]
        mask = np.array(
            Image.fromarray(mask).resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
        )

    mask_rgb = np.zeros_like(orig)
    mask_rgb[:, :, 0] = mask  # red channel

    h_mask = mask > 127
    overlay = orig.copy().astype(float)
    overlay[h_mask] = (1 - alpha) * orig[h_mask] + alpha * mask_rgb[h_mask]
    overlay = overlay.astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig);      axes[0].set_title("Original");  axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Predicted Mask"); axes[1].axis("off")
    axes[2].imshow(overlay);   axes[2].set_title("Overlay");  axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on new images",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config",     "-c", required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", "-k", required=True,
                        help="Path to best_model.pth")
    parser.add_argument("--input",      "-i", required=True,
                        help="Input image file or directory of images")
    parser.add_argument("--output",     "-o", default="outputs/predictions",
                        help="Output directory for predicted masks")
    parser.add_argument("--threshold",  "-t", type=float, default=0.5,
                        help="Binarization threshold (default: 0.5)")
    parser.add_argument("--tta",  dest="tta", action="store_true",  default=False,
                        help="Enable TTA (default: off)")
    parser.add_argument("--overlay", action="store_true", default=False,
                        help="Also save overlay visualization PNG")
    parser.add_argument("overrides", nargs="*", metavar="key.subkey=value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    config = override_config(config, args.overrides)

    set_seed(config.seed)
    device = get_device()

    # Collect images
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
    else:
        log.error(f"Input path not found: {input_path}")
        sys.exit(1)

    if not image_paths:
        log.error("No .jpg/.png images found.")
        sys.exit(1)

    log.info(f"Found {len(image_paths)} image(s) | threshold={args.threshold} | TTA={args.tta}")

    # Model
    model = create_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    # Output dir
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predict
    for img_path in tqdm(image_paths, desc="Predicting"):
        tensor = preprocess(img_path, config)
        mask   = predict_single(model, tensor, device, args.threshold, args.tta)

        # Save mask
        mask_save = out_dir / f"{img_path.stem}_pred_mask.png"
        Image.fromarray(mask).save(mask_save)

        # Optionally save overlay
        if args.overlay:
            overlay_save = out_dir / f"{img_path.stem}_overlay.png"
            save_overlay(img_path, mask, overlay_save)

    log.info(f"Done. Predictions saved to: {out_dir}")


if __name__ == "__main__":
    main()
