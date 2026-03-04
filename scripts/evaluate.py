#!/usr/bin/env python3
"""
Evaluation script — chạy trên test set với TTA + threshold optimization.

Usage (local):
    python scripts/evaluate.py \
        --config configs/experiments/resnet34_unet_v1.yaml \
        --checkpoint outputs/resnet34_unet_v1/best_model.pth

Usage (Kaggle):
    ! python scripts/evaluate.py \
        --config configs/experiments/resnet34_unet_v1.yaml \
        --checkpoint /kaggle/working/resnet34_unet_v1/best_model.pth \
        data.root=/kaggle/input/isic-2018 \
        output.dir=/kaggle/working \
        logging.use_wandb=false

Options:
    --tta       Enable Test-Time Augmentation (horizontal + vertical flip)
    --no-tta    Disable TTA (default: enabled)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import ISICDataset
from src.data.transforms import get_transforms
from src.inference.tta import tta_predict
from src.losses.segmentation import CombinedLoss
from src.models.segmentation import create_model
from src.utils.checkpoint import load_state_dict_with_aux_compat
from src.utils.config import load_config, override_config
from src.utils.misc import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: CombinedLoss,
    threshold: float = 0.5,
    use_tta: bool = True,
) -> dict:
    """Run full evaluation pass. Returns metric dict."""
    model.eval()
    total_loss = 0.0
    thresholds = [round(t, 2) for t in torch.arange(0.3, 0.71, 0.05).tolist()]
    dice_sums = [0.0 for _ in thresholds]
    iou_sums = [0.0 for _ in thresholds]
    total_samples = 0

    for images, masks in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)

        if use_tta:
            probs = tta_predict(model, images)
            logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))  # probs → logits for loss
        else:
            logits = model(images)
            probs = torch.sigmoid(logits)

        loss = criterion(logits, masks)

        n = images.size(0)
        total_loss += loss.item() * n
        total_samples += n

        probs_flat = probs.view(n, -1)
        masks_flat = masks.float().view(n, -1)
        target_sum = masks_flat.sum(dim=1)

        for idx, thr in enumerate(thresholds):
            pred_flat = (probs_flat > thr).float()
            intersection = (pred_flat * masks_flat).sum(dim=1)
            pred_sum = pred_flat.sum(dim=1)

            dice = (2.0 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7)
            iou = (intersection + 1e-7) / (pred_sum + target_sum - intersection + 1e-7)

            dice_sums[idx] += float(dice.sum().item())
            iou_sums[idx] += float(iou.sum().item())

    if total_samples == 0:
        raise RuntimeError("Loader rỗng, không có sample để evaluate.")

    mean_dice = [s / total_samples for s in dice_sums]
    mean_iou = [s / total_samples for s in iou_sums]
    idx_05 = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - threshold))
    best_idx = max(range(len(thresholds)), key=lambda i: mean_dice[i])

    return {
        "loss": total_loss / total_samples,
        "dice": mean_dice[idx_05],
        "iou": mean_iou[idx_05],
        "best_threshold": thresholds[best_idx],
        "best_dice_at_best_thr": mean_dice[best_idx],
        "best_iou_at_best_thr": mean_iou[best_idx],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation model on test set",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--checkpoint", "-k", required=True, help="Path to best_model.pth checkpoint"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--tta",
        dest="tta",
        action="store_true",
        default=True,
        help="Use Test-Time Augmentation (default: on)",
    )
    parser.add_argument("--no-tta", dest="tta", action="store_false", help="Disable TTA")
    parser.add_argument("overrides", nargs="*", metavar="key.subkey=value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    config = override_config(config, args.overrides)
    if not config.logging.experiment_name:
        config["logging"]["experiment_name"] = Path(args.config).stem

    set_seed(config.seed, deterministic=bool(getattr(config.training, "deterministic", True)))
    device = get_device()
    log.info(f"Device: {device} | TTA: {args.tta} | Split: {args.split}")

    # Dataset
    root = Path(config.data.root)
    dataset = ISICDataset(
        img_dir=root / args.split / "images",
        mask_dir=root / args.split / "masks",
        transform=get_transforms("val", config),  # no augmentation for eval
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    log.info(f"{args.split} set: {len(dataset)} samples")

    # Model
    model = create_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    load_state_dict_with_aux_compat(model, state, context=str(args.checkpoint))
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    # Loss
    criterion = CombinedLoss(config)

    # Evaluate
    results = evaluate(model, loader, device, criterion, use_tta=args.tta)

    # Print
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS ({args.split.upper()} SET)")
    print("=" * 60)
    print(f"  Loss:                    {results['loss']:.4f}")
    print(f"  Dice  @ thr=0.50:        {results['dice']:.4f}")
    print(f"  IoU   @ thr=0.50:        {results['iou']:.4f}")
    print(f"  Best threshold:          {results['best_threshold']:.2f}")
    print(f"  Dice  @ best thr:        {results['best_dice_at_best_thr']:.4f}")
    print(f"  IoU   @ best thr:        {results['best_iou_at_best_thr']:.4f}")
    print(f"  TTA:                     {args.tta}")
    print("=" * 60)

    # Save results JSON next to checkpoint
    ckpt_path = Path(args.checkpoint)
    out_path = ckpt_path.parent / f"eval_{args.split}_results.json"
    with open(out_path, "w") as f:
        json.dump({"split": args.split, "tta": args.tta, **results}, f, indent=2)
    log.info(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
