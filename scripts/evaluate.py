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
from src.losses.segmentation import CombinedLoss
from src.metrics.segmentation import dice_coefficient, find_best_threshold, iou_score
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
# TTA helpers
# ---------------------------------------------------------------------------

def _tta_predict(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Test-Time Augmentation: average predictions over original + hflip + vflip.
    Inputs/outputs are raw logits.
    """
    preds = torch.sigmoid(model(images))
    preds += torch.sigmoid(model(torch.flip(images, dims=[3])).flip(dims=[3]))  # h-flip
    preds += torch.sigmoid(model(torch.flip(images, dims=[2])).flip(dims=[2]))  # v-flip
    return preds / 3.0  # already in [0,1], skip sigmoid later


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
    total_loss = total_dice = total_iou = 0.0

    all_preds:  list[torch.Tensor] = []
    all_masks:  list[torch.Tensor] = []

    for images, masks in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        masks  = masks.to(device)

        if use_tta:
            probs  = _tta_predict(model, images)
            logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))  # probs → logits for loss
        else:
            logits = model(images)
            probs  = torch.sigmoid(logits)

        loss = criterion(logits, masks)

        n = images.size(0)
        total_loss += loss.item() * n
        total_dice += dice_coefficient(logits, masks, threshold) * n
        total_iou  += iou_score(logits, masks, threshold) * n

        all_preds.append(probs.cpu())
        all_masks.append(masks.cpu())

    n_total = len(loader.dataset)

    # Concatenate all preds for threshold search
    all_preds_t = torch.cat(all_preds, dim=0)
    all_masks_t = torch.cat(all_masks, dim=0)

    # Find best threshold on the full eval set
    best_thr, best_dice = find_best_threshold(
        torch.logit(all_preds_t.clamp(1e-6, 1 - 1e-6)),
        all_masks_t,
    )

    return {
        "loss":       total_loss / n_total,
        "dice":       total_dice / n_total,
        "iou":        total_iou  / n_total,
        "best_threshold": best_thr,
        "best_dice_at_best_thr": best_dice,
        "best_iou_at_best_thr":  iou_score(
            torch.logit(all_preds_t.clamp(1e-6, 1 - 1e-6)),
            all_masks_t,
            best_thr,
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation model on test set",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", "-k", required=True,
                        help="Path to best_model.pth checkpoint")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--tta", dest="tta", action="store_true", default=True,
                        help="Use Test-Time Augmentation (default: on)")
    parser.add_argument("--no-tta", dest="tta", action="store_false",
                        help="Disable TTA")
    parser.add_argument("overrides", nargs="*", metavar="key.subkey=value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    config = override_config(config, args.overrides)
    if not config.logging.experiment_name:
        config["logging"]["experiment_name"] = Path(args.config).stem

    set_seed(config.seed)
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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
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
