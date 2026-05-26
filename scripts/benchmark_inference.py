#!/usr/bin/env python3
"""Benchmark real-image model inference latency and throughput."""

from __future__ import annotations

import argparse
import gc
import logging
import time
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import IMAGE_EXTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEntry:
    """One model benchmark entry loaded from the YAML mapping file."""

    label: str
    config: Path
    checkpoint: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Benchmark real-image segmentation inference latency and throughput",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--models",
        default="configs/benchmark_models.yaml",
        help="YAML model mapping file (default: configs/benchmark_models.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory recursively scanned for images (default: data)",
    )
    parser.add_argument(
        "--output",
        default="outputs/inference_benchmark.json",
        help="JSON output path (default: outputs/inference_benchmark.json)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 50, 100],
        help="Batch sizes to benchmark (default: 1 50 100)",
    )
    parser.add_argument("--device", default="cpu", help="Device to run benchmark on (default: cpu)")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional image limit for quick smoke runs",
    )
    return parser.parse_args(argv)


def load_model_entries(path: Path) -> list[ModelEntry]:
    """Load model benchmark entries from a YAML mapping file."""

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "models" not in data:
        raise ValueError("Model mapping must contain a top-level 'models' list.")
    if not isinstance(data["models"], list) or not data["models"]:
        raise ValueError("Model mapping 'models' must be a non-empty list.")

    entries: list[ModelEntry] = []
    for idx, item in enumerate(data["models"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Model entry #{idx} must be a mapping.")
        missing = [key for key in ("label", "config", "checkpoint") if key not in item]
        if missing:
            raise ValueError(f"Model entry #{idx} missing required key(s): {', '.join(missing)}")
        entries.append(
            ModelEntry(
                label=str(item["label"]),
                config=Path(str(item["config"])),
                checkpoint=Path(str(item["checkpoint"])),
            )
        )

    return entries


def collect_image_paths(data_dir: Path, max_images: int | None = None) -> list[Path]:
    """Collect supported image files recursively."""

    paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    if max_images is not None:
        return paths[:max_images]
    return paths


def preprocess_image(image_path: Path, transform) -> torch.Tensor:
    """Load and preprocess one RGB image."""

    image = np.array(Image.open(image_path).convert("RGB"))
    dummy_mask = np.zeros(image.shape[:2], dtype=np.float32)
    out = transform(image=image, mask=dummy_mask)
    return out["image"]


def make_batch(
    image_paths: list[Path],
    transform,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Create one benchmark batch, repeating images if needed."""

    tensors = [
        preprocess_image(image_paths[idx % len(image_paths)], transform)
        for idx in range(batch_size)
    ]
    return torch.stack(tensors, dim=0).to(device)


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    _ = args
    raise SystemExit("benchmark implementation is added in later tasks")


if __name__ == "__main__":
    main()
