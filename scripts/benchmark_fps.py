#!/usr/bin/env python3
"""
Benchmark FPS for model inference.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.utils.benchmark as benchmark

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.segmentation import create_model
from src.utils.checkpoint import load_state_dict_with_aux_compat
from src.utils.config import load_config, override_config
from src.utils.misc import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark inference FPS",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True, help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", "-k", required=True, help="Path to best_model.pth")
    parser.add_argument("--device", "-d", default="cpu", help="Device to run benchmark on (default: cpu)")
    parser.add_argument("overrides", nargs="*", metavar="key.subkey=value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    config = override_config(config, args.overrides)

    set_seed(config.seed)

    device = torch.device(args.device)
    log.info(f"Using device: {device}")

    # Model setup
    model = create_model(config)
    
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    load_state_dict_with_aux_compat(model, state, context=str(args.checkpoint))
    
    model = model.to(device)
    model.eval()
    
    log.info(f"Loaded model onto {device} and set to eval mode.")

    H, W = config.data.input_size
    log.info(f"Creating dummy input tensor of shape (1, 3, {H}, {W}) on {device}")
    x = torch.randn(1, 3, H, W, device=device)

    log.info("Running benchmark...")
    timer = benchmark.Timer(
        stmt="with torch.inference_mode(): model(x)",
        globals={"model": model, "x": x, "torch": torch}
    )
    
    stats = timer.blocked_autorange(min_run_time=2.0)
    
    mean_latency = stats.mean
    fps = 1.0 / mean_latency
    
    print("\n--- Benchmark Results ---")
    print(f"Device      : {args.device}")
    print(f"Input size  : {H}x{W}")
    print(f"Mean latency: {mean_latency * 1000:.2f} ms")
    print(f"FPS         : {fps:.2f}")
    print("-------------------------")

if __name__ == "__main__":
    main()
