#!/usr/bin/env python3
"""
Benchmark FPS for model inference.
"""

from __future__ import annotations

import argparse
import gc
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


def _is_out_of_memory_error(exc: RuntimeError) -> bool:
    """Return whether a runtime error indicates an out-of-memory failure.

    Args:
        exc: The runtime error raised by PyTorch.

    Returns:
        True when the exception message or type indicates OOM, otherwise False.
    """

    oom_error = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_error) or "out of memory" in str(exc).lower()


def _benchmark_batch_size(
    model: torch.nn.Module,
    device: torch.device,
    height: int,
    width: int,
    batch_size: int,
) -> tuple[float, float] | None:
    """Benchmark a single batch size and return latency/FPS if it fits.

    Args:
        model: The loaded segmentation model.
        device: Device used for inference.
        height: Input image height.
        width: Input image width.
        batch_size: Number of images per forward pass.

    Returns:
        A ``(latency_ms, fps)`` tuple on success, or ``None`` if the batch OOMs.
    """

    x: torch.Tensor | None = None
    timer: benchmark.Timer | None = None

    try:
        x = torch.randn(batch_size, 3, height, width, device=device)
        timer = benchmark.Timer(
            stmt="with torch.inference_mode(): model(x)",
            globals={"model": model, "x": x, "torch": torch},
        )

        stats = timer.blocked_autorange(min_run_time=2.0)
        mean_latency = stats.mean
        return mean_latency * 1000, batch_size / mean_latency
    except RuntimeError as exc:
        if not _is_out_of_memory_error(exc):
            raise

        log.warning("Skipping batch size %s due to OOM: %s", batch_size, exc)
        return None
    finally:
        del timer, x
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark inference FPS",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True, help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", "-k", required=True, help="Path to best_model.pth")
    parser.add_argument("--device", "-d", default="cpu", help="Device to run benchmark on (default: cpu)")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 50, 100],
        help="Batch sizes to benchmark (default: 1 50 100)",
    )
    parser.add_argument("overrides", nargs="*", metavar="key.subkey=value")
    return parser.parse_args(argv)


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

    results = []
    for bs in args.batch_sizes:
        log.info(f"Benchmarking batch size {bs}...")
        result = _benchmark_batch_size(model, device, H, W, bs)
        if result is None:
            continue
        results.append((bs, *result))

    print("\n--- Benchmark Results ---")
    print(f"Device      : {args.device}")
    print(f"Input size  : {H}x{W}")
    print("-" * 50)
    print(f"{'Batch Size':<12} | {'Latency (ms)':>12} | {'Throughput (fps)':>16}")
    print("-" * 50)

    for bs, latency, fps in results:
        print(f"{bs:<12} | {latency:>12.2f} | {fps:>16.2f}")

    print("-" * 50)


if __name__ == "__main__":
    main()
