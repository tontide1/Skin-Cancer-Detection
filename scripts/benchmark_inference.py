#!/usr/bin/env python3
"""Benchmark real-image model inference latency and throughput."""

from __future__ import annotations

import argparse
import gc
import json
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

from src.data.dataset import IMAGE_EXTS  # noqa: E402
from src.data.transforms import get_transforms  # noqa: E402
from src.models.segmentation import create_model  # noqa: E402
from src.utils.checkpoint import load_state_dict_with_aux_compat  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.misc import set_seed  # noqa: E402

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


@dataclass(frozen=True)
class BatchBenchmarkResult:
    """Benchmark metrics for one batch."""

    latency_ms: float
    throughput_fps: float


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


def is_out_of_memory_error(exc: RuntimeError) -> bool:
    """Return whether a runtime error is an out-of-memory failure."""

    oom_error = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_error) or "out of memory" in str(exc).lower()


def benchmark_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> BatchBenchmarkResult | None:
    """Benchmark one prebuilt batch."""

    try:
        with torch.inference_mode():
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        start = time.perf_counter()
        with torch.inference_mode():
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        latency_ms = elapsed * 1000.0
        throughput_fps = batch_size / elapsed
        return BatchBenchmarkResult(latency_ms=latency_ms, throughput_fps=throughput_fps)
    except RuntimeError as exc:
        if not is_out_of_memory_error(exc):
            raise
        log.warning("Skipping batch size %s due to OOM: %s", batch_size, exc)
        return None
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def format_result_record(
    entry: ModelEntry,
    device: torch.device,
    input_size: tuple[int, int],
    batch_size: int,
    result: BatchBenchmarkResult,
) -> dict[str, object]:
    """Format one benchmark result record for JSON output."""

    height, width = input_size
    return {
        "model": entry.label,
        "device": str(device),
        "input_size": f"{height}x{width}",
        "batch_size": batch_size,
        "latency_ms": round(result.latency_ms, 2),
        "throughput_fps": round(result.throughput_fps, 2),
        "checkpoint": entry.checkpoint.as_posix(),
    }


def write_results_json(output_path: Path, payload: dict[str, object]) -> None:
    """Write benchmark payload to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_model_for_entry(entry: ModelEntry, device: torch.device) -> tuple[object, torch.nn.Module]:
    """Load config, model, and checkpoint for one mapping entry."""

    config = load_config(entry.config)
    set_seed(config.seed)

    model = create_model(config).to(device)
    ckpt = torch.load(entry.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    load_state_dict_with_aux_compat(model, state, context=str(entry.checkpoint))
    model.eval()
    return config, model


def run_benchmark(
    entries: list[ModelEntry],
    data_dir: Path,
    output_path: Path,
    batch_sizes: list[int],
    device: torch.device,
    max_images: int | None,
    image_paths: list[Path] | None = None,
) -> dict[str, object]:
    """Run the complete benchmark and write JSON output."""

    images = image_paths if image_paths is not None else collect_image_paths(data_dir, max_images=max_images)
    results: list[dict[str, object]] = []

    for entry in entries:
        log.info("Benchmarking model: %s", entry.label)
        config, model = load_model_for_entry(entry, device)
        height, width = (int(v) for v in config.data.input_size)
        transform = get_transforms("val", config)

        for batch_size in batch_sizes:
            log.info("Benchmarking %s batch_size=%s", entry.label, batch_size)
            batch = make_batch(images, transform, batch_size, device)
            result = benchmark_batch(model, batch, batch_size, device)
            if result is None:
                continue
            results.append(
                format_result_record(
                    entry=entry,
                    device=device,
                    input_size=(height, width),
                    batch_size=batch_size,
                    result=result,
                )
            )

    payload: dict[str, object] = {
        "device": str(device),
        "data_dir": data_dir.as_posix(),
        "num_images": len(images),
        "batch_sizes": batch_sizes,
        "results": results,
    }
    write_results_json(output_path, payload)
    return payload


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    models_path = Path(args.models)
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    if not models_path.exists():
        log.error("Model mapping file not found: %s", models_path)
        sys.exit(1)
    if not data_dir.exists():
        log.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    try:
        entries = load_model_entries(models_path)
    except ValueError as exc:
        log.error("Invalid model mapping file %s: %s", models_path, exc)
        sys.exit(1)

    for entry in entries:
        if not entry.config.exists():
            log.error("Config not found for %s: %s", entry.label, entry.config)
            sys.exit(1)
        if not entry.checkpoint.exists():
            log.error("Checkpoint not found for %s: %s", entry.label, entry.checkpoint)
            sys.exit(1)

    image_paths = collect_image_paths(data_dir, max_images=args.max_images)
    if not image_paths:
        log.error("No supported images found under: %s", data_dir)
        sys.exit(1)

    run_benchmark(
        entries=entries,
        data_dir=data_dir,
        output_path=output_path,
        batch_sizes=args.batch_sizes,
        device=torch.device(args.device),
        max_images=args.max_images,
        image_paths=image_paths,
    )
    log.info("Benchmark results saved: %s", output_path)


if __name__ == "__main__":
    main()
