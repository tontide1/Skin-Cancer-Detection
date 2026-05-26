# Inference Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-image inference benchmark script that benchmarks mapped checkpoints from `outputs/` on recursive images from `data/`, then writes latency and throughput records to JSON.

**Architecture:** Add a new CLI script, `scripts/benchmark_inference.py`, without changing the existing synthetic `scripts/benchmark_fps.py`. A YAML mapping file controls display labels, configs, and checkpoints; the script loads each configured model through the existing project factory and measures one forward pass per requested batch size. Unit tests cover CLI parsing, mapping validation, recursive image collection, batching, metric math, OOM skip behavior, and JSON shape without requiring real checkpoints.

**Tech Stack:** Python 3.12, PyTorch, Pillow, PyYAML, Albumentations transforms via `get_transforms("val", config)`, existing project helpers from `src.models`, `src.utils.config`, and `src.utils.checkpoint`.

---

## File Structure

- Create: `scripts/benchmark_inference.py`
  - Responsibility: CLI entrypoint plus focused helpers for loading mapping entries, scanning image files, preprocessing batches, loading models, measuring inference latency, computing throughput, and writing JSON.
- Create: `configs/benchmark_models.yaml`
  - Responsibility: Default benchmark mapping for the requested display labels and local output checkpoints.
- Create: `tests/test_benchmark_inference.py`
  - Responsibility: Fast unit tests for the benchmark script helpers. Tests must mock model loading/timing where needed and must not require real checkpoints, real `data/`, or GPU.
- Do not modify: `scripts/benchmark_fps.py`
  - Reason: Existing synthetic tensor benchmark remains separate.

## Implementation Notes

- Use `from __future__ import annotations` at the top of new Python files.
- Follow existing script pattern for repo-root `sys.path` injection.
- Use `log.error(...)` plus `sys.exit(1)` for CLI-facing missing paths or empty image set.
- Use `load_state_dict_with_aux_compat()` when loading checkpoint state.
- Use `.get("model_state_dict", ckpt)` so raw state dict checkpoints still work.
- Do not compare model display labels against checkpoint metadata.
- Do not emit `resolved_model_name`.
- Do not emit warnings about label/checkpoint metadata mismatch.
- Do not save prediction masks or overlays.

---

### Task 1: Add CLI Parsing and Model Mapping Loader

**Files:**
- Create: `tests/test_benchmark_inference.py`
- Create: `scripts/benchmark_inference.py`

- [ ] **Step 1: Write failing tests for CLI defaults and mapping loading**

Add this initial content to `tests/test_benchmark_inference.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

import scripts.benchmark_inference as benchmark_inference


def test_parse_args_defaults() -> None:
    args = benchmark_inference.parse_args([])

    assert args.models == "configs/benchmark_models.yaml"
    assert args.data_dir == "data"
    assert args.output == "outputs/inference_benchmark.json"
    assert args.batch_sizes == [1, 50, 100]
    assert args.device == "cpu"
    assert args.max_images is None


def test_parse_args_custom_values() -> None:
    args = benchmark_inference.parse_args(
        [
            "--models",
            "tmp/models.yaml",
            "--data-dir",
            "tmp/images",
            "--output",
            "tmp/out.json",
            "--batch-sizes",
            "2",
            "4",
            "--device",
            "cuda",
            "--max-images",
            "8",
        ]
    )

    assert args.models == "tmp/models.yaml"
    assert args.data_dir == "tmp/images"
    assert args.output == "tmp/out.json"
    assert args.batch_sizes == [2, 4]
    assert args.device == "cuda"
    assert args.max_images == 8


def test_load_model_entries_preserves_label_and_paths(tmp_path: Path) -> None:
    mapping = tmp_path / "models.yaml"
    mapping.write_text(
        """
models:
  - label: SAM 2
    config: configs/experiments/mobilenetv3_deeplabv3_v1.yaml
    checkpoint: outputs/sam2_vit/best_model.pth
""".strip(),
        encoding="utf-8",
    )

    entries = benchmark_inference.load_model_entries(mapping)

    assert len(entries) == 1
    assert entries[0].label == "SAM 2"
    assert entries[0].config == Path("configs/experiments/mobilenetv3_deeplabv3_v1.yaml")
    assert entries[0].checkpoint == Path("outputs/sam2_vit/best_model.pth")


def test_load_model_entries_rejects_missing_models_key(tmp_path: Path) -> None:
    mapping = tmp_path / "models.yaml"
    mapping.write_text("not_models: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="models"):
        benchmark_inference.load_model_entries(mapping)


def test_load_model_entries_rejects_incomplete_entry(tmp_path: Path) -> None:
    mapping = tmp_path / "models.yaml"
    mapping.write_text(
        """
models:
  - label: Broken
    config: cfg.yaml
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint"):
        benchmark_inference.load_model_entries(mapping)
```

- [ ] **Step 2: Run tests and verify they fail because the script does not exist**

Run:

```bash
pytest tests/test_benchmark_inference.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'scripts.benchmark_inference'`.

- [ ] **Step 3: Add minimal script with parser and mapping loader**

Create `scripts/benchmark_inference.py` with this content:

```python
#!/usr/bin/env python3
"""Benchmark real-image model inference latency and throughput."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEntry:
    """One model benchmark entry loaded from the YAML mapping file.

    Args:
        label: Display name written to JSON exactly as provided.
        config: Path to the experiment config used to build the model.
        checkpoint: Path to the checkpoint used for model weights.
    """

    label: str
    config: Path
    checkpoint: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list for tests.

    Returns:
        Parsed argparse namespace.
    """

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
    """Load model benchmark entries from a YAML mapping file.

    Args:
        path: Mapping file path.

    Returns:
        List of model entries.

    Raises:
        ValueError: If the mapping shape or any entry is invalid.
    """

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


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    _ = args
    raise SystemExit("benchmark implementation is added in later tasks")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run Task 1 tests and verify they pass**

Run:

```bash
pytest tests/test_benchmark_inference.py -v
```

Expected: PASS for the five tests in Task 1.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add scripts/benchmark_inference.py tests/test_benchmark_inference.py
git commit -m "test: add inference benchmark mapping contract"
```

---

### Task 2: Add Recursive Image Collection

**Files:**
- Modify: `tests/test_benchmark_inference.py`
- Modify: `scripts/benchmark_inference.py`

- [ ] **Step 1: Add failing tests for recursive image collection**

Append these tests to `tests/test_benchmark_inference.py`:

```python
from PIL import Image


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(path)


def test_collect_image_paths_recurses_and_sorts_supported_extensions(tmp_path: Path) -> None:
    _write_image(tmp_path / "b" / "second.png")
    _write_image(tmp_path / "a" / "first.jpg")
    (tmp_path / "ignore.txt").write_text("not an image", encoding="utf-8")

    paths = benchmark_inference.collect_image_paths(tmp_path)

    assert [p.relative_to(tmp_path).as_posix() for p in paths] == [
        "a/first.jpg",
        "b/second.png",
    ]


def test_collect_image_paths_applies_max_images_after_sort(tmp_path: Path) -> None:
    _write_image(tmp_path / "c.jpg")
    _write_image(tmp_path / "a.jpg")
    _write_image(tmp_path / "b.jpg")

    paths = benchmark_inference.collect_image_paths(tmp_path, max_images=2)

    assert [p.name for p in paths] == ["a.jpg", "b.jpg"]
```

- [ ] **Step 2: Run new tests and verify they fail because helper is missing**

Run:

```bash
pytest tests/test_benchmark_inference.py::test_collect_image_paths_recurses_and_sorts_supported_extensions tests/test_benchmark_inference.py::test_collect_image_paths_applies_max_images_after_sort -v
```

Expected: FAIL with `AttributeError: module 'scripts.benchmark_inference' has no attribute 'collect_image_paths'`.

- [ ] **Step 3: Implement image collection helper**

Add this import near the other imports in `scripts/benchmark_inference.py`:

```python
from src.data.dataset import IMAGE_EXTS
```

Add this helper after `load_model_entries()`:

```python
def collect_image_paths(data_dir: Path, max_images: int | None = None) -> list[Path]:
    """Collect supported image files recursively.

    Args:
        data_dir: Root directory to scan.
        max_images: Optional maximum number of sorted images to return.

    Returns:
        Sorted image paths.
    """

    paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    if max_images is not None:
        return paths[:max_images]
    return paths
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
pytest tests/test_benchmark_inference.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add scripts/benchmark_inference.py tests/test_benchmark_inference.py
git commit -m "feat: collect benchmark images recursively"
```

---

### Task 3: Add Preprocessing and Batch Construction

**Files:**
- Modify: `tests/test_benchmark_inference.py`
- Modify: `scripts/benchmark_inference.py`

- [ ] **Step 1: Add failing tests for preprocessing and batch construction**

Append these tests to `tests/test_benchmark_inference.py`:

```python
import numpy as np
import torch


def test_preprocess_image_uses_supplied_transform(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8)).save(image_path)
    calls = {"count": 0}

    def fake_transform(*, image, mask):
        calls["count"] += 1
        assert image.shape == (10, 12, 3)
        assert mask.shape == (10, 12)
        return {"image": torch.ones((3, 8, 8), dtype=torch.float32)}

    tensor = benchmark_inference.preprocess_image(image_path, fake_transform)

    assert calls["count"] == 1
    assert tensor.shape == (3, 8, 8)
    assert tensor.dtype == torch.float32


def test_make_batch_repeats_images_until_batch_size(tmp_path: Path, monkeypatch) -> None:
    image_paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    tensors = {
        image_paths[0]: torch.zeros((3, 4, 4), dtype=torch.float32),
        image_paths[1]: torch.ones((3, 4, 4), dtype=torch.float32),
    }

    monkeypatch.setattr(
        benchmark_inference,
        "preprocess_image",
        lambda path, transform: tensors[path],
    )

    batch = benchmark_inference.make_batch(
        image_paths=image_paths,
        transform=object(),
        batch_size=5,
        device=torch.device("cpu"),
    )

    assert batch.shape == (5, 3, 4, 4)
    assert torch.equal(batch[0], tensors[image_paths[0]])
    assert torch.equal(batch[1], tensors[image_paths[1]])
    assert torch.equal(batch[2], tensors[image_paths[0]])
    assert torch.equal(batch[4], tensors[image_paths[0]])
```

- [ ] **Step 2: Run new tests and verify they fail because helpers are missing**

Run:

```bash
pytest tests/test_benchmark_inference.py::test_preprocess_image_uses_supplied_transform tests/test_benchmark_inference.py::test_make_batch_repeats_images_until_batch_size -v
```

Expected: FAIL with missing helper attributes.

- [ ] **Step 3: Implement preprocessing and batch helpers**

Add these imports in `scripts/benchmark_inference.py`:

```python
import numpy as np
import torch
from PIL import Image
```

Add these helpers after `collect_image_paths()`:

```python
def preprocess_image(image_path: Path, transform) -> torch.Tensor:
    """Load and preprocess one RGB image.

    Args:
        image_path: Image file path.
        transform: Validation transform from `get_transforms("val", config)`.

    Returns:
        Tensor with shape `(C, H, W)`.
    """

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
    """Create one benchmark batch, repeating images if needed.

    Args:
        image_paths: Available image paths.
        transform: Validation transform.
        batch_size: Desired batch size.
        device: Inference device.

    Returns:
        Tensor with shape `(batch_size, C, H, W)` on `device`.
    """

    tensors = [
        preprocess_image(image_paths[idx % len(image_paths)], transform)
        for idx in range(batch_size)
    ]
    return torch.stack(tensors, dim=0).to(device)
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
pytest tests/test_benchmark_inference.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add scripts/benchmark_inference.py tests/test_benchmark_inference.py
git commit -m "feat: build real-image benchmark batches"
```

---

### Task 4: Add Benchmark Timing, Throughput Math, and OOM Skip

**Files:**
- Modify: `tests/test_benchmark_inference.py`
- Modify: `scripts/benchmark_inference.py`

- [ ] **Step 1: Add failing tests for metric math and OOM handling**

Append these tests to `tests/test_benchmark_inference.py`:

```python
from types import SimpleNamespace


def test_benchmark_batch_reports_batch_latency_and_throughput(monkeypatch) -> None:
    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return x

    times = iter([10.0, 10.2, 20.0, 20.5])
    monkeypatch.setattr(benchmark_inference.time, "perf_counter", lambda: next(times))

    result = benchmark_inference.benchmark_batch(
        model=FakeModel(),
        batch=torch.zeros((5, 3, 4, 4), dtype=torch.float32),
        batch_size=5,
        device=torch.device("cpu"),
    )

    assert result is not None
    assert result.latency_ms == pytest.approx(500.0)
    assert result.throughput_fps == pytest.approx(10.0)


def test_benchmark_batch_skips_out_of_memory(monkeypatch) -> None:
    class OOMModel(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(benchmark_inference.gc, "collect", lambda: None)

    result = benchmark_inference.benchmark_batch(
        model=OOMModel(),
        batch=torch.zeros((2, 3, 4, 4), dtype=torch.float32),
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert result is None


def test_benchmark_batch_reraises_non_oom_errors() -> None:
    class BrokenModel(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        benchmark_inference.benchmark_batch(
            model=BrokenModel(),
            batch=torch.zeros((2, 3, 4, 4), dtype=torch.float32),
            batch_size=2,
            device=torch.device("cpu"),
        )
```

- [ ] **Step 2: Run new tests and verify they fail because helper/types are missing**

Run:

```bash
pytest tests/test_benchmark_inference.py::test_benchmark_batch_reports_batch_latency_and_throughput tests/test_benchmark_inference.py::test_benchmark_batch_skips_out_of_memory tests/test_benchmark_inference.py::test_benchmark_batch_reraises_non_oom_errors -v
```

Expected: FAIL with missing `benchmark_batch` or missing `time`/`gc` module attributes.

- [ ] **Step 3: Implement timing helper and result dataclass**

Add these imports in `scripts/benchmark_inference.py`:

```python
import gc
import time
```

Add this dataclass after `ModelEntry`:

```python
@dataclass(frozen=True)
class BatchBenchmarkResult:
    """Benchmark metrics for one model and batch size.

    Args:
        latency_ms: Total forward latency for the whole batch in milliseconds.
        throughput_fps: Throughput computed as batch size divided by latency seconds.
    """

    latency_ms: float
    throughput_fps: float
```

Add these helpers after `make_batch()`:

```python
def is_out_of_memory_error(exc: RuntimeError) -> bool:
    """Return whether a PyTorch runtime error is an out-of-memory failure.

    Args:
        exc: Runtime error raised during inference.

    Returns:
        True for CUDA OOM or messages containing "out of memory".
    """

    oom_error = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_error) or "out of memory" in str(exc).lower()


def benchmark_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> BatchBenchmarkResult | None:
    """Benchmark one prebuilt batch.

    Args:
        model: Model in eval mode.
        batch: Input batch tensor on the target device.
        batch_size: Number of images in the batch.
        device: Inference device.

    Returns:
        Benchmark result, or `None` when the batch OOMs.
    """

    try:
        with torch.inference_mode():
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        start = time.perf_counter()
        with torch.inference_mode():
            _ = model(batch)
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
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
pytest tests/test_benchmark_inference.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add scripts/benchmark_inference.py tests/test_benchmark_inference.py
git commit -m "feat: measure benchmark batch latency"
```

---

### Task 5: Add Model-Level Benchmarking and JSON Output Shape

**Files:**
- Modify: `tests/test_benchmark_inference.py`
- Modify: `scripts/benchmark_inference.py`

- [ ] **Step 1: Add failing tests for model-level records and JSON writing**

Append these tests to `tests/test_benchmark_inference.py`:

```python
import json


def test_format_result_record_uses_mapping_label() -> None:
    entry = benchmark_inference.ModelEntry(
        label="SAM 2",
        config=Path("cfg.yaml"),
        checkpoint=Path("outputs/sam2_vit/best_model.pth"),
    )
    result = benchmark_inference.BatchBenchmarkResult(latency_ms=250.0, throughput_fps=8.0)

    record = benchmark_inference.format_result_record(
        entry=entry,
        device=torch.device("cpu"),
        input_size=(256, 256),
        batch_size=2,
        result=result,
    )

    assert record == {
        "model": "SAM 2",
        "device": "cpu",
        "input_size": "256x256",
        "batch_size": 2,
        "latency_ms": 250.0,
        "throughput_fps": 8.0,
        "checkpoint": "outputs/sam2_vit/best_model.pth",
    }
    assert "resolved_model_name" not in record
    assert "warnings" not in record


def test_write_results_json_creates_parent_and_file(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "benchmark.json"
    payload = {
        "device": "cpu",
        "data_dir": "data",
        "num_images": 1,
        "batch_sizes": [1],
        "results": [
            {
                "model": "Resnet-Unet",
                "device": "cpu",
                "input_size": "256x256",
                "batch_size": 1,
                "latency_ms": 10.0,
                "throughput_fps": 100.0,
                "checkpoint": "outputs/resnet34_unet/best_model.pth",
            }
        ],
    }

    benchmark_inference.write_results_json(output_path, payload)

    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
```

- [ ] **Step 2: Run new tests and verify they fail because helpers are missing**

Run:

```bash
pytest tests/test_benchmark_inference.py::test_format_result_record_uses_mapping_label tests/test_benchmark_inference.py::test_write_results_json_creates_parent_and_file -v
```

Expected: FAIL with missing helper attributes.

- [ ] **Step 3: Implement formatting and JSON writing helpers**

Add this import in `scripts/benchmark_inference.py`:

```python
import json
```

Add these helpers after `benchmark_batch()`:

```python
def format_result_record(
    entry: ModelEntry,
    device: torch.device,
    input_size: tuple[int, int],
    batch_size: int,
    result: BatchBenchmarkResult,
) -> dict[str, object]:
    """Format one benchmark result record for JSON output.

    Args:
        entry: Model mapping entry.
        device: Inference device.
        input_size: Input height and width.
        batch_size: Benchmarked batch size.
        result: Measured benchmark metrics.

    Returns:
        JSON-serializable result record.
    """

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
    """Write benchmark payload to JSON.

    Args:
        output_path: Destination JSON path.
        payload: JSON-serializable benchmark payload.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
pytest tests/test_benchmark_inference.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add scripts/benchmark_inference.py tests/test_benchmark_inference.py
git commit -m "feat: format inference benchmark json"
```

---

### Task 6: Add Full CLI Flow

**Files:**
- Modify: `tests/test_benchmark_inference.py`
- Modify: `scripts/benchmark_inference.py`

- [ ] **Step 1: Add failing test for full flow orchestration with mocks**

Append this test to `tests/test_benchmark_inference.py`:

```python
def test_run_benchmark_orchestrates_entries_batches_and_payload(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    _write_image(data_dir / "sample.jpg")
    output_path = tmp_path / "benchmark.json"
    entry = benchmark_inference.ModelEntry(
        label="SAM 2",
        config=Path("cfg.yaml"),
        checkpoint=Path("ckpt.pth"),
    )

    class FakeConfig:
        seed = 42

        class data:
            input_size = [256, 256]

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return x

    calls = {"set_seed": 0, "load_state": 0}

    monkeypatch.setattr(benchmark_inference, "load_config", lambda path: FakeConfig())
    monkeypatch.setattr(
        benchmark_inference,
        "set_seed",
        lambda seed: calls.__setitem__("set_seed", calls["set_seed"] + 1),
    )
    monkeypatch.setattr(benchmark_inference, "create_model", lambda config: FakeModel())
    monkeypatch.setattr(benchmark_inference.torch, "load", lambda path, **kwargs: {"model_state_dict": {}})
    monkeypatch.setattr(
        benchmark_inference,
        "load_state_dict_with_aux_compat",
        lambda model, state, context: calls.__setitem__("load_state", calls["load_state"] + 1),
    )
    monkeypatch.setattr(benchmark_inference, "get_transforms", lambda split, config: object())
    monkeypatch.setattr(
        benchmark_inference,
        "make_batch",
        lambda image_paths, transform, batch_size, device: torch.zeros(
            (batch_size, 3, 4, 4),
            dtype=torch.float32,
        ),
    )
    monkeypatch.setattr(
        benchmark_inference,
        "benchmark_batch",
        lambda model, batch, batch_size, device: benchmark_inference.BatchBenchmarkResult(
            latency_ms=100.0,
            throughput_fps=float(batch_size) * 10.0,
        ),
    )

    payload = benchmark_inference.run_benchmark(
        entries=[entry],
        data_dir=data_dir,
        output_path=output_path,
        batch_sizes=[1, 2],
        device=torch.device("cpu"),
        max_images=None,
    )

    assert calls == {"set_seed": 1, "load_state": 1}
    assert payload["device"] == "cpu"
    assert payload["data_dir"] == data_dir.as_posix()
    assert payload["num_images"] == 1
    assert payload["batch_sizes"] == [1, 2]
    assert [record["batch_size"] for record in payload["results"]] == [1, 2]
    assert [record["model"] for record in payload["results"]] == ["SAM 2", "SAM 2"]
    assert output_path.exists()
```

- [ ] **Step 2: Run new test and verify it fails because full flow is missing**

Run:

```bash
pytest tests/test_benchmark_inference.py::test_run_benchmark_orchestrates_entries_batches_and_payload -v
```

Expected: FAIL with missing `run_benchmark` or missing imported project helpers.

- [ ] **Step 3: Add project-helper imports**

Add these imports after repo-root injection in `scripts/benchmark_inference.py`:

```python
from src.data.transforms import get_transforms
from src.models.segmentation import create_model
from src.utils.checkpoint import load_state_dict_with_aux_compat
from src.utils.config import load_config
from src.utils.misc import set_seed
```

- [ ] **Step 4: Implement model loading and full benchmark orchestration**

Add these helpers after `write_results_json()`:

```python
def load_model_for_entry(entry: ModelEntry, device: torch.device) -> tuple[object, torch.nn.Module]:
    """Load config, model, and checkpoint for one mapping entry.

    Args:
        entry: Model mapping entry.
        device: Inference device.

    Returns:
        `(config, model)` tuple.
    """

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
) -> dict[str, object]:
    """Run the complete benchmark and write JSON output.

    Args:
        entries: Model mapping entries.
        data_dir: Directory recursively scanned for images.
        output_path: Destination JSON path.
        batch_sizes: Batch sizes to benchmark.
        device: Inference device.
        max_images: Optional image limit.

    Returns:
        JSON-serializable benchmark payload.
    """

    image_paths = collect_image_paths(data_dir, max_images=max_images)
    results: list[dict[str, object]] = []

    for entry in entries:
        log.info("Benchmarking model: %s", entry.label)
        config, model = load_model_for_entry(entry, device)
        height, width = (int(v) for v in config.data.input_size)
        transform = get_transforms("val", config)

        for batch_size in batch_sizes:
            log.info("Benchmarking %s batch_size=%s", entry.label, batch_size)
            batch = make_batch(image_paths, transform, batch_size, device)
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
        "num_images": len(image_paths),
        "batch_sizes": batch_sizes,
        "results": results,
    }
    write_results_json(output_path, payload)
    return payload
```

- [ ] **Step 5: Replace temporary `main()` with real CLI validation and run call**

Replace the existing `main()` in `scripts/benchmark_inference.py` with:

```python
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
    )
    log.info("Benchmark results saved: %s", output_path)
```

- [ ] **Step 6: Run tests and verify they pass**

Run:

```bash
pytest tests/test_benchmark_inference.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 6**

Run:

```bash
git add scripts/benchmark_inference.py tests/test_benchmark_inference.py
git commit -m "feat: add inference benchmark cli flow"
```

---

### Task 7: Add Default Benchmark Mapping File

**Files:**
- Create: `configs/benchmark_models.yaml`

- [ ] **Step 1: Create mapping file**

Create `configs/benchmark_models.yaml` with:

```yaml
models:
  - label: Resnet-Unet
    config: configs/experiments/resnet34_unet_v1.yaml
    checkpoint: outputs/resnet34_unet/best_model.pth

  - label: Unet
    config: configs/experiments/unet_original_v1.yaml
    checkpoint: outputs/unet_original_v1/best_model.pth

  - label: Deeplabv3
    config: configs/experiments/mobilenetv3_deeplabv3_v1.yaml
    checkpoint: outputs/mobilenetv3_deeplabv3_v1/best_model.pth

  - label: Deeplabv3 Plus
    config: configs/experiments/resnet50_deeplabv3plus_v1.yaml
    checkpoint: outputs/deeplabv3_plus/best_model.pth

  - label: TransUNet
    config: configs/experiments/transunet_r50_vitb16_v1.yaml
    checkpoint: outputs/trans_unet_vit/best_model.pth

  - label: SAM 2
    config: configs/experiments/mobilenetv3_deeplabv3_v1.yaml
    checkpoint: outputs/sam2_vit/best_model.pth
```

- [ ] **Step 2: Validate mapping file with loader**

Run:

```bash
python -c "from pathlib import Path; from scripts.benchmark_inference import load_model_entries; entries=load_model_entries(Path('configs/benchmark_models.yaml')); print([e.label for e in entries])"
```

Expected output contains:

```text
['Resnet-Unet', 'Unet', 'Deeplabv3', 'Deeplabv3 Plus', 'TransUNet', 'SAM 2']
```

- [ ] **Step 3: Commit Task 7**

Run:

```bash
git add configs/benchmark_models.yaml
git commit -m "config: add inference benchmark models"
```

---

### Task 8: Final Verification

**Files:**
- Verify: `scripts/benchmark_inference.py`
- Verify: `configs/benchmark_models.yaml`
- Verify: `tests/test_benchmark_inference.py`

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
pytest tests/test_benchmark_inference.py -v
```

Expected: PASS.

- [ ] **Step 2: Run relevant benchmark-related tests**

Run:

```bash
pytest tests/test_benchmark_inference.py tests/test_benchmark_fps.py -v
```

Expected: PASS.

- [ ] **Step 3: Run ruff on touched Python files**

Run:

```bash
ruff check scripts/benchmark_inference.py tests/test_benchmark_inference.py
```

Expected: PASS with no lint errors.

- [ ] **Step 4: Optional smoke run if local data and checkpoints are available**

Run:

```bash
python scripts/benchmark_inference.py --max-images 1 --batch-sizes 1 --output /tmp/inference_benchmark.json
```

Expected if local `data/` contains at least one image and all mapped checkpoints/configs are loadable:

```text
Benchmark results saved: /tmp/inference_benchmark.json
```

Then inspect JSON:

```bash
python -m json.tool /tmp/inference_benchmark.json
```

Expected JSON fields:

```json
{
  "device": "cpu",
  "data_dir": "data",
  "num_images": 1,
  "batch_sizes": [1],
  "results": []
}
```

The `results` list should contain one record per successfully benchmarked model/batch pair. It may be shorter only if a batch size OOMs.

- [ ] **Step 5: Commit final verification fixes only if needed**

If verification required code or test fixes, run:

```bash
git add scripts/benchmark_inference.py tests/test_benchmark_inference.py configs/benchmark_models.yaml
git commit -m "fix: stabilize inference benchmark verification"
```

If no fixes were needed, do not create an empty commit.

---

## Self-Review

- Spec coverage: The plan covers the new script, mapping file, recursive `data/` scan, default batch sizes, CPU default, JSON-only output, label-as-source-of-truth behavior, OOM skip behavior, and final verification commands.
- Scope check: The plan does not modify `scripts/benchmark_fps.py`, does not save masks, does not add SAM 2 dependencies, and does not validate label/checkpoint metadata.
- Deferred-work scan: No incomplete implementation notes are present; every task has concrete tests, code snippets, commands, and expected outcomes.
- Type consistency: `ModelEntry`, `BatchBenchmarkResult`, `collect_image_paths`, `preprocess_image`, `make_batch`, `benchmark_batch`, `format_result_record`, `write_results_json`, `load_model_for_entry`, and `run_benchmark` are introduced before later tasks use them.
