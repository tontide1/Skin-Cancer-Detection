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
