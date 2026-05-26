# Inference Benchmark Design

## Goal

Add a real-image inference benchmark script for the trained segmentation models under `outputs/`.
The script measures batch latency and throughput for batch sizes `1`, `50`, and `100`, then writes
the results to JSON for reporting.

## Scope

In scope:

- Add a new script: `scripts/benchmark_inference.py`.
- Add a mapping file: `configs/benchmark_models.yaml`.
- Recursively scan `data/` for supported image files.
- Benchmark every model entry declared in the mapping file.
- Use batch sizes `1 50 100` by default.
- Use CPU by default.
- Save benchmark results to JSON only.

Out of scope:

- Do not modify `scripts/benchmark_fps.py`; it remains the synthetic tensor benchmark.
- Do not save predicted masks or overlay images.
- Do not add new model architectures or SAM 2 dependencies as part of this script.
- Do not validate whether a display label matches checkpoint metadata.

## CLI Contract

Default command:

```bash
python scripts/benchmark_inference.py \
  --models configs/benchmark_models.yaml \
  --data-dir data \
  --output outputs/inference_benchmark.json
```

Optional arguments:

- `--batch-sizes 1 50 100`: override batch sizes.
- `--device cpu`: override device.
- `--max-images N`: limit the number of scanned images for quick smoke runs.

## Model Mapping

The mapping file is the source of truth for model display names and checkpoint/config paths.
The `label` value is written to JSON exactly as provided.

Example:

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

Even if a checkpoint stores different metadata, the script does not emit `resolved_model_name`
and does not warn about label/metadata mismatch. The config path controls how the model is built.

## Data Flow

1. Parse CLI arguments.
2. Load model entries from `configs/benchmark_models.yaml`.
3. Recursively collect image files under `--data-dir` using `IMAGE_EXTS` from `src.data.dataset`.
4. Optionally truncate to `--max-images`.
5. For each model entry:
   - load config with `load_config()`;
   - set seed with `set_seed(config.seed)`;
   - build model with `create_model(config)`;
   - load checkpoint using `.get("model_state_dict", ckpt)`;
   - apply `load_state_dict_with_aux_compat()`;
   - move model to `--device` and call `eval()`;
   - preprocess images with `get_transforms("val", config)`.
6. For each batch size:
   - create batches from the preprocessed image stream;
   - run a small warmup forward pass;
   - measure forward-pass time under `torch.inference_mode()`;
   - compute latency and throughput.
7. Write JSON to `--output`.

## Metric Definitions

- `latency_ms`: total elapsed forward-pass time for one batch.
- `throughput_fps`: `batch_size / (latency_ms / 1000)`.
- `input_size`: formatted as `<height>x<width>` from `config.data.input_size`.

The benchmark does not include mask post-processing, saving predictions, metric computation, or TTA.

## JSON Output

The output JSON contains run metadata and flat result records:

```json
{
  "device": "cpu",
  "data_dir": "data",
  "num_images": 2594,
  "batch_sizes": [1, 50, 100],
  "results": [
    {
      "model": "Resnet-Unet",
      "device": "cpu",
      "input_size": "256x256",
      "batch_size": 1,
      "latency_ms": 170.19,
      "throughput_fps": 5.88,
      "checkpoint": "outputs/resnet34_unet/best_model.pth"
    }
  ]
}
```

## Error Handling

- Missing model mapping file: log error and exit with status `1`.
- Missing data directory: log error and exit with status `1`.
- No supported images found: log error and exit with status `1`.
- Missing config/checkpoint path for an entry: log error and exit with status `1`.
- Out-of-memory for a batch size: skip that model/batch-size record and continue with the next batch size.
- Other model load or inference errors: raise normally so the failed model is not silently reported.

## Verification

Minimal verification after implementation:

```bash
pytest tests/test_benchmark_inference.py -v
python scripts/benchmark_inference.py --max-images 1 --batch-sizes 1 --output /tmp/inference_benchmark.json
```

The second command depends on valid local checkpoints and at least one image under `data/`.
