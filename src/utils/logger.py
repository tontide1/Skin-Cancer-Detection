"""
Logger wrapper — W&B + local file logging.
Graceful fallback: nếu W&B không available hoặc use_wandb=false → chỉ log local.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Logger:
    """
    Unified logger: W&B + local JSON/CSV.

    Usage:
        log = Logger(config, output_dir)
        log.log({"train_loss": 0.12, "val_dice": 0.89}, step=1)
        log.log_summary({"test_dice": 0.90})
        log.finish()
    """

    def __init__(self, config: Any, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.use_wandb = config.logging.use_wandb
        self.history: list[dict] = []
        self._wandb_run = None

        self._init_wandb(config)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_wandb(self, config: Any) -> None:
        if not self.use_wandb:
            return

        try:
            import wandb
        except ImportError:
            logger.warning("wandb không được cài. Fallback về local logging.")
            self.use_wandb = False
            return

        experiment_name = config.logging.experiment_name
        entity = config.logging.entity if config.logging.entity else None

        try:
            self._wandb_run = wandb.init(
                project=config.logging.project,
                entity=entity,
                name=experiment_name,
                config=config.to_dict(),
                dir=str(self.output_dir),
            )
            logger.info(f"W&B run: {self._wandb_run.url}")
        except Exception as e:
            logger.warning(f"W&B init thất bại: {e}. Fallback về local logging.")
            self.use_wandb = False

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics cho 1 epoch/step."""
        if step is not None:
            metrics = {"step": step, **metrics}
        self.history.append(metrics)

        if self.use_wandb and self._wandb_run:
            import wandb

            self._wandb_run.log(metrics, step=step)

    def log_summary(self, summary: dict) -> None:
        """Log summary metrics (test results, best metrics, ...)."""
        if self.use_wandb and self._wandb_run:
            for k, v in summary.items():
                self._wandb_run.summary[k] = v

    def log_image(self, key: str, image_path: str | Path) -> None:
        """Log image lên W&B."""
        if self.use_wandb and self._wandb_run:
            import wandb

            self._wandb_run.log({key: wandb.Image(str(image_path))})

    # ------------------------------------------------------------------
    # Save local
    # ------------------------------------------------------------------

    def save_history(self) -> None:
        """Lưu training history ra CSV và JSON."""
        if not self.history:
            return

        import csv

        # JSON
        json_path = self.output_dir / "training_history.json"
        with open(json_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # CSV
        csv_path = self.output_dir / "training_history.csv"
        fieldnames: list[str] = []
        for row in self.history:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

        # Metrics summary (mean/std/min/max/median)
        metrics_summary = self._build_metrics_summary(self.history)
        if metrics_summary:
            summary_json_path = self.output_dir / "metrics_summary.json"
            with open(summary_json_path, "w") as f:
                json.dump(metrics_summary, f, indent=2)

            summary_csv_path = self.output_dir / "metrics_summary.csv"
            with open(summary_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["metric", "mean", "std", "min", "max", "median"]
                )
                writer.writeheader()
                writer.writerows(metrics_summary)

    def _build_metrics_summary(self, history: list[dict]) -> list[dict[str, float | str]]:
        """
        Tính Mean/Std/Min/Max/Median cho từng metric trong history.

        Chỉ lấy các key có giá trị numeric (int/float) và bỏ qua "step".
        """
        metric_values: dict[str, list[float]] = {}
        for row in history:
            for key, value in row.items():
                if key == "step":
                    continue
                if isinstance(value, (int, float)):
                    metric_values.setdefault(key, []).append(float(value))

        summary: list[dict[str, float | str]] = []
        for metric, values in metric_values.items():
            if not values:
                continue
            mean = float(statistics.fmean(values))
            std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
            summary.append(
                {
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "median": float(statistics.median(values)),
                }
            )
        return summary

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def finish(self) -> None:
        """Kết thúc logging, save history local."""
        self.save_history()
        if self.use_wandb and self._wandb_run:
            self._wandb_run.finish()
