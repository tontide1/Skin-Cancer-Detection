from __future__ import annotations

import csv

from src.utils.config import Config
from src.utils.logger import Logger


def test_save_history_handles_non_uniform_metric_keys(tmp_path) -> None:
    cfg = Config({"logging": {"use_wandb": False}})
    logger = Logger(cfg, tmp_path)
    logger.history = [
        {"step": 1, "train_loss": 0.9},
        {"step": 2, "train_loss": 0.8, "val_loss": 0.7},
    ]

    logger.save_history()

    csv_path = tmp_path / "training_history.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["train_loss"] == "0.9"
    assert rows[0]["val_loss"] == ""
    assert rows[1]["val_loss"] == "0.7"
