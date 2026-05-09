"""
Config loader với:
- YAML loading + _base_ inheritance (recursive)
- Dot-notation access: config.model.encoder_name
- CLI override: key.subkey=value
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


class Config(dict):
    """
    Dict với dot-notation access. Hỗ trợ nested access.

    Nested dicts được wrap thành Config ngay trong ``__init__`` (eager wrapping),
    do đó ``config.model is config.model`` luôn trả về cùng object và
    mutation qua ``config.model.name = "x"`` hoạt động đúng.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Wrap tất cả nested dict thành Config một lần duy nhất
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, Config):
                super().__setitem__(k, Config(v))

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __setitem__(self, key: str, val: Any) -> None:
        # Auto-wrap on assignment too
        if isinstance(val, dict) and not isinstance(val, Config):
            val = Config(val)
        super().__setitem__(key, val)

    def __repr__(self) -> str:
        import json

        return json.dumps(dict(self), indent=2, default=str)

    def to_dict(self) -> dict:
        """Chuyển về plain dict (bao gồm nested)."""
        result = {}
        for k, v in self.items():
            result[k] = v.to_dict() if isinstance(v, Config) else v
        return result


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Merge override vào base, recursive cho nested dicts.
    Override thắng base ở tất cả các level.
    """
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str | Path) -> Config:
    """
    Load YAML config với _base_ inheritance.

    Ví dụ trong experiment YAML:
        _base_: ../base.yaml
        training:
            lr: 3.0e-4   # chỉ override cái này

    Args:
        config_path: Đường dẫn đến experiment YAML

    Returns:
        Config object với dot-notation access
    """
    config_path = Path(config_path)
    raw = _load_yaml(config_path)

    # Xử lý _base_ inheritance
    if "_base_" in raw:
        base_rel = raw.pop("_base_")
        base_path = (config_path.parent / base_rel).resolve()
        base_cfg = load_config(base_path)  # recursive (hỗ trợ multi-level)
        merged = _deep_merge(base_cfg.to_dict(), raw)
    else:
        merged = raw

    return Config(merged)


def _cast_value(value_str: str) -> Any:
    """
    Tự động cast string từ CLI sang đúng kiểu Python.
    "true"/"false" → bool, "1.5e-4" → float, "42" → int, "[1,2]" → list
    """
    # Bool
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "null" or value_str.lower() == "none":
        return None

    # List: [256,256] hoặc [256, 256]
    if value_str.startswith("[") and value_str.endswith("]"):
        inner = value_str[1:-1].split(",")
        return [_cast_value(v.strip()) for v in inner if v.strip()]

    # Int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Float (bao gồm scientific notation: 2e-4, 1.5e-4)
    try:
        return float(value_str)
    except ValueError:
        pass

    # String
    return value_str


def _set_nested(d: dict, key_path: str, value: Any) -> None:
    """
    Set giá trị theo đường dẫn dot-notation.
    Ví dụ: _set_nested(d, "training.lr", 1e-3) → d["training"]["lr"] = 1e-3
    """
    keys = key_path.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _validate_key_exists(d: dict, key_path: str) -> None:
    """
    Validate that a dot-notation key path exists in the config dict.

    Raises:
        ValueError: If the key path does not exist (likely a typo).
    """
    keys = key_path.split(".")
    current = d
    for i, key in enumerate(keys):
        if not isinstance(current, dict) or key not in current:
            traversed = ".".join(keys[: i + 1])
            raise ValueError(
                f"Config key '{key_path}' không tồn tại "
                f"(failed at '{traversed}'). Kiểm tra lại tên key."
            )
        current = current[key]


def override_config(
    config: Config,
    overrides: list[str],
    strict: bool = True,
) -> Config:
    """
    Override config từ CLI args dạng "key.subkey=value".

    Ví dụ:
        overrides = [
            "data.root=/kaggle/input/isic-2018",
            "output.dir=/kaggle/working",
            "training.batch_size=32",
            "logging.use_wandb=false",
        ]

    Args:
        config:    Config object đã load
        overrides: List các string "key=value"
        strict:    If True (default), raise ValueError for unknown keys.
                   Set to False to allow adding new keys.

    Returns:
        Config đã được override
    """
    if not overrides:
        return config

    config_dict = config.to_dict()

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' không hợp lệ. Phải có dạng 'key.subkey=value'")
        key_path, _, value_str = item.partition("=")
        key_path = key_path.strip()

        if strict:
            _validate_key_exists(config_dict, key_path)

        value = _cast_value(value_str)
        _set_nested(config_dict, key_path, value)

    return Config(config_dict)
