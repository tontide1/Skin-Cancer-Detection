from __future__ import annotations

import pytest
import yaml

from src.utils.config import (
    Config,
    _cast_value,
    _deep_merge,
    _set_nested,
    _validate_key_exists,
    load_config,
    override_config,
)


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------


class TestConfig:
    def test_dot_notation_access(self) -> None:
        cfg = Config({"model": {"name": "unet", "lr": 1e-3}})
        assert cfg.model.name == "unet"
        assert cfg.model.lr == 1e-3

    def test_nested_returns_config(self) -> None:
        cfg = Config({"a": {"b": {"c": 1}}})
        assert isinstance(cfg["a"], dict)
        assert cfg.a.b.c == 1

    def test_missing_key_raises(self) -> None:
        cfg = Config({"a": 1})
        with pytest.raises(AttributeError, match="no attribute"):
            _ = cfg.nonexistent

    def test_setattr(self) -> None:
        cfg = Config({})
        cfg.new_key = 42
        assert cfg["new_key"] == 42

    def test_to_dict_nested(self) -> None:
        cfg = Config({"a": {"b": 1}, "c": [2, 3]})
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert not isinstance(d["a"], Config)
        assert d == {"a": {"b": 1}, "c": [2, 3]}

    def test_repr_json(self) -> None:
        cfg = Config({"x": 1})
        r = repr(cfg)
        assert '"x"' in r


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_flat_override(self) -> None:
        base = {"a": 1, "b": 2}
        over = {"b": 99}
        assert _deep_merge(base, over) == {"a": 1, "b": 99}

    def test_nested_merge(self) -> None:
        base = {"model": {"name": "unet", "lr": 1e-3}}
        over = {"model": {"lr": 2e-4}}
        result = _deep_merge(base, over)
        assert result["model"]["name"] == "unet"
        assert result["model"]["lr"] == 2e-4

    def test_new_key_added(self) -> None:
        base = {"a": 1}
        over = {"b": 2}
        assert _deep_merge(base, over) == {"a": 1, "b": 2}

    def test_base_unchanged(self) -> None:
        base = {"a": {"x": 1}}
        over = {"a": {"x": 99}}
        _deep_merge(base, over)
        assert base["a"]["x"] == 1


# ---------------------------------------------------------------------------
# _cast_value
# ---------------------------------------------------------------------------


class TestCastValue:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("true", True),
            ("True", True),
            ("false", False),
            ("FALSE", False),
            ("null", None),
            ("None", None),
            ("42", 42),
            ("3.14", 3.14),
            ("2e-4", 2e-4),
            ("1.5e-4", 1.5e-4),
            ("hello", "hello"),
            ("[256,256]", [256, 256]),
            ("[1, 2, 3]", [1, 2, 3]),
        ],
    )
    def test_cast(self, raw: str, expected) -> None:
        assert _cast_value(raw) == expected


# ---------------------------------------------------------------------------
# _set_nested
# ---------------------------------------------------------------------------


class TestSetNested:
    def test_single_level(self) -> None:
        d: dict = {"a": 1}
        _set_nested(d, "a", 99)
        assert d["a"] == 99

    def test_multi_level(self) -> None:
        d: dict = {"training": {"lr": 1e-3}}
        _set_nested(d, "training.lr", 2e-4)
        assert d["training"]["lr"] == 2e-4

    def test_creates_intermediate(self) -> None:
        d: dict = {}
        _set_nested(d, "a.b.c", 5)
        assert d["a"]["b"]["c"] == 5


# ---------------------------------------------------------------------------
# override_config
# ---------------------------------------------------------------------------


class TestValidateKeyExists:
    def test_valid_key(self) -> None:
        d = {"training": {"lr": 1e-3}}
        _validate_key_exists(d, "training.lr")

    def test_invalid_key_raises(self) -> None:
        d = {"training": {"lr": 1e-3}}
        with pytest.raises(ValueError, match="không tồn tại"):
            _validate_key_exists(d, "trainig.lr")

    def test_deep_invalid_key_raises(self) -> None:
        d = {"training": {"lr": 1e-3}}
        with pytest.raises(ValueError, match="không tồn tại"):
            _validate_key_exists(d, "training.learning_rate")


class TestOverrideConfig:
    def test_override_values(self) -> None:
        cfg = Config({"training": {"lr": 1e-3, "batch_size": 32}})
        result = override_config(cfg, ["training.lr=2e-4", "training.batch_size=64"])
        assert result.training.lr == 2e-4
        assert result.training.batch_size == 64

    def test_empty_overrides(self) -> None:
        cfg = Config({"a": 1})
        assert override_config(cfg, []) is cfg

    def test_invalid_format_raises(self) -> None:
        cfg = Config({"a": 1})
        with pytest.raises(ValueError, match="không hợp lệ"):
            override_config(cfg, ["bad_format"])

    def test_strict_mode_rejects_typo(self) -> None:
        cfg = Config({"training": {"lr": 1e-3}})
        with pytest.raises(ValueError, match="không tồn tại"):
            override_config(cfg, ["trainig.lr=2e-4"], strict=True)

    def test_non_strict_allows_new_keys(self) -> None:
        cfg = Config({"training": {"lr": 1e-3}})
        result = override_config(cfg, ["new_section.key=42"], strict=False)
        assert result["new_section"]["key"] == 42


# ---------------------------------------------------------------------------
# load_config  (with _base_ inheritance)
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_simple_load(self, tmp_path) -> None:
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"seed": 42, "model": {"name": "unet"}}))

        cfg = load_config(p)
        assert cfg.seed == 42
        assert cfg.model.name == "unet"

    def test_base_inheritance(self, tmp_path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"seed": 42, "model": {"name": "unet", "lr": 1e-3}}))

        child = tmp_path / "child.yaml"
        child.write_text(yaml.dump({"_base_": "base.yaml", "model": {"lr": 2e-4}}))

        cfg = load_config(child)
        assert cfg.seed == 42
        assert cfg.model.name == "unet"
        assert cfg.model.lr == 2e-4

    def test_empty_yaml(self, tmp_path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        cfg = load_config(p)
        assert cfg.to_dict() == {}
