from __future__ import annotations

import pytest

from src.data.transforms import get_transforms
from src.utils.config import Config


def test_get_transforms_rejects_unknown_split() -> None:
    cfg = Config({"data": {"input_size": [256, 256]}})

    with pytest.raises(ValueError, match="Unknown split"):
        get_transforms("typo", cfg)
