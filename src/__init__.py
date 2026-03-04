"""
Skin Cancer Detection — src package.

Public API shortcuts:
    from src import create_model, ISICDataset
"""

from src.models.segmentation import create_model
from src.data.dataset import ISICDataset

__all__ = ["create_model", "ISICDataset"]
