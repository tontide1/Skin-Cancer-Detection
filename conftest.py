from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that `from src.…` imports work
# regardless of how tests are invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))
