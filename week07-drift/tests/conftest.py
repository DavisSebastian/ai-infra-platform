"""Week 7 test setup: make the scripts importable, and load the committed baseline window.

Run:  uv run --group dev --group week07-drift pytest week07-drift/tests
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))  # the scripts aren't an installed package


@pytest.fixture
def baseline() -> pd.DataFrame:
    """150 real records: Qwen2.5-0.5B answering everyday questions through the gateway (card v1)."""
    return pd.read_parquet(HERE.parent / "data" / "baseline.parquet")
