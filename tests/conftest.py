import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("MTPOT_RAW", ROOT / "data" / "raw"))


def raw_file(kind, alloy_id, ext):
    """Path of a raw sample file (e.g. raw_file("elast", "01", "out")); skip if absent."""
    path = RAW / f"{kind}_{alloy_id}.{ext}"
    if not path.exists():
        pytest.skip(f"{path} not found (put the samples in data/raw or set MTPOT_RAW)")
    return path
