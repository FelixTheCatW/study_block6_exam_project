"""Redirect pytest temporary directory away from the OS temp root.

On Windows 11, C:\\Temp is protected by the system, so pytest's default
tmp_path root (tempfile.gettempdir()) is not writable. We point the temp
root to a project-local folder instead.

Also exposes a session fixture with a small sample of the real
Health and fitness dataset for tests that need data.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.config import RAW_DATA_PATH

TMP_ROOT = Path(__file__).resolve().parent / ".pytest_tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

tempfile.tempdir = str(TMP_ROOT)


@pytest.fixture(scope="session")
def real_health_sample() -> pd.DataFrame:
    """Образец реального датасета Health and fitness (500 строк)."""
    if not RAW_DATA_PATH.exists():
        pytest.skip(f"Реальный датасет не найден: {RAW_DATA_PATH}")

    data = pd.read_csv(RAW_DATA_PATH)
    return data.sample(n=500, random_state=11).reset_index(drop=True)