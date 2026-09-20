"""Redirect pytest temporary directory away from the OS temp root.

On Windows 11, C:\\Temp is protected by the system, so pytest's default
tmp_path root (tempfile.gettempdir()) is not writable. We point the temp
root to a project-local folder instead.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

TMP_ROOT = Path(__file__).resolve().parent / ".pytest_tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

tempfile.tempdir = str(TMP_ROOT)