from __future__ import annotations

import os
from typing import Literal

import pandas as pd


def _infer_format(path: str) -> Literal["csv", "excel"]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        return "csv"
    if ext in [".xlsx", ".xls", ".xlsm"]:
        return "excel"
    raise ValueError(f"Unsupported file extension: {ext!r}. Use .csv or Excel.")


def load_data(path: str) -> pd.DataFrame:
    """Load a dataset into a pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    fmt = _infer_format(path)
    if fmt == "csv":
        return pd.read_csv(path)

    # Excel
    return pd.read_excel(path)

