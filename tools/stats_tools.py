from __future__ import annotations

import pandas as pd


def numeric_correlation(df: pd.DataFrame, target_column: str | None) -> dict:
    """Compute a simple Pearson correlation with the target (if possible)."""
    if not target_column or target_column not in df.columns:
        return {}

    corr_series = df.corr(numeric_only=True)[target_column] if df.shape[1] else None
    if corr_series is None:
        return {}

    # Ensure Python-native floats.
    return {str(col): (None if pd.isna(val) else float(val)) for col, val in corr_series.items()}

