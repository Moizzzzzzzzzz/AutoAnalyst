from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def simple_regression_benchmark(
    df: pd.DataFrame, target_column: Optional[str]
) -> Dict[str, Any]:
    """Run a small linear regression benchmark (if data is suitable)."""
    if not target_column or target_column not in df.columns:
        return {}

    # Only attempt if target is numeric and enough rows exist.
    try:
        y = pd.to_numeric(df[target_column], errors="coerce")
    except Exception:
        return {}

    mask = y.notna()
    if mask.sum() < 10:
        return {"note": "Not enough numeric rows for regression benchmark."}

    X = df.loc[mask].drop(columns=[target_column])
    X = X.select_dtypes(include="number")
    if X.shape[1] == 0:
        return {"note": "No numeric features available for regression benchmark."}

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y[mask].values, test_size=0.2, random_state=42
    )
    if len(y_train) < 2 or len(y_test) < 1:
        return {"note": "Not enough data after split for evaluation."}

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "model": "LinearRegression",
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(X.shape[1]),
    }

