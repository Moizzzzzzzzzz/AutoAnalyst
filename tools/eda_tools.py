from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd


def dataframe_basic_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Return basic EDA summary in JSON-serializable structures."""
    # Use `to_json` + `json.loads` to avoid numpy scalar serialization issues.
    describe = json.loads(df.describe(include="all").round(6).to_json())
    head = json.loads(df.head(5).to_json(orient="records"))
    dtypes = df.dtypes.astype(str).to_dict()

    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "dtypes": dtypes,
        "describe": describe,
        "head": head,
    }

