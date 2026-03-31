from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from ..tools.ml_tools import simple_regression_benchmark


class MLAgent:
    """AutoML specialist (lightweight baseline)."""

    def run(self, df: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        return {"benchmark": simple_regression_benchmark(df, target_column)}

