from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..tools.viz_tools import empty_visual_bundle


class VizAgent:
    """Visualization specialist (scaffolding)."""

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        return empty_visual_bundle(df)

