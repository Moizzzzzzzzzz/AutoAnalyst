from __future__ import annotations

import pandas as pd

from autoanalyst.agent.eda_agent import EDAAgent
from autoanalyst.agent.ml_agent import MLAgent
from autoanalyst.agent.stats_agent import StatsAgent
from autoanalyst.agent.viz_agent import VizAgent


def test_agents_return_dicts():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "y": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]})

    assert isinstance(EDAAgent().run(df), dict)
    assert isinstance(StatsAgent().run(df, target_column="y"), dict)
    assert isinstance(MLAgent().run(df, target_column="y"), dict)
    assert isinstance(VizAgent().run(df), dict)

