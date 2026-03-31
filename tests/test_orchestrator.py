from __future__ import annotations

import pandas as pd

from autoanalyst.agent.orchestrator import AutoAnalystOrchestrator


def test_orchestrator_runs(tmp_path):
    df = pd.DataFrame(
        {
            "x1": [i for i in range(20)],
            "x2": [i * 2 for i in range(20)],
            "y": [i + (i * 2) * 0.1 for i in range(20)],
        }
    )
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)

    orch = AutoAnalystOrchestrator()
    result = orch.run(str(p), target_column="y")

    assert result["data_path"] == str(p)
    assert result["eda_result"]["shape"] == [20, 3]
    assert "correlation" in result["stats_result"]
    assert "benchmark" in result["ml_result"]

