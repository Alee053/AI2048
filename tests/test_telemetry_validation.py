from __future__ import annotations

import csv

import pytest


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_episode_loader_rejects_non_finite_metrics(tmp_path, value):
    from scripts.aggregate import _load_episode_rows
    from scripts.benchmark_io import EPISODE_COLUMNS

    row = {column: "0" for column in EPISODE_COLUMNS}
    row.update({"use_expectimax": "false", "mean_merge_score": str(value)})
    path = tmp_path / "episodes.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match="must be finite"):
        _load_episode_rows(path)


def test_normal_aggregation_uses_strict_episode_loader(tmp_path):
    from scripts.aggregate import load_episodes_csv
    from scripts.benchmark_io import EPISODE_COLUMNS

    path = tmp_path / "episodes.csv"
    row = {column: "0" for column in EPISODE_COLUMNS}
    row.update({"use_expectimax": "false", "total_nodes": ""})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match="missing episode metric values"):
        load_episodes_csv(tmp_path)
