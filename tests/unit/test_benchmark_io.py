"""Tests for scripts.benchmark_io — schema, dataclasses, writers."""
from __future__ import annotations

import pytest


def test_schema_version_is_semver():
    from scripts.benchmark_io import EPISODE_SCHEMA_VERSION

    parts = EPISODE_SCHEMA_VERSION.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_episode_columns_is_nonempty_list_of_str():
    from scripts.benchmark_io import EPISODE_COLUMNS

    assert isinstance(EPISODE_COLUMNS, list)
    assert len(EPISODE_COLUMNS) > 10
    assert all(isinstance(c, str) for c in EPISODE_COLUMNS)
    assert "schema_version" in EPISODE_COLUMNS
    assert "score" in EPISODE_COLUMNS
    assert "max_tile" in EPISODE_COLUMNS


def test_move_columns_is_nonempty_list_of_str():
    from scripts.benchmark_io import MOVE_COLUMNS

    assert isinstance(MOVE_COLUMNS, list)
    assert len(MOVE_COLUMNS) > 10
    assert all(isinstance(c, str) for c in MOVE_COLUMNS)
    assert "score_up" in MOVE_COLUMNS
    assert "score_left" in MOVE_COLUMNS


def test_episode_columns_no_duplicates():
    from scripts.benchmark_io import EPISODE_COLUMNS

    assert len(EPISODE_COLUMNS) == len(set(EPISODE_COLUMNS))


def test_move_columns_no_duplicates():
    from scripts.benchmark_io import MOVE_COLUMNS

    assert len(MOVE_COLUMNS) == len(set(MOVE_COLUMNS))