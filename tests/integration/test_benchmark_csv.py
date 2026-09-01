"""End-to-end integration tests for benchmark.py CLI.

Real-model benchmark cases are marked slow/integration and skipped if the
production model is missing. The schema-rejection case stays in the default
fast suite because it does not launch a model or benchmark.
"""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_MODEL = _REPO_ROOT / "data" / "archive" / "v3-100m" / "models" / "release" / "Hybrid-PPO-Expectimax-v3.zip"


@pytest.fixture
def benchmark_output_dir(tmp_path):
    out = tmp_path / "bench_out"
    out.mkdir()
    return out


def _run_benchmark(
    args: list, timeout: int = 600, env: dict | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.benchmark", *args],
        cwd=str(_REPO_ROOT),
        capture_output=True, text=True, timeout=timeout, env=env,
    )


# --- Task 18 ---

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _PRODUCTION_MODEL.exists(),
    reason="Production model not found",
)
def test_single_worker_benchmark_writes_all_files(benchmark_output_dir):
    result = _run_benchmark([
        str(_PRODUCTION_MODEL),
        "--n-runs", "3",
        "--depth", "0",
        "--workers", "1",
        "--output", str(benchmark_output_dir),
        "--base-eval-seed", "42",
    ])
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    assert (benchmark_output_dir / "config.json").exists()
    assert (benchmark_output_dir / "episodes.csv").exists()
    assert (benchmark_output_dir / "summary.json").exists()

    with open(benchmark_output_dir / "config.json") as f:
        config = json.load(f)
    assert config["benchmark_schema_version"] == "2.1.0"
    assert config["n_workers"] == 1
    assert config["worker_timeout"] == 300.0
    assert config["worker_inactivity_timeout"] == 300.0
    assert config["worker_timeout_scope"] == (
        "per-worker-inactivity-between-episode-heartbeats"
    )
    assert config["use_expectimax"] is False
    assert config["base_eval_seed"] == 42
    assert config["status"] == "completed"

    with open(benchmark_output_dir / "episodes.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    for r in rows:
        assert int(r["eval_seed"]) in (42, 43, 44)
        assert int(r["worker_id"]) == 0
        assert r["schema_version"] == "2.1.0"
        assert "total_alpha_beta_cuts" not in r
        assert r["termination_reason"] == "board_full"
        assert int(r["score"]) >= 0


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _PRODUCTION_MODEL.exists(),
    reason="Production model not found",
)
def test_two_worker_benchmark_all_episodes_present(benchmark_output_dir):
    result = _run_benchmark([
        str(_PRODUCTION_MODEL),
        "--n-runs", "6",
        "--depth", "0",
        "--workers", "2",
        "--device", "cpu",
        "--output", str(benchmark_output_dir),
        "--base-eval-seed", "100",
    ])
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    with open(benchmark_output_dir / "episodes.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6

    by_idx = {int(r["episode_idx"]): r for r in rows}
    assert set(by_idx.keys()) == set(range(6))
    for i in range(6):
        assert int(by_idx[i]["eval_seed"]) == 100 + i
    worker_ids = {int(r["worker_id"]) for r in rows}
    assert worker_ids.issubset({0, 1})
    assert len(worker_ids) == 2, f"Expected both workers, got {worker_ids}"


# --- Task 19 ---

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _PRODUCTION_MODEL.exists(),
    reason="Production model not found",
)
def test_reproducibility_same_seed_same_score(tmp_path):
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    out1.mkdir()
    out2.mkdir()

    args = [
        str(_PRODUCTION_MODEL), "--n-runs", "5", "--depth", "0",
        "--workers", "1", "--device", "cpu",
        "--base-eval-seed", "999",
    ]
    r1 = _run_benchmark([*args, "--output", str(out1)])
    r2 = _run_benchmark([*args, "--output", str(out2)])
    assert r1.returncode == 0, r1.stderr
    assert r2.returncode == 0, r2.stderr

    with open(out1 / "episodes.csv") as f:
        rows1 = sorted(csv.DictReader(f), key=lambda r: int(r["episode_idx"]))
    with open(out2 / "episodes.csv") as f:
        rows2 = sorted(csv.DictReader(f), key=lambda r: int(r["episode_idx"]))
    assert len(rows1) == len(rows2) == 5
    for a, b in zip(rows1, rows2):
        assert a["eval_seed"] == b["eval_seed"]
        assert a["episode_idx"] == b["episode_idx"]
        assert a["score"] == b["score"]
        assert a["max_tile"] == b["max_tile"]
        assert a["steps"] == b["steps"]
        assert a["termination_reason"] == b["termination_reason"]


# --- Task 20 ---

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _PRODUCTION_MODEL.exists(),
    reason="Production model not found",
)
def test_worker_count_invariance_same_eval_seeds_per_episode(tmp_path):
    """--workers 1 vs --workers 2 must produce identical episode outcomes.

    With per-episode environment reset seeding, exact match should hold
    across worker counts.
    """
    out1 = tmp_path / "w1"
    out4 = tmp_path / "w4"
    out1.mkdir()
    out4.mkdir()

    base = [
        str(_PRODUCTION_MODEL), "--n-runs", "4", "--depth", "0",
        "--device", "cpu", "--base-eval-seed", "7",
    ]
    _run_benchmark([*base, "--workers", "1", "--output", str(out1)])
    _run_benchmark([*base, "--workers", "2", "--output", str(out4)])

    with open(out1 / "episodes.csv") as f:
        rows1 = sorted(csv.DictReader(f), key=lambda r: int(r["episode_idx"]))
    with open(out4 / "episodes.csv") as f:
        rows4 = sorted(csv.DictReader(f), key=lambda r: int(r["episode_idx"]))
    assert len(rows1) == len(rows4) == 4
    for a, b in zip(rows1, rows4):
        assert a["eval_seed"] == b["eval_seed"], (
            f"eval_seed mismatch at episode_idx={a['episode_idx']}: "
            f"{a['eval_seed']} vs {b['eval_seed']}"
        )
        # With per-episode reset seeding, score/max_tile/steps are
        # deterministic across worker counts (same eval_seed => same
        # private tile RNG stream => same game).
        assert a["score"] == b["score"], (
            f"score mismatch at episode_idx={a['episode_idx']}: "
            f"{a['score']} vs {b['score']}"
        )
        assert a["max_tile"] == b["max_tile"]
        assert a["steps"] == b["steps"]


# --- Task 21 ---

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _PRODUCTION_MODEL.exists(),
    reason="Production model not found",
)
def test_interrupt_writes_partial_output(tmp_path):
    out = tmp_path / "intr"
    out.mkdir()

    proc = subprocess.Popen(
        [sys.executable, "-m", "scripts.benchmark",
         str(_PRODUCTION_MODEL),
         "--n-runs", "10", "--depth", "0",
         "--workers", "1", "--device", "cpu",
         "--log-moves", "--worker-timeout", "60",
         "--output", str(out), "--base-eval-seed", "1"],
        cwd=str(_REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    csv_path = out / "episodes.csv"
    deadline = time.time() + 120
    rows_seen = 0
    while time.time() < deadline and rows_seen < 2:
        if csv_path.exists():
            with open(csv_path) as f:
                rows_seen = sum(1 for _ in f) - 1
        time.sleep(0.2)
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=30)

    assert proc.returncode == 2
    assert csv_path.exists()
    assert (out / "moves.csv").exists()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert 0 < len(rows) < 10, f"Expected partial rows, got {len(rows)}"

    with open(out / "config.json") as f:
        cfg = json.load(f)
    assert cfg["interrupted"] is True
    assert cfg["status"] == "interrupted"
    assert cfg["n_completed"] == len(rows)


# --- Task 23 ---

def test_aggregate_rejects_unsupported_schema_major(tmp_path):
    depth_dir = tmp_path / "sweep_depth0"
    depth_dir.mkdir()
    with open(depth_dir / "config.json", "w") as f:
        json.dump({
            "benchmark_schema_version": "1.0.0",
            "sweep_name": "sweep",
            "run_name": "run_0",
            "n_runs": 1,
        }, f)

    proc = subprocess.run(
        [sys.executable, "-m", "scripts.aggregate", str(tmp_path), "--sweep", "sweep"],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 2
    assert "invalid benchmark manifest metadata" in proc.stdout


# --- Task 24 ---

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _PRODUCTION_MODEL.exists(),
    reason="Production model not found",
)
def test_log_moves_off_vs_on_produces_identical_episodes_csv(tmp_path):
    out_off = tmp_path / "off"
    out_on = tmp_path / "on"
    out_off.mkdir()
    out_on.mkdir()

    base = [
        str(_PRODUCTION_MODEL), "--n-runs", "3", "--depth", "2",
        "--workers", "1", "--device", "cpu",
        "--base-eval-seed", "5",
    ]
    _run_benchmark([*base, "--output", str(out_off)], timeout=900)
    _run_benchmark(
        [*base, "--log-moves", "--output", str(out_on)],
        timeout=900,
    )

    with open(out_off / "episodes.csv") as f:
        rows_off = sorted(csv.DictReader(f), key=lambda r: int(r["episode_idx"]))
    with open(out_on / "episodes.csv") as f:
        rows_on = sorted(csv.DictReader(f), key=lambda r: int(r["episode_idx"]))

    assert len(rows_off) == len(rows_on) == 3
    must_match = (
        "episode_idx", "eval_seed", "score", "max_tile",
        "max_log_tile", "steps", "termination_reason",
        "win_1024", "win_2048", "win_4096", "win_8192",
    )
    for a, b in zip(rows_off, rows_on):
        for k in must_match:
            assert a[k] == b[k], f"Mismatch in column {k}: {a[k]} vs {b[k]}"

    assert (out_on / "moves.csv").exists()
    assert not (out_off / "moves.csv").exists()


# --- Task 25 ---

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _PRODUCTION_MODEL.exists(),
    reason="Production model not found",
)
def test_worker_crash_detection_marks_status_failed(tmp_path):
    """A worker that raises sets config.json status='failed' and exits non-zero.

    Crash is simulated via the BENCHMARK_FORCE_CRASH=1 env var hook in
    Benchmarker.run_episode. The harness writes status='failed' when it
    sees a "failed" message on the status queue.
    """
    out = tmp_path / "crash"
    out.mkdir()
    env = {**os.environ, "BENCHMARK_FORCE_CRASH": "1"}

    proc = subprocess.run(
        [sys.executable, "-m", "scripts.benchmark",
         str(_PRODUCTION_MODEL),
         "--n-runs", "3", "--depth", "0",
         "--workers", "1", "--device", "cpu",
         "--output", str(out), "--base-eval-seed", "0"],
        cwd=str(_REPO_ROOT), capture_output=True, text=True,
        timeout=60, env=env,
    )
    assert proc.returncode != 0, (
        f"Expected non-zero exit on worker crash, got {proc.returncode}\n"
        f"stderr:\n{proc.stderr}"
    )

    cfg_path = out / "config.json"
    assert cfg_path.exists(), "config.json should exist even on worker crash"
    with open(cfg_path) as f:
        cfg = json.load(f)
    assert cfg["status"] == "failed", (
        f"Expected status='failed', got {cfg.get('status')!r}"
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _PRODUCTION_MODEL.exists(),
    reason="Production model not found",
)
def test_sigkill_worker_detection_marks_status_failed(tmp_path):
    out = tmp_path / "sigkill"
    out.mkdir()
    env = {**os.environ, "BENCHMARK_FORCE_SIGKILL": "1"}

    result = _run_benchmark([
        str(_PRODUCTION_MODEL), "--n-runs", "1", "--depth", "0",
        "--workers", "1", "--device", "cpu", "--worker-timeout", "2",
        "--output", str(out), "--base-eval-seed", "0",
    ], timeout=30, env=env)

    assert result.returncode != 0
    with open(out / "config.json") as f:
        config = json.load(f)
    assert config["status"] == "failed"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _PRODUCTION_MODEL.exists(),
    reason="Production model not found",
)
def test_hung_worker_times_out_and_marks_status_failed(tmp_path):
    out = tmp_path / "hang"
    out.mkdir()
    env = {**os.environ, "BENCHMARK_FORCE_HANG": "1"}

    result = _run_benchmark([
        str(_PRODUCTION_MODEL), "--n-runs", "1", "--depth", "0",
        "--workers", "1", "--device", "cpu", "--worker-timeout", "0.1",
        "--output", str(out), "--base-eval-seed", "0",
    ], timeout=30, env=env)

    assert result.returncode != 0
    with open(out / "config.json") as f:
        config = json.load(f)
    assert config["status"] == "failed"
    assert config["worker_timeout"] == 0.1
    assert config["worker_inactivity_timeout"] == 0.1
    assert config["worker_timeout_scope"] == (
        "per-worker-inactivity-between-episode-heartbeats"
    )
