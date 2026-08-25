"""Unit tests for benchmark argument validation and worker supervision."""
from __future__ import annotations

from argparse import Namespace
from queue import Empty
from types import SimpleNamespace

import multiprocessing as mp
import pytest


_CURRENT_FAKE_PROCESS = None
_LAST_FAKE_PROCESS = None


def test_parse_args_rejects_non_positive_n_runs():
    from scripts.benchmark import parse_args

    with pytest.raises(SystemExit):
        parse_args(["model.zip", "--n-runs", "0"])
    with pytest.raises(SystemExit):
        parse_args(["model.zip", "--n-runs", "-1"])


def test_parse_args_requires_positive_worker_timeout():
    from scripts.benchmark import parse_args

    assert parse_args(["model.zip"]).worker_timeout == 300.0
    assert parse_args(["model.zip", "--worker-timeout", "1.5"]).worker_timeout == 1.5
    with pytest.raises(SystemExit):
        parse_args(["model.zip", "--worker-timeout", "0"])


def test_build_config_records_worker_timeout(monkeypatch):
    from scripts.benchmark import build_config, parse_args

    monkeypatch.setattr("scripts.benchmark._git_dirty", lambda: False)
    monkeypatch.setattr("scripts.benchmark._git_commit", lambda: "commit")
    monkeypatch.setattr(
        "scripts.benchmark.collect_runtime_provenance",
        lambda **_: {},
    )

    args = parse_args(["model.zip", "--worker-timeout", "12.5"])
    config = build_config(
        args, "run", 100, "deterministic-offset", "2026-01-01T00:00:00Z",
    )

    assert config["worker_timeout"] == 12.5


def test_programmatic_runner_rejects_non_positive_n_runs_before_output_or_workers(
    monkeypatch, tmp_path,
):
    from scripts import benchmark_runner

    output_dir = tmp_path / "run"
    args = Namespace(
        n_runs=0,
        log_moves=False,
        yes_large_move_log=False,
        workers=1,
        output=str(output_dir),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "build_config",
        lambda *_: pytest.fail("config must not be built for invalid n_runs"),
    )
    monkeypatch.setattr(
        benchmark_runner.mp,
        "get_context",
        lambda *_: pytest.fail("workers must not be created for invalid n_runs"),
    )

    assert benchmark_runner.run_benchmark(args) == 1
    assert not output_dir.exists()


def _worker_queues():
    ctx = mp.get_context("spawn")
    return ctx.Queue(), ctx.Queue(), ctx.Event()


def test_worker_posts_started_and_complete_identity_ack(monkeypatch):
    from scripts import benchmark_worker

    results = [
        SimpleNamespace(episode_idx=0, eval_seed=100),
        SimpleNamespace(episode_idx=1, eval_seed=101),
    ]

    class FakeBenchmarker:
        def __init__(self, *args):
            self._results = iter(results)

        def run_episode(self, **kwargs):
            return next(self._results)

    result_queue, status_queue, stop_event = _worker_queues()
    monkeypatch.setattr(benchmark_worker, "Benchmarker", FakeBenchmarker)

    benchmark_worker.run_worker(
        3, "model.zip", "cpu", 0, [100, 101], False, "run", 100,
        result_queue, status_queue, stop_event,
    )

    statuses = [status_queue.get(timeout=2) for _ in range(4)]
    assert statuses[0] == {
        "worker_id": 3,
        "run_id": "run",
        "status": "started",
    }
    assert statuses[1] == {
        "worker_id": 3,
        "run_id": "run",
        "status": "episode_started",
        "episode_idx": 0,
        "eval_seed": 100,
    }
    assert statuses[2] == {
        "worker_id": 3,
        "run_id": "run",
        "status": "episode_started",
        "episode_idx": 1,
        "eval_seed": 101,
    }
    assert statuses[3] == {
        "worker_id": 3,
        "run_id": "run",
        "status": "completed",
        "expected_episode_indices": [0, 1],
        "actual_episode_indices": [0, 1],
        "expected_eval_seeds": [100, 101],
        "actual_eval_seeds": [100, 101],
    }


def test_worker_reports_base_exception_before_reraising(monkeypatch):
    from scripts import benchmark_worker

    class FakeBenchmarker:
        def __init__(self, *args):
            pass

        def run_episode(self, **kwargs):
            raise KeyboardInterrupt("stop")

    result_queue, status_queue, stop_event = _worker_queues()
    monkeypatch.setattr(benchmark_worker, "Benchmarker", FakeBenchmarker)

    with pytest.raises(KeyboardInterrupt):
        benchmark_worker.run_worker(
            0, "model.zip", "cpu", 0, [42], False, "run", 42,
            result_queue, status_queue, stop_event,
        )

    assert status_queue.get(timeout=2)["status"] == "started"
    assert status_queue.get(timeout=2)["status"] == "episode_started"
    failed = status_queue.get(timeout=2)
    assert failed["worker_id"] == 0
    assert failed["run_id"] == "run"
    assert failed["status"] == "failed"
    assert "KeyboardInterrupt" in failed["error"]


def test_runner_accepts_valid_completion(monkeypatch, tmp_path):
    code, writer = _run_with_fake_worker(monkeypatch, tmp_path, "valid")

    assert code == 0
    assert writer.configs[-1]["status"] == "completed"
    assert len(writer.rows) == 1


@pytest.mark.parametrize("mode", ["invalid_cap_hits", "invalid_moves_unresolved"])
def test_runner_rejects_search_failure_in_normal_mode(monkeypatch, tmp_path, mode):
    code, writer = _run_with_fake_worker(monkeypatch, tmp_path, mode)

    assert code == 2
    assert writer.configs[-1]["status"] == "failed"
    assert writer.rows == []
    assert writer.summaries[-1]["error"] in {
        "search reported 1 cap hits",
        "search reported 1 unresolved moves",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_cap_hits", False),
        ("total_cap_hits", 0.5),
        ("total_cap_hits", -1),
        ("total_cap_hits", "0"),
        ("total_moves_unresolved", True),
        ("total_moves_unresolved", 0.5),
        ("total_moves_unresolved", -1),
        ("total_moves_unresolved", "0"),
    ],
)
def test_paper_search_failure_reason_rejects_malformed_counters(field, value):
    from scripts.benchmark_runner import paper_search_failure_reason

    row = {"total_cap_hits": 0, "total_moves_unresolved": 0}
    row[field] = value

    reason = paper_search_failure_reason(row)

    assert reason is not None
    assert "invalid" in reason


def _heartbeat_worker_state():
    return {
        0: {
            "run_id": "run",
            "expected_episode_indices": [0],
            "expected_eval_seeds": [100],
            "expected_eval_seed_by_index": {0: 100},
            "started": False,
            "completed": False,
            "last_progress": None,
            "completed_at": None,
        },
    }


def test_runner_accepts_episode_heartbeat_and_updates_progress():
    from scripts import benchmark_runner

    states = _heartbeat_worker_state()

    assert benchmark_runner._handle_worker_status(
        {"worker_id": 0, "run_id": "run", "status": "started"},
        states,
        progress_at=1.0,
    ) is None
    assert benchmark_runner._handle_worker_status(
        {
            "worker_id": 0,
            "run_id": "run",
            "status": "episode_started",
            "episode_idx": 0,
            "eval_seed": 100,
        },
        states,
        progress_at=2.0,
    ) is None
    assert states[0]["last_progress"] == 2.0


def test_runner_rejects_mismatched_episode_heartbeat():
    from scripts import benchmark_runner

    states = _heartbeat_worker_state()
    benchmark_runner._handle_worker_status(
        {"worker_id": 0, "run_id": "run", "status": "started"},
        states,
        progress_at=1.0,
    )

    error = benchmark_runner._handle_worker_status(
        {
            "worker_id": 0,
            "run_id": "run",
            "status": "episode_started",
            "episode_idx": 0,
            "eval_seed": 101,
        },
        states,
        progress_at=2.0,
    )

    assert error is not None
    assert "eval_seed" in error


def test_keyboard_interrupt_from_episode_serialization_reaches_interrupt_handler(
    monkeypatch, tmp_path,
):
    code, writer = _run_with_fake_worker(
        monkeypatch, tmp_path, "serialize_keyboard_interrupt",
    )

    assert code == 2
    assert writer.configs[-1]["status"] == "interrupted"
    assert writer.configs[-1]["interrupted"] is True


@pytest.mark.parametrize("mode", ["interrupt_start", "interrupt_liveness"])
def test_keyboard_interrupt_during_worker_supervision_is_interrupted(
    monkeypatch, tmp_path, mode,
):
    code, writer = _run_with_fake_worker(monkeypatch, tmp_path, mode)

    assert code == 2
    assert writer.configs[-1]["status"] == "interrupted"
    assert writer.configs[-1]["interrupted"] is True


@pytest.mark.parametrize(
    ("mode", "expected_status"),
    [("interrupt_start", "interrupted"), ("start_exception", "failed")],
)
def test_start_failure_registers_candidate_for_cleanup(
    monkeypatch, tmp_path, mode, expected_status,
):
    from scripts import benchmark_runner

    monkeypatch.setattr(benchmark_runner, "WORKER_CLEANUP_JOIN_TIMEOUT_S", 0.01)
    code, writer = _run_with_fake_worker(monkeypatch, tmp_path, mode)

    assert code == 2
    assert writer.configs[-1]["status"] == expected_status
    assert _LAST_FAKE_PROCESS is not None
    assert _LAST_FAKE_PROCESS.terminate_called is True


def test_failed_status_ack_wins_over_interrupt_and_stopped_is_ignored(
    monkeypatch, tmp_path,
):
    code, writer = _run_with_fake_worker(
        monkeypatch, tmp_path, "interrupt_status_race",
    )

    assert code == 2
    assert writer.configs[-1]["status"] == "failed"
    assert writer.configs[-1]["interrupted"] is False
    assert "failed" in writer.summaries[-1]["error"]


def test_runner_drains_completed_ack_after_clean_exit(monkeypatch, tmp_path):
    code, writer = _run_with_fake_worker(monkeypatch, tmp_path, "delayed_ack")

    assert code == 0
    assert writer.configs[-1]["status"] == "completed"
    assert len(writer.rows) == 1


def test_completed_ack_allows_process_to_exit_within_exit_grace(
    monkeypatch, tmp_path,
):
    from scripts import benchmark_runner

    monkeypatch.setattr(benchmark_runner, "WORKER_EXIT_GRACE_S", 0.3)
    code, writer = _run_with_fake_worker(
        monkeypatch, tmp_path, "completed_alive_then_exit", timeout=0.01,
    )

    assert code == 0
    assert writer.configs[-1]["status"] == "completed"


def test_completed_ack_fails_if_process_exceeds_exit_grace(monkeypatch, tmp_path):
    from scripts import benchmark_runner

    monkeypatch.setattr(benchmark_runner, "WORKER_EXIT_GRACE_S", 0.01)
    monkeypatch.setattr(benchmark_runner, "WORKER_CLEANUP_JOIN_TIMEOUT_S", 0.01)
    code, writer = _run_with_fake_worker(
        monkeypatch, tmp_path, "completed_never_exits", timeout=1.0,
    )

    assert code == 2
    assert writer.configs[-1]["status"] == "failed"
    assert "exit grace" in writer.summaries[-1]["error"]


def test_runner_rejects_status_ack_from_another_run(monkeypatch, tmp_path):
    code, writer = _run_with_fake_worker(
        monkeypatch, tmp_path, "bad_status_run_id",
    )

    assert code == 2
    assert writer.configs[-1]["status"] == "failed"
    assert writer.rows == []


@pytest.mark.parametrize("mode", ["bad_run_id", "bad_worker_id", "bad_seed"])
def test_runner_rejects_unexpected_result_identity(monkeypatch, tmp_path, mode):
    code, writer = _run_with_fake_worker(monkeypatch, tmp_path, mode)

    assert code == 2
    assert writer.configs[-1]["status"] == "failed"
    assert writer.rows == []
    assert "unexpected" in writer.summaries[-1]["error"] or "expected" in writer.summaries[-1]["error"]


def test_runner_reports_missing_result_after_completed_ack_grace_window(
    monkeypatch, tmp_path,
):
    code, writer = _run_with_fake_worker(
        monkeypatch, tmp_path, "missing_result", n_runs=2, timeout=1.0,
    )

    assert code == 2
    assert writer.configs[-1]["status"] == "failed"
    assert "missing" in writer.summaries[-1]["error"]


def test_stop_workers_reports_unstoppable_process(monkeypatch):
    from scripts import benchmark_runner

    monkeypatch.setattr(benchmark_runner, "WORKER_CLEANUP_JOIN_TIMEOUT_S", 0.01)

    class UnstoppableProcess:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

        def terminate(self):
            pass

        def kill(self):
            pass

    error = benchmark_runner._stop_workers(
        [(4, UnstoppableProcess())], _FakeEvent(),
    )

    assert error is not None
    assert "worker 4" in error


def test_stop_workers_attempts_kill_after_terminate_failure():
    from scripts import benchmark_runner

    class TerminateRaisesProcess:
        def __init__(self):
            self.alive = True
            self.kill_called = False

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            raise RuntimeError("join failed")

        def terminate(self):
            raise RuntimeError("terminate failed")

        def kill(self):
            self.kill_called = True
            self.alive = False

    process = TerminateRaisesProcess()
    error = benchmark_runner._stop_workers([(7, process)], _FakeEvent())

    assert process.kill_called is True
    assert error is not None
    assert process.is_alive() is False


def test_stop_workers_uses_one_global_deadline(monkeypatch):
    from scripts import benchmark_runner

    clock = [0.0]
    monkeypatch.setattr(benchmark_runner, "WORKER_CLEANUP_JOIN_TIMEOUT_S", 1.0)
    monkeypatch.setattr(
        benchmark_runner.time, "monotonic", lambda: clock[0],
    )

    class SlowProcess:
        def __init__(self):
            self.alive = True
            self.kill_called = False

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            clock[0] += 0.6

        def terminate(self):
            pass

        def kill(self):
            self.kill_called = True
            self.alive = False

    processes = [SlowProcess(), SlowProcess()]
    error = benchmark_runner._stop_workers(
        [(index, process) for index, process in enumerate(processes)],
        _FakeEvent(),
    )

    assert error is None
    assert clock[0] < 2.0
    assert all(process.kill_called for process in processes)


def test_cleanup_failure_prevents_completed_status(monkeypatch, tmp_path):
    from scripts import benchmark_runner

    monkeypatch.setattr(
        benchmark_runner,
        "_stop_workers",
        lambda workers, stop_event: "worker 0 remains alive",
    )
    code, writer = _run_with_fake_worker(monkeypatch, tmp_path, "valid")

    assert code == 2
    assert writer.configs[-1]["status"] == "failed"
    assert "cleanup" in writer.summaries[-1]["error"]


def test_worker_timeout_is_per_worker_inactivity(monkeypatch, tmp_path):
    code, writer = _run_with_fake_worker(
        monkeypatch, tmp_path, "paced", n_runs=8, timeout=0.2,
    )

    assert code == 0
    assert writer.configs[-1]["status"] == "completed"
    assert len(writer.rows) == 8


def test_worker_timeout_rejects_blocked_episode_after_heartbeat(monkeypatch, tmp_path):
    from scripts import benchmark_runner

    monkeypatch.setattr(benchmark_runner, "WORKER_CLEANUP_JOIN_TIMEOUT_S", 0.01)
    code, writer = _run_with_fake_worker(
        monkeypatch, tmp_path, "episode_timeout", timeout=0.01,
    )

    assert code == 2
    assert writer.configs[-1]["status"] == "failed"
    assert "inactive" in writer.summaries[-1]["error"]


def test_interrupted_result_drain_is_bounded(monkeypatch):
    from scripts import benchmark_runner

    class EndlessQueue:
        def get_nowait(self):
            return object()

    messages = benchmark_runner._drain_queue_bounded(
        EndlessQueue(), 0.1, max_messages=3,
    )

    assert len(messages) == 3


@pytest.mark.parametrize("mode", ["missing_move", "extra_move"])
def test_runner_requires_move_count_to_match_steps(monkeypatch, tmp_path, mode):
    code, writer = _run_with_fake_worker(
        monkeypatch, tmp_path, mode, log_moves=True,
    )

    assert code == 2
    assert writer.configs[-1]["status"] == "failed"
    assert writer.moves == []


def test_interrupt_with_existing_worker_failure_is_failed(monkeypatch, tmp_path):
    code, writer = _run_with_fake_worker(
        monkeypatch, tmp_path, "interrupt_failed_exit",
    )

    assert code == 2
    assert writer.configs[-1]["status"] == "failed"
    assert writer.configs[-1]["interrupted"] is False
    assert "exited with code" in writer.summaries[-1]["error"]


@pytest.mark.parametrize(
    "mode",
    [
        "bad_move_run_id",
        "bad_move_worker_id",
        "bad_move_episode_idx",
        "duplicate_move",
    ],
)
def test_runner_rejects_invalid_move_records(monkeypatch, tmp_path, mode):
    code, writer = _run_with_fake_worker(
        monkeypatch, tmp_path, mode, log_moves=True,
    )

    assert code == 2
    assert writer.configs[-1]["status"] == "failed"
    assert writer.moves == []


def test_runner_writes_valid_move_records(monkeypatch, tmp_path):
    code, writer = _run_with_fake_worker(
        monkeypatch, tmp_path, "valid", log_moves=True,
    )

    assert code == 0
    assert writer.moves == [{"move_idx": 0}]


def test_interrupt_drains_only_valid_queued_results(monkeypatch, tmp_path):
    code, writer = _run_with_fake_worker(
        monkeypatch, tmp_path, "interrupt_drain",
    )

    assert code == 2
    assert writer.configs[-1]["status"] == "interrupted"
    assert len(writer.rows) == 1
    assert writer.rows[0]["episode_idx"] == 0


class _ImmediateQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def get_nowait(self):
        if not self.items:
            raise Empty
        return self.items.pop(0)


class _FakeEvent:
    def __init__(self):
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True


class _FakeProcess:
    def __init__(self, target, args, mode):
        self._target = target
        self._args = args
        self._mode = mode
        self._alive = False
        self.exitcode = None
        self._progress_callback = None
        self._finish_pending = False
        self._delayed_status_callback = None
        self._on_terminate = None
        self._on_interrupt = None
        self._liveness_interrupt_raised = False
        self._exit_checks_before_exit = 0
        self.terminate_called = False
        self.kill_called = False

    def start(self):
        global _CURRENT_FAKE_PROCESS, _LAST_FAKE_PROCESS
        if self._mode in {"interrupt_start", "start_exception"}:
            self._alive = True
            _LAST_FAKE_PROCESS = self
            if self._mode == "interrupt_start":
                raise KeyboardInterrupt("stop during worker start")
            raise RuntimeError("worker start failed")
        if self._mode == "interrupt_status_race":
            self._alive = True
            _LAST_FAKE_PROCESS = self
            _CURRENT_FAKE_PROCESS = self
            try:
                self._target(*self._args)
            finally:
                _CURRENT_FAKE_PROCESS = None
            return
        if self._mode == "episode_timeout":
            self._alive = True
            _CURRENT_FAKE_PROCESS = self
            try:
                self._target(*self._args)
            finally:
                _CURRENT_FAKE_PROCESS = None
            return
        if self._mode in {
            "completed_alive_then_exit", "completed_never_exits",
        }:
            self._alive = True
            _CURRENT_FAKE_PROCESS = self
            try:
                self._target(*self._args)
            finally:
                _CURRENT_FAKE_PROCESS = None
            return
        if self._mode == "paced":
            self._alive = True
            _CURRENT_FAKE_PROCESS = self
            try:
                self._target(*self._args)
            finally:
                _CURRENT_FAKE_PROCESS = None
            return
        if self._mode == "delayed_ack":
            self._alive = True
            _CURRENT_FAKE_PROCESS = self
            try:
                self._target(*self._args)
            finally:
                _CURRENT_FAKE_PROCESS = None
            return
        if self._mode == "interrupt_drain":
            self._alive = True
            _CURRENT_FAKE_PROCESS = self
            try:
                self._target(*self._args)
            finally:
                _CURRENT_FAKE_PROCESS = None
            return
        if self._mode == "interrupt_failed_exit":
            self._alive = True
            _LAST_FAKE_PROCESS = self
            self._target(*self._args)
            return
        if self._mode == "timeout":
            self._alive = True
            return
        self._target(*self._args)
        self._alive = False
        self.exitcode = 9 if self._mode == "nonzero" else 0

    def is_alive(self):
        if (
            self._mode == "interrupt_liveness"
            and not self._liveness_interrupt_raised
        ):
            self._liveness_interrupt_raised = True
            raise KeyboardInterrupt("stop during worker liveness check")
        if self._mode == "completed_alive_then_exit" and self._alive:
            if self._exit_checks_before_exit == 0:
                self._alive = False
                self.exitcode = 0
            else:
                self._exit_checks_before_exit -= 1
        if self._mode == "paced" and self._alive:
            if self._finish_pending:
                self._alive = False
                self.exitcode = 0
            elif self._progress_callback is not None:
                callback = self._progress_callback
                self._progress_callback = None
                callback()
        if self._mode == "delayed_ack" and not self._alive:
            if self._delayed_status_callback is not None:
                callback = self._delayed_status_callback
                self._delayed_status_callback = None
                callback()
        return self._alive

    def join(self, timeout=None):
        if self._mode in {"interrupt_drain", "interrupt_status_race"}:
            if self._on_terminate is not None:
                callback = self._on_terminate
                self._on_terminate = None
                callback()
            self._alive = False
            self.exitcode = 0
        return None

    def terminate(self):
        self.terminate_called = True
        if self._on_terminate is not None:
            callback = self._on_terminate
            self._on_terminate = None
            callback()
        self._alive = False
        self.exitcode = -15

    def kill(self):
        self.kill_called = True
        self._alive = False
        self.exitcode = -9


class _FakeContext:
    def __init__(self, mode):
        self.mode = mode

    def Queue(self):
        return _ImmediateQueue()

    def Event(self):
        return _FakeEvent()

    def Process(self, target, args):
        return _FakeProcess(target, args, self.mode)


class _RecordingWriter:
    instances = []

    def __init__(self, output_dir, log_moves):
        self.rows = []
        self.moves = []
        self.configs = []
        self.summaries = []
        self.__class__.instances.append(self)

    def write_config(self, config):
        self.configs.append(dict(config))

    def writerow_episode(self, row):
        self.rows.append(dict(row))

    def writerow_moves(self, rows):
        self.moves.extend(rows)

    def write_summary(self, summary):
        self.summaries.append(dict(summary))

    def close(self):
        pass


def _run_with_fake_worker(
    monkeypatch, tmp_path, mode, n_runs=1, timeout=1.0, log_moves=False,
):
    from scripts import benchmark_runner
    from scripts.benchmark import parse_args

    global _CURRENT_FAKE_PROCESS, _LAST_FAKE_PROCESS
    _CURRENT_FAKE_PROCESS = None
    _LAST_FAKE_PROCESS = None

    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"model")
    output_path = tmp_path / mode
    argv = [
        str(model_path), "--n-runs", str(n_runs), "--workers", "1",
        "--base-eval-seed", "100", "--worker-timeout", str(timeout),
        "--output", str(output_path),
    ]
    if log_moves:
        argv.append("--log-moves")
    args = parse_args(argv)

    def fake_build_config(args, run_name, env_seed_base, eval_seed_strategy, started_at):
        return {
            "run_name": run_name,
            "n_runs": args.n_runs,
            "use_expectimax": False,
            "status": "running",
            "interrupted": False,
            "worker_timeout": args.worker_timeout,
        }

    def fake_episode_to_row(result):
        if mode == "serialize_keyboard_interrupt":
            raise KeyboardInterrupt
        row = {
            "episode_idx": result.episode_idx,
            "eval_seed": result.eval_seed,
            "score": result.score,
            "max_tile": result.max_tile,
            "max_log_tile": 1,
            "steps": result.steps,
            "termination_reason": "board_full",
            "use_expectimax": False,
            "total_moves_unresolved": (
                1 if mode == "invalid_moves_unresolved" else 0
            ),
            "total_cap_hits": 1 if mode == "invalid_cap_hits" else 0,
        }
        from scripts.benchmark_io import OUTCOME_FINGERPRINT_COLUMNS
        for field in OUTCOME_FINGERPRINT_COLUMNS:
            if field not in row:
                row[field] = False if field.startswith("win_") else 0
        return row

    def fake_worker(
        worker_id, model_path, device, depth, eval_seeds, log_moves, run_id,
        env_seed_base, result_queue, status_queue, stop_event,
    ):
        expected_indices = [seed - env_seed_base for seed in eval_seeds]
        status_queue.put({
            "worker_id": worker_id,
            "run_id": (
                "unexpected-run" if mode == "bad_status_run_id" else run_id
            ),
            "status": "started",
        })
        if mode == "delayed_ack":
            process = _CURRENT_FAKE_PROCESS
            for index, seed in zip(expected_indices, eval_seeds):
                result_queue.put(SimpleNamespace(
                    episode_idx=index,
                    eval_seed=seed,
                    run_id=run_id,
                    worker_id=worker_id,
                    score=10,
                    max_tile=2,
                    steps=1,
                    mean_nps=0.0,
                    move_records=[],
                ))

            def send_completed():
                status_queue.put({
                    "worker_id": worker_id,
                    "run_id": run_id,
                    "status": "completed",
                    "expected_episode_indices": expected_indices,
                    "actual_episode_indices": expected_indices,
                    "expected_eval_seeds": list(eval_seeds),
                    "actual_eval_seeds": list(eval_seeds),
                })

            process._alive = False
            process.exitcode = 0
            process._delayed_status_callback = send_completed
            return
        if mode == "paced":
            process = _CURRENT_FAKE_PROCESS
            progress_index = 0

            def progress():
                nonlocal progress_index
                index = expected_indices[progress_index]
                seed = eval_seeds[progress_index]
                status_queue.put({
                    "worker_id": worker_id,
                    "run_id": run_id,
                    "status": "episode_started",
                    "episode_idx": index,
                    "eval_seed": seed,
                })
                result_queue.put(SimpleNamespace(
                    episode_idx=index,
                    eval_seed=seed,
                    run_id=run_id,
                    worker_id=worker_id,
                    score=10,
                    max_tile=2,
                    steps=1,
                    mean_nps=0.0,
                    move_records=[],
                ))
                progress_index += 1
                if progress_index == len(expected_indices):
                    status_queue.put({
                        "worker_id": worker_id,
                        "run_id": run_id,
                        "status": "completed",
                        "expected_episode_indices": expected_indices,
                        "actual_episode_indices": expected_indices,
                        "expected_eval_seeds": list(eval_seeds),
                        "actual_eval_seeds": list(eval_seeds),
                    })
                    process._finish_pending = True
                else:
                    process._progress_callback = progress

            process._progress_callback = progress
            return
        if mode == "timeout":
            return
        if mode == "episode_timeout":
            status_queue.put({
                "worker_id": worker_id,
                "run_id": run_id,
                "status": "episode_started",
                "episode_idx": expected_indices[0],
                "eval_seed": eval_seeds[0],
            })
            return
        if mode == "interrupt_drain":
            process = _CURRENT_FAKE_PROCESS
            valid_result = SimpleNamespace(
                episode_idx=0,
                eval_seed=eval_seeds[0],
                run_id=run_id,
                worker_id=worker_id,
                score=10,
                max_tile=2,
                steps=1,
                mean_nps=0.0,
                move_records=[],
            )
            invalid_result = SimpleNamespace(
                episode_idx=99,
                eval_seed=999,
                run_id=run_id,
                worker_id=worker_id,
                score=10,
                max_tile=2,
                steps=1,
                mean_nps=0.0,
                move_records=[],
            )
            process._on_terminate = lambda: (
                result_queue.put(valid_result), result_queue.put(invalid_result),
            )
            return
        if mode == "interrupt_failed_exit":
            return
        if mode == "interrupt_status_race":
            process = _CURRENT_FAKE_PROCESS
            process._on_interrupt = lambda: status_queue.put({
                "worker_id": worker_id,
                "run_id": run_id,
                "status": "failed",
                "error": "worker failed before SIGINT",
            })
            process._on_terminate = lambda: status_queue.put({
                "worker_id": worker_id,
                "run_id": run_id,
                "status": "stopped",
            })
            return
        if mode in {"completed_alive_then_exit", "completed_never_exits"}:
            process = _CURRENT_FAKE_PROCESS
            result_queue.put(SimpleNamespace(
                episode_idx=0,
                eval_seed=eval_seeds[0],
                run_id=run_id,
                worker_id=worker_id,
                score=10,
                max_tile=2,
                steps=1,
                mean_nps=0.0,
                move_records=[],
            ))
            status_queue.put({
                "worker_id": worker_id,
                "run_id": run_id,
                "status": "completed",
                "expected_episode_indices": [0],
                "actual_episode_indices": [0],
                "expected_eval_seeds": list(eval_seeds),
                "actual_eval_seeds": list(eval_seeds),
            })
            if mode == "completed_alive_then_exit":
                process._exit_checks_before_exit = 3
            return
        for index, seed in zip(expected_indices, eval_seeds):
            result = SimpleNamespace(
                episode_idx=index,
                eval_seed=seed,
                run_id=("unexpected-run" if mode == "bad_run_id" else run_id),
                worker_id=(worker_id + 1 if mode == "bad_worker_id" else worker_id),
                score=10,
                max_tile=2,
                steps=1,
                mean_nps=0.0,
                move_records=[],
            )
            if log_moves:
                move_records = [SimpleNamespace(
                    run_id=(
                        "unexpected-run"
                        if mode == "bad_move_run_id" else run_id
                    ),
                    worker_id=(
                        worker_id + 1
                        if mode == "bad_move_worker_id" else worker_id
                    ),
                    episode_idx=(
                        index + 1
                        if mode == "bad_move_episode_idx" else index
                    ),
                    move_idx=0,
                )]
                if mode == "duplicate_move":
                    move_records.append(SimpleNamespace(
                        run_id=run_id,
                        worker_id=worker_id,
                        episode_idx=index,
                        move_idx=0,
                    ))
                if mode == "extra_move":
                    move_records.append(SimpleNamespace(
                        run_id=run_id,
                        worker_id=worker_id,
                        episode_idx=index,
                        move_idx=1,
                    ))
                if mode == "missing_move":
                    result.steps = 2
                result.move_records = move_records
            if mode == "bad_seed":
                result.eval_seed += 1000
            if mode == "missing_result" and index == expected_indices[-1]:
                continue
            result_queue.put(result)
        if mode == "duplicate":
            result_queue.put(result)
        if mode != "clean_exit":
            status_queue.put({
                "worker_id": worker_id,
                "run_id": run_id,
                "status": "completed",
                "expected_episode_indices": expected_indices,
                "actual_episode_indices": expected_indices,
                "expected_eval_seeds": list(eval_seeds),
                "actual_eval_seeds": list(eval_seeds),
            })

    _RecordingWriter.instances.clear()
    monkeypatch.setattr(benchmark_runner, "build_config", fake_build_config)
    monkeypatch.setattr(benchmark_runner, "CSVWriter", _RecordingWriter)
    monkeypatch.setattr(benchmark_runner, "episode_to_row", fake_episode_to_row)
    monkeypatch.setattr(
        benchmark_runner,
        "move_to_row",
        lambda move: {"move_idx": move.move_idx},
    )
    monkeypatch.setattr(
        benchmark_runner.mp,
        "get_context",
        lambda name: _FakeContext(mode),
    )
    monkeypatch.setattr(benchmark_runner, "run_worker", fake_worker)
    if mode in {
        "interrupt_drain", "interrupt_failed_exit", "interrupt_status_race",
    }:
        original_sleep = benchmark_runner.time.sleep
        interrupt_raised = False

        def interrupt_sleep(_):
            nonlocal interrupt_raised
            if interrupt_raised:
                return original_sleep(_)
            interrupt_raised = True
            if mode == "interrupt_failed_exit":
                _LAST_FAKE_PROCESS._alive = False
                _LAST_FAKE_PROCESS.exitcode = -9
            if mode == "interrupt_status_race":
                _LAST_FAKE_PROCESS._on_interrupt()
            raise KeyboardInterrupt

        monkeypatch.setattr(
            benchmark_runner.time,
            "sleep",
            interrupt_sleep,
        )

    return benchmark_runner.run_benchmark(args), _RecordingWriter.instances[-1]


@pytest.mark.parametrize("mode", ["clean_exit", "nonzero", "duplicate"])
def test_runner_marks_invalid_worker_completion_failed(monkeypatch, tmp_path, mode):
    code, writer = _run_with_fake_worker(monkeypatch, tmp_path, mode)

    assert code == 2
    assert writer.configs[-1]["status"] == "failed"
    assert len(writer.rows) == 1


def test_runner_times_out_worker_without_response(monkeypatch, tmp_path):
    code, writer = _run_with_fake_worker(
        monkeypatch, tmp_path, "timeout", timeout=0.01,
    )

    assert code == 2
    assert writer.configs[-1]["status"] == "failed"
    assert "inactive" in writer.summaries[-1]["error"]
