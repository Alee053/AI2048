"""Tests for visualizer handling of search outcomes."""
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from twenty_forty_eight_ai.utils.visualizer import Visualizer


class _ActionEnv:
    def __init__(self, *, terminated, truncated):
        self.game = SimpleNamespace(score=0, max_tile=0)
        self.terminated = terminated
        self.truncated = truncated
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        return None, 0.0, self.terminated, self.truncated, {}


@pytest.mark.parametrize(
    "stats",
    [
        {
            "best_move": -1,
            "has_legal_move": False,
            "search_complete": True,
            "failure_reason": "no_legal_move",
        },
        {
            "best_move": -1,
            "has_legal_move": True,
            "search_complete": False,
            "failure_reason": "search_incomplete",
        },
    ],
)
def test_invalid_search_outcome_stops_game_before_action_execution(stats):
    visualizer = Visualizer.__new__(Visualizer)
    visualizer._result_queue = queue.Queue()
    visualizer._result_queue.put((stats, 7))
    visualizer._game_id = 7
    visualizer._current_result = {"best_move": 0}
    visualizer._searching = True
    visualizer.terminated = False

    visualizer._on_search_complete()

    assert visualizer.terminated is True
    assert visualizer._searching is False
    assert visualizer._current_result is None
    assert visualizer.search_failure_reason == stats["failure_reason"]


def test_search_worker_marks_exception_as_terminal():
    visualizer = Visualizer.__new__(Visualizer)
    visualizer._search_event = threading.Event()
    visualizer._search_event.set()
    visualizer._worker_running = True
    visualizer._searching = True
    visualizer.search_depth = 3
    visualizer._current_board_for_search = np.zeros((4, 4), dtype=np.int8)
    visualizer._game_id = 3
    visualizer.terminated = False
    visualizer.search_failure_reason = None

    def fail_search(*args):
        visualizer._worker_running = False
        raise RuntimeError("evaluator failed")

    visualizer.searcher = SimpleNamespace(find_best_move=fail_search)

    visualizer._search_worker()

    assert visualizer.terminated is True
    assert visualizer.search_failure_reason == "search_exception"
    assert visualizer._searching is False


def test_reset_clears_search_failure_and_restarts_search():
    class PauseButton:
        def set_text(self, text):
            self.text = text

    class ResetEnv:
        def __init__(self):
            self.reset_called = False

        def reset(self):
            self.reset_called = True

    visualizer = Visualizer.__new__(Visualizer)
    visualizer._game_id = 4
    visualizer._searching = False
    visualizer._current_result = {"best_move": 0}
    visualizer.terminated = True
    visualizer.search_failure_reason = "search_exception"
    visualizer.paused = True
    visualizer.pause_button = PauseButton()
    visualizer.env = ResetEnv()
    visualizer.move_history = [1]
    visualizer.history_labels = []
    visualizer._last_history_count = 1
    visualizer.score_history = [1]
    visualizer.think_time_history = [1]
    visualizer.nodes_history = [1]
    visualizer.show_stats = False
    visualizer.cumulative = {}
    visualizer.search_thread = None
    restarted = []
    visualizer._start_search_if_idle = lambda: restarted.append(True)

    visualizer._reset_game()

    assert visualizer._game_id == 5
    assert visualizer.terminated is False
    assert visualizer.search_failure_reason is None
    assert visualizer.paused is False
    assert visualizer.env.reset_called is True
    assert restarted == [True]


@pytest.mark.parametrize(
    ("terminated", "truncated"),
    [(True, False), (False, True)],
)
def test_execute_action_stops_game_on_termination_or_truncation(
    terminated, truncated,
):
    visualizer = Visualizer.__new__(Visualizer)
    visualizer.env = _ActionEnv(
        terminated=terminated,
        truncated=truncated,
    )
    visualizer.terminated = False
    drawn = []
    visualizer._draw_game_over = lambda *args: drawn.append(args)

    visualizer._execute_action(0)

    assert visualizer.env.actions == [0]
    assert visualizer.terminated is True
    assert drawn == [(0, 1)]
