import pytest

from scripts.train import linear_decay_learning_rate, make_linear_decay_schedule


INITIAL_LR = 1e-3
TOTAL_TIMESTEPS = 100


def test_linear_decay_starts_at_initial_learning_rate():
    assert linear_decay_learning_rate(INITIAL_LR, 0, TOTAL_TIMESTEPS) == pytest.approx(
        INITIAL_LR
    )


def test_linear_decay_is_halfway_at_half_budget():
    assert linear_decay_learning_rate(INITIAL_LR, 50, TOTAL_TIMESTEPS) == pytest.approx(
        INITIAL_LR / 2
    )


def test_linear_decay_reaches_zero_at_budget():
    assert linear_decay_learning_rate(INITIAL_LR, TOTAL_TIMESTEPS, TOTAL_TIMESTEPS) == 0.0


def test_linear_decay_clamps_timestep_beyond_budget():
    assert linear_decay_learning_rate(INITIAL_LR, TOTAL_TIMESTEPS + 1, TOTAL_TIMESTEPS) == 0.0


@pytest.mark.parametrize("timestep", [-100, 0, 1, 50, 100, 101, 10_000])
def test_linear_decay_never_returns_negative_learning_rate(timestep):
    assert linear_decay_learning_rate(INITIAL_LR, timestep, TOTAL_TIMESTEPS) >= 0.0


def test_schedule_clamps_negative_progress_remaining():
    schedule = make_linear_decay_schedule(INITIAL_LR, TOTAL_TIMESTEPS)

    assert schedule(1.0) == pytest.approx(INITIAL_LR)
    assert schedule(0.5) == pytest.approx(INITIAL_LR / 2)
    assert schedule(0.0) == 0.0
    assert schedule(-0.1) == 0.0
