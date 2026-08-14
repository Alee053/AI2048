from __future__ import annotations

import numpy as np

from twenty_forty_eight_ai.env.d4_transforms import ACTION_TO_CANONICAL
from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.env.game import DOWN, LEFT, RIGHT, UP, Fast2048
from twenty_forty_eight_ai.utils.effective_config import (
    derive_d4_rank_seed_sequences,
)


_ACTIONS = (LEFT, UP, RIGHT, DOWN, LEFT, DOWN, RIGHT, UP)


def _game_trace(game: Fast2048, actions=_ACTIONS) -> list[tuple]:
    trace = [tuple(game.board.flatten().tolist())]
    for action in actions:
        if game.done:
            break
        game.move(action)
        trace.append(tuple(game.board.flatten().tolist()))
    return trace


def _spawn_trace(game: Fast2048, count: int = 16) -> list[tuple]:
    trace = []
    for _ in range(count):
        game.board.fill(0)
        game.generate_random()
        trace.append(tuple(game.board.flatten().tolist()))
    return trace


def test_same_seed_reproduces_board_and_complete_game_trace():
    first = Fast2048()
    second = Fast2048()

    first.reset(seed=2026)
    second.reset(seed=2026)

    assert _game_trace(first) == _game_trace(second)


def test_different_seeds_produce_different_random_streams():
    first = Fast2048()
    second = Fast2048()

    first.reset(seed=1)
    second.reset(seed=2)

    assert _spawn_trace(first) != _spawn_trace(second)


def test_repeated_explicit_reset_reproduces_the_sequence():
    game = Fast2048()

    game.reset(seed=123)
    first_trace = _game_trace(game)
    game.reset(seed=123)
    second_trace = _game_trace(game)

    assert first_trace == second_trace


def test_reset_without_seed_continues_the_current_stream():
    game = Fast2048()
    control = Fast2048()

    game.reset(seed=456)
    control.reset(seed=456)
    game.reset()
    control.reset()

    assert np.array_equal(game.board, control.board)
    assert game.score == control.score == 0
    assert game.done is control.done is False


def test_fast2048_instances_do_not_share_random_state():
    standalone = Fast2048()
    standalone.reset(seed=77)
    standalone_trace = _spawn_trace(standalone)

    interleaved = Fast2048()
    unrelated = Fast2048()
    interleaved.reset(seed=77)
    unrelated.reset(seed=88)
    interleaved_trace = []
    for _ in range(16):
        interleaved.board.fill(0)
        interleaved.generate_random()
        interleaved_trace.append(tuple(interleaved.board.flatten().tolist()))
        unrelated.board.fill(0)
        unrelated.generate_random()

    assert interleaved_trace == standalone_trace


def _rank_game_traces(training_seed: int, n_envs: int) -> list[list[tuple]]:
    from scripts.train import make_training_env_factories

    factories = make_training_env_factories(
        {"d4_augment": True},
        derive_d4_rank_seed_sequences(training_seed, n_envs),
    )
    traces = []
    environments = [factory() for factory in factories]
    try:
        for rank, environment in enumerate(environments):
            environment.reset(seed=training_seed + rank)
            traces.append(_spawn_trace(environment.env.game))
    finally:
        for environment in environments:
            environment.close()
    return traces


def test_environment_ranks_have_distinct_reproducible_game_streams():
    first = _rank_game_traces(training_seed=19, n_envs=3)
    second = _rank_game_traces(training_seed=19, n_envs=3)

    assert first == second
    assert len({tuple(trace) for trace in first}) == 3


def test_consuming_d4_rng_does_not_change_game_rng():
    plain = Game2048Env()
    augmented = Game2048Env(d4_augment=True, d4_seed=42)
    plain.reset(seed=314)
    augmented.reset(seed=314)

    for action in _ACTIONS:
        augmented._sample_d4()
        plain.game.move(action)
        augmented.game.move(action)
        np.testing.assert_array_equal(plain.game.board, augmented.game.board)


def test_d4_and_no_d4_match_in_canonical_coordinates():
    plain = Game2048Env()
    augmented = Game2048Env(d4_augment=True, d4_seed=99)
    plain.reset(seed=2718)
    augmented.reset(seed=2718)

    for canonical_action in _ACTIONS:
        agent_actions = np.flatnonzero(
            ACTION_TO_CANONICAL[augmented._current_d4] == canonical_action
        )
        assert agent_actions.size == 1

        plain.step(canonical_action)
        augmented.step(int(agent_actions[0]))
        np.testing.assert_array_equal(plain.game.board, augmented.game.board)

        if plain.game.done:
            break


def test_same_d4_seed_reproduces_transform_sequence():
    first = Game2048Env(d4_augment=True, d4_seed=1234)
    second = Game2048Env(d4_augment=True, d4_seed=1234)
    first.reset(seed=5678)
    second.reset(seed=5678)

    first_transforms = [first._current_d4]
    second_transforms = [second._current_d4]
    for action in _ACTIONS:
        first.step(action)
        second.step(action)
        first_transforms.append(first._current_d4)
        second_transforms.append(second._current_d4)

    assert first_transforms == second_transforms


def test_training_passes_training_seed_to_maskable_ppo(tmp_path, monkeypatch):
    import scripts.train as train_module

    captured = {}

    class FakeModel:
        num_timesteps = 0

        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def learn(self, *args, **kwargs):
            return self

        def save(self, path):
            captured["saved_path"] = path

    monkeypatch.setattr(train_module, "MaskablePPO", FakeModel)
    monkeypatch.setattr(train_module, "DummyVecEnv", lambda factories: object())
    monkeypatch.setattr(train_module, "WandbLoggingCallback", lambda: object())
    monkeypatch.setattr(train_module, "CheckpointCallback", lambda **kwargs: object())
    monkeypatch.setattr(train_module, "set_global_seed", lambda seed: None)
    monkeypatch.setattr(train_module, "persist_training_manifest", lambda *args: None)
    monkeypatch.setattr(train_module.wandb, "init", lambda **kwargs: object())

    train_module.train(
        {
            "project_name": "test",
            "run_name": "seed-test",
            "output_dir": str(tmp_path),
            "total_timesteps": 1,
            "n_envs": 2,
            "save_interval": 1,
            "features_dim": 8,
            "seed": 37,
            "env_kwargs": {"d4_augment": True},
            "ppo_params": {
                "n_steps": 1,
                "learning_rate": {
                    "type": "linear_decay",
                    "initial_value": 1e-4,
                },
            },
        }
    )

    assert captured["seed"] == 37
