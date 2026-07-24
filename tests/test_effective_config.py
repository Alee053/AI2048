from pathlib import Path

import numpy as np
from stable_baselines3.common.monitor import Monitor

from twenty_forty_eight_ai.utils.effective_config import (
    compare_behavioral_configs,
    derive_d4_rank_seed_sequences,
    load_effective_config,
)


CONFIG_DIR = Path(__file__).parents[1] / "configs" / "train"
D4_CONFIG = CONFIG_DIR / "hybrid_ppo_v3.yaml"
NO_D4_CONFIG = CONFIG_DIR / "hybrid_ppo_v3_no_d4.yaml"


def test_d4_rank_spawn_keys_are_stable_and_distinct():
    first = derive_d4_rank_seed_sequences(1234, 4)
    second = derive_d4_rank_seed_sequences(1234, 4)

    assert [sequence.spawn_key for sequence in first] == [
        sequence.spawn_key for sequence in second
    ]
    assert [sequence.spawn_key for sequence in first] == [(0,), (1,), (2,), (3,)]


def test_v3_conditions_differ_only_by_d4_augment():
    diff = compare_behavioral_configs(
        load_effective_config(D4_CONFIG),
        load_effective_config(NO_D4_CONFIG),
    )

    assert diff == {"env_kwargs.d4_augment": (True, False)}


def test_resolved_training_config_records_d4_seed_provenance():
    from scripts.train import resolve_training_config

    effective_config, d4_rank_seed_sequences = resolve_training_config(
        {"seed": 9, "n_envs": 3, "env_kwargs": {"d4_augment": False}}
    )

    assert effective_config["env_kwargs"] == {"d4_augment": False}
    assert effective_config["root_training_seed"] == 9
    assert effective_config["d4_seed_derivation"] == (
        "numpy.random.SeedSequence(seed).spawn(n_envs)"
    )
    assert effective_config["d4_rank_spawn_keys"] == [[0], [1], [2]]
    assert [sequence.spawn_key for sequence in d4_rank_seed_sequences] == [
        (0,),
        (1,),
        (2,),
    ]


def test_training_env_factories_wrap_each_rank_and_assign_its_d4_seed_sequence():
    from scripts.train import make_training_env_factories

    d4_rank_seed_sequences = derive_d4_rank_seed_sequences(101, 2)
    expected_rngs = [
        np.random.default_rng(sequence) for sequence in d4_rank_seed_sequences
    ]
    factories = make_training_env_factories(
        {"d4_augment": True},
        d4_rank_seed_sequences,
    )
    first = factories[0]()
    second = factories[1]()
    try:
        assert isinstance(first, Monitor)
        assert isinstance(second, Monitor)
        assert first.env.d4_augment is True
        assert second.env.d4_augment is True
        assert first.env._d4_rng.integers(0, 2**32) == expected_rngs[0].integers(0, 2**32)
        assert second.env._d4_rng.integers(0, 2**32) == expected_rngs[1].integers(0, 2**32)
    finally:
        first.close()
        second.close()
