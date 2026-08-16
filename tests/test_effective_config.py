from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.monitor import Monitor

from twenty_forty_eight_ai.utils.effective_config import (
    compare_behavioral_configs,
    derive_d4_rank_seed_sequences,
    load_effective_config,
    validate_v3_experiment_config,
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


def test_v3_conditions_share_the_same_training_definition():
    d4 = load_effective_config(D4_CONFIG)["experiment_definition"]
    no_d4 = load_effective_config(NO_D4_CONFIG)["experiment_definition"]

    assert d4 == no_d4 == {
        "name": "v3",
        "policy_class": (
            "twenty_forty_eight_ai.agent.policy.ValueNormalizedMaskablePolicy"
        ),
        "ppo_class": (
            "twenty_forty_eight_ai.agent.ppo.ValueHeadLRMaskablePPO"
        ),
        "value_head_lr_multiplier": 10.0,
    }


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("d4_augment", [True, False])
def test_v3_seed_sweep_config_validates_from_definition(seed, d4_augment):
    config = load_effective_config(D4_CONFIG if d4_augment else NO_D4_CONFIG)
    config["seed"] = seed
    config["run_name"] = f"{config['run_name']}-seed{seed}"

    validate_v3_experiment_config(config)


@pytest.mark.parametrize(
    "run_name", ["hybrid_ppo_v3", "hybrid_ppo_v3_no_d4", "hybrid_ppo_v3-seed0"]
)
def test_reserved_v3_run_name_cannot_fall_back_to_legacy_config(run_name):
    with pytest.raises(ValueError, match="experiment_definition"):
        validate_v3_experiment_config({"run_name": run_name})


def test_malformed_experiment_definition_fails_with_configuration_error():
    with pytest.raises(ValueError, match="experiment_definition"):
        validate_v3_experiment_config(
            {"run_name": "custom", "experiment_definition": None}
        )


def test_v3_configs_explicitly_define_fresh_four_seed_experiments():
    for path, expected_d4 in ((D4_CONFIG, True), (NO_D4_CONFIG, False)):
        config = load_effective_config(path)

        assert config["seed"] == 0
        assert config["training_seeds"] == [0, 1, 2, 3]
        assert config["load_model"] is False
        assert config["checkpoint_path"] is None
        assert type(config["env_kwargs"]["d4_augment"]) is bool
        assert config["env_kwargs"]["d4_augment"] is expected_d4


@pytest.mark.parametrize(
    "invalid_config",
    [
        {
            "run_name": "hybrid_ppo_v3",
            "seed": 0,
            "training_seeds": [0, 1, 2, 3],
            "load_model": False,
            "checkpoint_path": None,
            "env_kwargs": {},
            "experiment_definition": {
                "name": "v3",
                "policy_class": (
                    "twenty_forty_eight_ai.agent.policy.ValueNormalizedMaskablePolicy"
                ),
                "ppo_class": (
                    "twenty_forty_eight_ai.agent.ppo.ValueHeadLRMaskablePPO"
                ),
                "value_head_lr_multiplier": 10.0,
            },
        },
        {
            "run_name": "hybrid_ppo_v3_no_d4",
            "seed": 0,
            "training_seeds": [0, 1, 2, 3],
            "load_model": True,
            "checkpoint_path": "legacy.zip",
            "env_kwargs": {"d4_augment": False},
            "experiment_definition": {
                "name": "v3",
                "policy_class": (
                    "twenty_forty_eight_ai.agent.policy.ValueNormalizedMaskablePolicy"
                ),
                "ppo_class": (
                    "twenty_forty_eight_ai.agent.ppo.ValueHeadLRMaskablePPO"
                ),
                "value_head_lr_multiplier": 10.0,
            },
        },
    ],
)
def test_v3_validation_rejects_implicit_condition_or_legacy_resume(invalid_config):
    from twenty_forty_eight_ai.utils.effective_config import (
        validate_v3_experiment_config,
    )

    with pytest.raises(ValueError):
        validate_v3_experiment_config(invalid_config)


def test_v3_seed_sweep_requires_all_four_configured_seeds():
    from scripts.train import validate_v3_seed_sweep

    config = load_effective_config(D4_CONFIG)
    validate_v3_seed_sweep(config, requested_seed_count=4)

    with pytest.raises(ValueError, match="training_seeds"):
        validate_v3_seed_sweep(config, requested_seed_count=3)


def test_v3_dry_run_validates_definition_before_printing():
    from scripts.train import main_with_sweep

    config = load_effective_config(D4_CONFIG)
    config.pop("experiment_definition")
    config["__sweep"] = {
        "enabled": True,
        "n_seeds": 4,
        "parallel": False,
        "dry_run": True,
        "resume": False,
    }

    with pytest.raises(ValueError, match="experiment_definition"):
        main_with_sweep(config)


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
