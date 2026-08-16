from pathlib import Path

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from twenty_forty_eight_ai.agent.architecture import CustomCNN
from twenty_forty_eight_ai.agent.policy import ValueNormalizedMaskablePolicy
from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.utils.tensor_utils import board_to_tensor


CONFIG_DIR = Path(__file__).parents[1] / "configs" / "train"
FEATURES_DIM = 256


def _make_model(policy_class, d4_augment=False, seed=123):
    env = DummyVecEnv([lambda: Game2048Env(d4_augment=d4_augment)])
    return MaskablePPO(
        policy_class,
        env,
        policy_kwargs={
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {"features_dim": FEATURES_DIM},
        },
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        seed=seed,
        device="cpu",
        verbose=0,
    )


def _close_model(model):
    if model.get_env() is not None:
        model.get_env().close()


def _architecture_signature(model):
    policy = model.policy
    return {
        "policy_net": tuple(type(module) for module in policy.mlp_extractor.policy_net),
        "value_net": tuple(type(module) for module in policy.mlp_extractor.value_net),
        "policy_latent_dim": policy.mlp_extractor.latent_dim_pi,
        "value_latent_dim": policy.mlp_extractor.latent_dim_vf,
        "share_features_extractor": policy.share_features_extractor,
    }


def _value_tanh_outputs(model, boards):
    policy = model.policy
    observations = torch.as_tensor(board_to_tensor(boards))
    with torch.no_grad():
        x = policy.extract_features(observations)
        outputs = []
        for module in policy.mlp_extractor.value_net:
            x = module(x)
            if isinstance(module, nn.Tanh):
                outputs.append(x)
    return outputs


def _synthetic_boards():
    return np.asarray(
        [
            np.zeros((4, 4), dtype=np.int64),
            [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [0, 1, 2, 3], [4, 5, 6, 7]],
        ],
        dtype=np.int64,
    )


def test_value_policy_has_exact_normalized_value_architecture():
    model = _make_model(ValueNormalizedMaskablePolicy)
    try:
        policy = model.policy
        assert isinstance(policy, ValueNormalizedMaskablePolicy)
        assert _architecture_signature(model) == {
            "policy_net": (nn.Linear, nn.Tanh, nn.Linear, nn.Tanh),
            "value_net": (
                nn.Linear,
                nn.LayerNorm,
                nn.Tanh,
                nn.Linear,
                nn.LayerNorm,
                nn.Tanh,
            ),
            "policy_latent_dim": 64,
            "value_latent_dim": 64,
            "share_features_extractor": True,
        }

        value_net = policy.mlp_extractor.value_net
        assert value_net[0].in_features == FEATURES_DIM
        assert value_net[0].out_features == 64
        assert value_net[3].in_features == 64
        assert value_net[3].out_features == 64
        for layer_norm in (value_net[1], value_net[4]):
            assert tuple(layer_norm.normalized_shape) == (64,)
            assert layer_norm.elementwise_affine is False
    finally:
        _close_model(model)


def test_policy_mlp_and_action_head_match_default_policy():
    baseline = _make_model(MaskableActorCriticPolicy, seed=321)
    normalized = _make_model(ValueNormalizedMaskablePolicy, seed=321)
    try:
        assert tuple(type(module) for module in normalized.policy.mlp_extractor.policy_net) == (
            nn.Linear,
            nn.Tanh,
            nn.Linear,
            nn.Tanh,
        )
        assert not any(
            isinstance(module, nn.LayerNorm)
            for module in normalized.policy.mlp_extractor.policy_net
        )
        assert type(normalized.policy.action_net) is type(baseline.policy.action_net)
        assert normalized.policy.action_net.in_features == baseline.policy.action_net.in_features
        assert normalized.policy.action_net.out_features == baseline.policy.action_net.out_features

        for name, parameter in baseline.policy.mlp_extractor.policy_net.state_dict().items():
            torch.testing.assert_close(
                parameter,
                normalized.policy.mlp_extractor.policy_net.state_dict()[name],
            )
        for name, parameter in baseline.policy.action_net.state_dict().items():
            torch.testing.assert_close(
                parameter,
                normalized.policy.action_net.state_dict()[name],
            )
    finally:
        _close_model(baseline)
        _close_model(normalized)


def test_fresh_value_forward_is_finite_and_state_dependent():
    model = _make_model(ValueNormalizedMaskablePolicy)
    try:
        observations = torch.as_tensor(board_to_tensor(_synthetic_boards()))
        with torch.no_grad():
            values = model.policy.predict_values(observations).flatten()

        assert torch.isfinite(values).all()
        assert float(values.std(unbiased=False)) > 1e-4
    finally:
        _close_model(model)


def test_fresh_value_tanh_outputs_are_not_significantly_saturated():
    model = _make_model(ValueNormalizedMaskablePolicy)
    try:
        boards = np.random.default_rng(2026).integers(0, 8, size=(128, 4, 4))
        for output in _value_tanh_outputs(model, boards):
            saturation_pct = float((output.abs() > 0.99).float().mean())
            assert saturation_pct < 0.05
    finally:
        _close_model(model)


def test_save_load_preserves_normalized_value_architecture(tmp_path):
    model = _make_model(ValueNormalizedMaskablePolicy)
    path = tmp_path / "normalized_value_policy"
    try:
        boards = _synthetic_boards()
        observations = torch.as_tensor(board_to_tensor(boards))
        with torch.no_grad():
            expected_values = model.policy.predict_values(observations).flatten()
        expected_signature = _architecture_signature(model)
        model.save(path)
    finally:
        _close_model(model)

    loaded = MaskablePPO.load(path, device="cpu")
    try:
        assert isinstance(loaded.policy, ValueNormalizedMaskablePolicy)
        assert _architecture_signature(loaded) == expected_signature
        with torch.no_grad():
            actual_values = loaded.policy.predict_values(observations).flatten()
        torch.testing.assert_close(actual_values, expected_values)
    finally:
        _close_model(loaded)


def test_training_uses_normalized_policy_only_for_v3():
    from sb3_contrib import MaskablePPO
    from scripts.train import select_training_policy, select_training_ppo
    from twenty_forty_eight_ai.utils.effective_config import load_effective_config

    v3_config = load_effective_config(CONFIG_DIR / "hybrid_ppo_v3.yaml")
    assert select_training_policy(v3_config) is ValueNormalizedMaskablePolicy
    assert (
        select_training_policy(
            load_effective_config(CONFIG_DIR / "hybrid_ppo_v3_no_d4.yaml")
        )
        is ValueNormalizedMaskablePolicy
    )
    assert select_training_policy({"run_name": "hybrid_ppo_v3"}) == "CnnPolicy"
    assert select_training_ppo({"run_name": "hybrid_ppo_v3"}) is MaskablePPO
    assert select_training_policy({"run_name": "hybrid_ppo_v1"}) == "CnnPolicy"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "config_name", ("hybrid_ppo_v3.yaml", "hybrid_ppo_v3_no_d4.yaml")
)
def test_seed_sweep_resolves_actual_v3_components(config_name, seed):
    from scripts.train import (
        resolve_training_config,
        select_training_policy,
        select_training_ppo,
        select_value_head_lr_multiplier,
    )
    from twenty_forty_eight_ai.agent.policy import ValueNormalizedMaskablePolicy
    from twenty_forty_eight_ai.agent.ppo import ValueHeadLRMaskablePPO
    from twenty_forty_eight_ai.utils.effective_config import load_effective_config

    config = load_effective_config(CONFIG_DIR / config_name)
    config["seed"] = seed
    config["run_name"] = f"{config['run_name']}-seed{seed}"
    effective_config, _ = resolve_training_config(config)

    assert select_training_policy(effective_config) is ValueNormalizedMaskablePolicy
    assert select_training_ppo(effective_config) is ValueHeadLRMaskablePPO
    assert select_value_head_lr_multiplier(effective_config) == 10.0


def test_fresh_model_construction_uses_resolved_v3_components(monkeypatch):
    from scripts import train as train_module
    from twenty_forty_eight_ai.utils.effective_config import load_effective_config

    captured = {}

    class FakePPO:
        def __init__(self, policy, env, **kwargs):
            captured["policy"] = policy
            captured["env"] = env
            captured.update(kwargs)

    class FakePolicy:
        pass

    monkeypatch.setattr(train_module, "ValueHeadLRMaskablePPO", FakePPO)
    monkeypatch.setattr(train_module, "ValueNormalizedMaskablePolicy", FakePolicy)
    effective_config = load_effective_config(CONFIG_DIR / "hybrid_ppo_v3.yaml")
    effective_config["run_name"] = "hybrid_ppo_v3-seed0"

    train_module.build_fresh_model(
        effective_config,
        vec_env="vec-env",
        policy_kwargs={"features_dim": 256},
        ppo_params={},
        seed=0,
    )

    assert captured["policy"] is FakePolicy
    assert captured["env"] == "vec-env"
    assert captured["value_head_lr_multiplier"] == 10.0


@pytest.mark.parametrize("field", ["policy_class", "ppo_class"])
def test_v3_component_resolver_rejects_unregistered_class_path(field):
    from scripts.train import select_training_policy, select_training_ppo
    from twenty_forty_eight_ai.utils.effective_config import load_effective_config

    config = load_effective_config(CONFIG_DIR / "hybrid_ppo_v3.yaml")
    config["experiment_definition"][field] = "not.a.registered.class"

    resolver = select_training_policy if field == "policy_class" else select_training_ppo
    with pytest.raises(ValueError, match="Unsupported v3"):
        resolver(config)


@pytest.mark.parametrize(
    "config_name", ("hybrid_ppo_v3.yaml", "hybrid_ppo_v3_no_d4.yaml")
)
def test_real_seed_sweep_path_constructs_v3_components_for_all_seeds(
    config_name, tmp_path, monkeypatch
):
    from scripts import train as train_module
    from twenty_forty_eight_ai.utils.effective_config import load_effective_config

    constructed = []

    class FakePPO:
        def __init__(self, policy, _env, **kwargs):
            constructed.append((policy, kwargs["value_head_lr_multiplier"]))

    class FakePolicy:
        pass

    monkeypatch.setattr(train_module, "ValueHeadLRMaskablePPO", FakePPO)
    monkeypatch.setattr(train_module, "ValueNormalizedMaskablePolicy", FakePolicy)
    monkeypatch.setattr(train_module.wandb, "finish", lambda: None)

    config = load_effective_config(CONFIG_DIR / config_name)
    config["output_dir"] = str(tmp_path)
    config["__sweep"] = {
        "enabled": True,
        "n_seeds": 4,
        "resume": False,
        "parallel": False,
        "dry_run": False,
    }

    def fake_train(seed_config):
        effective_config, _ = train_module.resolve_training_config(seed_config)
        train_module.build_fresh_model(
            effective_config,
            vec_env="vec-env",
            policy_kwargs={},
            ppo_params={},
            seed=seed_config["seed"],
        )

    monkeypatch.setattr(train_module, "train", fake_train)
    train_module.main_with_sweep(config)

    assert len(constructed) == 4
    assert all(policy is FakePolicy for policy, _ in constructed)
    assert all(multiplier == 10.0 for _, multiplier in constructed)


@pytest.mark.parametrize(
    ("config_name", "expected_d4"),
    (("hybrid_ppo_v3.yaml", True), ("hybrid_ppo_v3_no_d4.yaml", False)),
)
def test_d4_conditions_materialize_identical_policy_architecture(config_name, expected_d4):
    from twenty_forty_eight_ai.utils.effective_config import load_effective_config

    config = load_effective_config(CONFIG_DIR / config_name)
    assert config["env_kwargs"]["d4_augment"] is expected_d4

    model = _make_model(
        ValueNormalizedMaskablePolicy,
        d4_augment=config["env_kwargs"]["d4_augment"],
    )
    try:
        assert _architecture_signature(model) == {
            "policy_net": (nn.Linear, nn.Tanh, nn.Linear, nn.Tanh),
            "value_net": (
                nn.Linear,
                nn.LayerNorm,
                nn.Tanh,
                nn.Linear,
                nn.LayerNorm,
                nn.Tanh,
            ),
            "policy_latent_dim": 64,
            "value_latent_dim": 64,
            "share_features_extractor": True,
        }
    finally:
        _close_model(model)
