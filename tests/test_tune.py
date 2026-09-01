from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
from optuna.exceptions import TrialPruned

from scripts import tune
from twenty_forty_eight_ai.utils.effective_config import (
    load_effective_config,
    validate_v3_experiment_config,
)


REPO_ROOT = Path(__file__).parents[1]
TUNING_CONFIG = REPO_ROOT / "configs" / "tune" / "bayesian_opt_search.yaml"
D4_CONFIG = REPO_ROOT / "configs" / "train" / "hybrid_ppo_v3.yaml"
NO_D4_CONFIG = REPO_ROOT / "configs" / "train" / "hybrid_ppo_v3_no_d4.yaml"


def _spec(tmp_path: Path) -> tune.StudySpec:
    config = tune.load_tuning_yaml(TUNING_CONFIG)
    config["storage_root"] = str(tmp_path / "tuning")
    config["wandb"]["enabled"] = False
    return tune.build_study_spec(config, config_path=TUNING_CONFIG)


def test_tuning_materialization_uses_v3_and_keeps_official_200m_validation(tmp_path):
    spec = _spec(tmp_path)
    sampled = {
        "learning_rate": 0.0002,
        "gamma": 0.96,
        "gae_lambda": 0.95,
        "ent_coef": 0.00001,
        "vf_coef": 0.6,
        "clip_range": 0.2,
    }

    effective = tune.materialize_tuning_config(
        spec,
        sampled,
        condition="d4",
        output_dir=tmp_path / "d4",
    )

    assert effective["total_timesteps"] == 20_000_000
    assert effective["tuning"]["protocol_status"] == "PRE-FREEZE"
    assert effective["load_model"] is False
    assert effective["checkpoint_path"] is None
    assert effective["experiment_definition"] == {
        "name": "v3",
        "policy_class": (
            "twenty_forty_eight_ai.agent.policy.ValueNormalizedMaskablePolicy"
        ),
        "ppo_class": "twenty_forty_eight_ai.agent.ppo.ValueHeadLRMaskablePPO",
        "value_head_lr_multiplier": 10.0,
    }
    assert effective["tuning"]["fixed"]["value_head_lr_multiplier"] == 10.0
    assert effective["ppo_params"]["learning_rate"] == {
        "type": "linear_decay",
        "initial_value": sampled["learning_rate"],
    }

    with pytest.raises(ValueError, match="total_timesteps"):
        validate_v3_experiment_config(effective)

    official = load_effective_config(D4_CONFIG)
    validate_v3_experiment_config(official)
    assert official["total_timesteps"] == 200_000_000


def test_shared_params_and_condition_diff_are_only_d4_augment(tmp_path):
    spec = _spec(tmp_path)
    sampled = tune.default_sampled_params(spec)
    d4 = tune.materialize_tuning_config(
        spec, sampled, condition="d4", output_dir=tmp_path / "d4"
    )
    no_d4 = tune.materialize_tuning_config(
        spec, sampled, condition="no_d4", output_dir=tmp_path / "no_d4"
    )

    assert d4["ppo_params"] == no_d4["ppo_params"]
    assert d4["seed"] == no_d4["seed"] == 0
    assert d4["root_training_seed"] == no_d4["root_training_seed"] == 0
    assert d4["d4_rank_spawn_keys"] == no_d4["d4_rank_spawn_keys"]
    assert d4["evaluation"]["seeds"] == no_d4["evaluation"]["seeds"]
    assert tune.condition_behavioral_diff(d4, no_d4) == {
        "env_kwargs.d4_augment": (True, False)
    }


def test_search_space_is_exact_and_baselines_are_documented(tmp_path):
    spec = _spec(tmp_path)

    assert tuple(spec.search_space) == (
        "learning_rate",
        "gamma",
        "gae_lambda",
        "ent_coef",
        "vf_coef",
        "clip_range",
    )
    assert all("baseline" in definition for definition in spec.search_space.values())
    assert spec.fixed["n_steps"] == 512
    assert spec.fixed["batch_size"] == 4096
    assert spec.fixed["n_epochs"] == 4
    assert spec.fixed["n_envs"] == 128
    assert spec.fixed["value_head_lr_multiplier"] == 10.0
    assert set(spec.study_config["provenance"]) >= {
        "git_commit",
        "native_extension_sha256",
        "uv_lock_sha256",
    }


def test_objective_is_symmetric_log_space_and_rejects_invalid_values():
    expected = (math.log1p(100.0) + math.log1p(400.0)) / 2.0
    assert tune.symmetric_log_objective(100.0, 400.0) == pytest.approx(expected)
    assert tune.symmetric_log_objective(400.0, 100.0) == pytest.approx(expected)

    with pytest.raises(tune.InvalidTrialError, match="finite non-negative"):
        tune.symmetric_log_objective(float("nan"), 1.0)
    with pytest.raises(tune.InvalidTrialError, match="finite non-negative"):
        tune.symmetric_log_objective(-1.0, 1.0)


def test_metrics_reject_nan_critic_and_search_failures():
    valid = tune.CheckpointMetrics(
        checkpoint_timesteps=5_000_000,
        actual_timesteps=5_000_128,
        depth0_mean_score=10.0,
        depth1_mean_score=20.0,
        depth1_minus_depth0=10.0,
        explained_variance=0.25,
        value_std=2.0,
        value_max_abs=5.0,
        episode_length=30.0,
        training_reward=1.0,
        cap_count=0,
        unresolved_count=0,
        invalid_count=0,
    )
    tune.validate_checkpoint_metrics(valid, critic_max_abs=1_000.0)

    for field in ("explained_variance", "value_std"):
        invalid = valid.__class__(**{**valid.__dict__, field: float("nan")})
        with pytest.raises(tune.InvalidTrialError, match="non-finite"):
            tune.validate_checkpoint_metrics(invalid, critic_max_abs=1_000.0)

    invalid = valid.__class__(**{**valid.__dict__, "cap_count": 1})
    with pytest.raises(tune.InvalidTrialError, match="cap/unresolved/invalid"):
        tune.validate_checkpoint_metrics(invalid, critic_max_abs=1_000.0)


class _FakeTrial:
    number = 3

    def __init__(self, prune: bool):
        self.prune = prune
        self.reports = []
        self.user_attrs = {}

    def report(self, value, step):
        self.reports.append((value, step))

    def should_prune(self):
        return self.prune

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value

    def suggest_float(self, name, low, high, log=False):
        del low, high, log
        return {
            "learning_rate": 0.0002,
            "gamma": 0.96,
            "gae_lambda": 0.95,
            "ent_coef": 0.00001,
            "vf_coef": 0.6,
        }[name]

    def suggest_categorical(self, name, choices):
        del choices
        assert name == "clip_range"
        return 0.2


def test_pruning_is_distinct_from_invalid_trial():
    trial = _FakeTrial(prune=True)
    with pytest.raises(TrialPruned, match="checkpoint"):
        tune.report_and_maybe_prune(trial, step=5_000_000, value=1.5)
    assert trial.user_attrs["prune_reason"]

    with pytest.raises(tune.InvalidTrialError, match="cap"):
        tune.raise_invalid_trial("cap hit at checkpoint")


def test_study_config_hash_is_immutable_and_resume_mismatch_fails(tmp_path):
    first = {"search_space": {"x": 1}, "seed": 0}
    second = {"search_space": {"x": 2}, "seed": 0}
    study_dir = tmp_path / "study"

    digest = tune.ensure_study_config(study_dir, first)
    assert digest == tune.study_config_hash(first)
    assert json.loads((study_dir / "study_config.json").read_text()) == first
    assert (study_dir / "study_config.sha256").read_text().strip() == digest
    assert tune.ensure_study_config(study_dir, first) == digest
    with pytest.raises(tune.StudyConfigurationMismatch, match="incompatible"):
        tune.ensure_study_config(study_dir, second)


def test_trial_artifact_layout_and_manifest_are_pre_freeze(tmp_path):
    paths = tune.trial_artifact_paths(tmp_path, study_name="study", trial_number=7)
    assert paths["trial"] == tmp_path / "study" / "trial_7"
    assert paths["d4"] == paths["trial"] / "d4"
    assert paths["no_d4"] == paths["trial"] / "no_d4"

    manifest = tune.build_tuning_manifest(
        study_name="study",
        trial_number=7,
        condition="d4",
        sampled_params={"gamma": 0.96},
        training_seed=0,
        evaluation_seeds=(10, 11),
        effective_config={"total_timesteps": 20_000_000},
        status="running",
        provenance={"git_commit": "a" * 40, "native_extension_sha256": "b" * 64},
    )
    assert manifest["protocol_status"] == "PRE-FREEZE"
    assert manifest["paper_grade"] is False
    assert manifest["condition"] == "d4"
    assert manifest["evaluation_seeds"] == [10, 11]
    assert manifest["provenance"]["native_extension_sha256"] == "b" * 64


def test_cleanup_closes_sessions_when_trial_fails(tmp_path):
    class FakeSession:
        def __init__(self, condition):
            self.condition = condition
            self.closed = False

        def advance_to(self, _target):
            return 5_000_000

        def checkpoint_path(self, _target):
            return tmp_path / f"{self.condition}.zip"

        def training_metrics(self, target):
            return tune.CheckpointMetrics(
                checkpoint_timesteps=target,
                actual_timesteps=target,
                depth0_mean_score=1.0,
                depth1_mean_score=2.0,
                depth1_minus_depth0=1.0,
                explained_variance=0.1,
                value_std=1.0,
                value_max_abs=1.0,
                episode_length=2.0,
                training_reward=0.0,
                cap_count=0,
                unresolved_count=0,
                invalid_count=0,
            )

        def close(self):
            self.closed = True

    sessions = {}

    def factory(condition, *_args, **_kwargs):
        sessions[condition] = FakeSession(condition)
        return sessions[condition]

    def evaluator(_path, condition, _target, _seeds, _device):
        if condition == "no_d4":
            raise tune.InvalidTrialError("synthetic evaluation failure")
        return {"depth0_mean_score": 1.0, "depth1_mean_score": 2.0}

    with pytest.raises(tune.InvalidTrialError, match="synthetic"):
        tune.execute_trial(
            _FakeTrial(prune=False),
            _spec(tmp_path),
            session_factory=factory,
            evaluation_fn=evaluator,
        )

    assert sessions["d4"].closed is True
    assert sessions["no_d4"].closed is True


def test_execute_trial_uses_one_param_set_and_prunes_only_after_both_conditions(tmp_path):
    spec = _spec(tmp_path)
    received_params = []
    evaluation_calls = []
    sessions = {}

    class FakeSession:
        def __init__(self, condition):
            self.condition = condition
            self.current = 0

        def advance_to(self, target):
            self.current = target
            return target

        def checkpoint_path(self, target):
            return tmp_path / f"{self.condition}-{target}.zip"

        def training_metrics(self, target):
            return {
                "actual_timesteps": target,
                "explained_variance": 0.2,
                "value_std": 1.0,
                "value_max_abs": 2.0,
                "episode_length": 10.0,
                "training_reward": 3.0,
                "cap_count": 0,
                "unresolved_count": 0,
                "invalid_count": 0,
            }

        def close(self):
            sessions[self.condition] = self

    def factory(condition, _spec, params, _directory, _trial_number):
        received_params.append(params)
        return FakeSession(condition)

    def evaluator(_path, condition, target, seeds, _device):
        evaluation_calls.append((condition, target, tuple(seeds)))
        return {
            "depth0_mean_score": 10.0,
            "depth1_mean_score": 100.0 if condition == "d4" else 80.0,
            "episode_length": 20.0,
        }

    trial = _FakeTrial(prune=False)
    result = tune.execute_trial(
        trial,
        spec,
        session_factory=factory,
        evaluation_fn=evaluator,
    )

    assert received_params[0] is received_params[1]
    assert len(evaluation_calls) == 6
    assert [step for _value, step in trial.reports] == list(spec.checkpoints)
    expected = tune.symmetric_log_objective(100.0, 80.0)
    assert result == pytest.approx(expected)
    assert all(call[2] == spec.evaluation_seeds for call in evaluation_calls)
    trial_dir = tmp_path / "tuning" / spec.study_name / "trial_3"
    assert json.loads((trial_dir / "d4" / "metrics.json").read_text())["checkpoints"]
    trial_manifest = json.loads((trial_dir / "trial_manifest.json").read_text())
    assert trial_manifest["status"] == "completed"
    assert trial_manifest["paper_grade"] is False


def test_dry_run_reports_cost_without_constructing_a_study(tmp_path):
    spec = _spec(tmp_path)
    dry_trial = tune.dry_run_trial(spec)
    assert dry_trial["d4"]["tuning"]["condition"] == "d4"
    assert dry_trial["no_d4"]["tuning"]["condition"] == "no_d4"
    assert tune.condition_behavioral_diff(dry_trial["d4"], dry_trial["no_d4"]) == {
        "env_kwargs.d4_augment": (True, False)
    }

    output = tune.render_dry_run(spec)
    assert "12 trials" in output
    assert "24 full training runs" in output
    assert "20,000,000" in output
    assert "PRE-FREEZE" in output


def test_sampler_and_pruner_configuration_are_reproducible(tmp_path):
    spec = _spec(tmp_path)
    sampler = tune.make_sampler(spec)
    pruner = tune.make_pruner(spec)

    assert sampler.__class__.__name__ == "TPESampler"
    assert sampler._rng is not None
    first_draws = sampler._rng.rng.random(8)
    second_draws = tune.make_sampler(spec)._rng.rng.random(8)
    assert np.array_equal(first_draws, second_draws)
    assert pruner.__class__.__name__ == "MedianPruner"
    assert pruner._n_startup_trials == 5
    assert pruner._n_warmup_steps == 5_000_000
    assert pruner._interval_steps == 5_000_000


def test_clamped_schedule_uses_global_timesteps_and_never_goes_negative():
    schedule = tune.make_clamped_linear_schedule(0.001, 100)
    assert schedule(1.0) == pytest.approx(0.001)
    schedule.current_timestep = 25
    assert schedule(0.5) == pytest.approx(0.00075)
    schedule.current_timestep = 150
    assert schedule(0.0) == 0.0


def test_rng_state_isolated_and_restorable_for_each_condition():
    previous_state = tune._capture_rng_state()
    random.seed(17)
    np.random.seed(17)
    import torch

    try:
        torch.manual_seed(17)
        state = tune._capture_rng_state()
        first = (random.random(), float(np.random.random()), float(torch.rand(1)))
        tune._restore_rng_state(state)
        second = (random.random(), float(np.random.random()), float(torch.rand(1)))
        assert first == second
    finally:
        tune._restore_rng_state(previous_state)


def test_v3_model_builder_uses_required_classes_without_training():
    import numpy as np
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from scripts.train import make_training_env_factories

    config = load_effective_config(D4_CONFIG)
    config["total_timesteps"] = 16
    config["n_envs"] = 1
    schedule = tune.make_clamped_linear_schedule(0.0002, 16)
    config["ppo_params"] = {
        "n_steps": 8,
        "gamma": 0.95,
        "gae_lambda": 0.95,
        "ent_coef": 0.00001,
        "vf_coef": 0.5,
        "learning_rate": schedule,
        "clip_range": 0.2,
        "batch_size": 8,
        "n_epochs": 1,
    }
    env = DummyVecEnv(
        make_training_env_factories(
            {"d4_augment": True},
            np.random.SeedSequence(0).spawn(1),
        )
    )
    try:
        model = tune.build_v3_tuning_model(config, env, seed=0)
        assert type(model).__name__ == "ValueHeadLRMaskablePPO"
        assert type(model.policy).__name__ == "ValueNormalizedMaskablePolicy"
        assert model.value_head_lr_multiplier == 10.0
        import torch

        assert any(
            isinstance(layer, torch.nn.LayerNorm)
            for layer in model.policy.mlp_extractor.value_net
        )
        assert model.lr_schedule(0.5) == schedule(0.5)
    finally:
        env.close()
