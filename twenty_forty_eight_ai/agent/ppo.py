"""PPO algorithm variants used by the training experiments."""

from __future__ import annotations

import math
from numbers import Real

from sb3_contrib import MaskablePPO


def validate_value_head_lr_multiplier(multiplier: Real) -> float:
    """Validate and normalize the scalar value-head learning-rate multiplier."""
    if isinstance(multiplier, bool) or not isinstance(multiplier, Real):
        raise ValueError("value_head_lr_multiplier must be a positive finite number")
    multiplier = float(multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("value_head_lr_multiplier must be a positive finite number")
    return multiplier


class ValueHeadLRMaskablePPO(MaskablePPO):
    """MaskablePPO with a separate decay-scaled LR for ``policy.value_net``."""

    def __init__(self, *args, value_head_lr_multiplier: float = 1.0, **kwargs):
        self.value_head_lr_multiplier = validate_value_head_lr_multiplier(
            value_head_lr_multiplier
        )
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        self._split_value_head_optimizer_group()

    def _split_value_head_optimizer_group(self) -> None:
        """Keep all parameters normal and isolate only the scalar value head."""
        head_parameters = list(self.policy.value_net.parameters())
        head_ids = {id(parameter) for parameter in head_parameters}
        optimizer = self.policy.optimizer

        if any(group.get("value_head", False) for group in optimizer.param_groups):
            return

        for group in optimizer.param_groups:
            group["params"] = [
                parameter
                for parameter in group["params"]
                if id(parameter) not in head_ids
            ]
            group["value_head"] = False

        initial_lr = float(self.lr_schedule(1.0))
        optimizer.add_param_group(
            {
                "params": head_parameters,
                "lr": initial_lr * self.value_head_lr_multiplier,
                "value_head": True,
            }
        )

    def _update_learning_rate(self, optimizers) -> None:
        """Apply the normal decay and preserve the value-head multiplier."""
        base_lr = float(self.lr_schedule(self._current_progress_remaining))
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        for optimizer in optimizers:
            for group in optimizer.param_groups:
                multiplier = (
                    self.value_head_lr_multiplier
                    if group.get("value_head", False)
                    else 1.0
                )
                group["lr"] = base_lr * multiplier

        if hasattr(self, "_logger"):
            self.logger.record("train/learning_rate", base_lr)
            self.logger.record(
                "train/value_head_learning_rate",
                base_lr * self.value_head_lr_multiplier,
            )


def load_ppo_model(path, device="auto"):
    """Load v3 optimizer-group models while retaining legacy model support."""
    try:
        return ValueHeadLRMaskablePPO.load(path, device=device)
    except (FileNotFoundError, ModuleNotFoundError, ValueError):
        return MaskablePPO.load(path, device=device)
