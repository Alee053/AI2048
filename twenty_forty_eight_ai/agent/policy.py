"""Custom MaskablePPO policy with a normalized critic branch."""

from __future__ import annotations

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from torch import nn


class ValueNormalizedMlpExtractor(MlpExtractor):
    """Keep SB3's actor MLP and normalize only critic pre-activations."""

    def __init__(self, feature_dim, net_arch, activation_fn, device):
        super().__init__(feature_dim, net_arch, activation_fn, device)

        normalized_value_layers = []
        for layer in self.value_net:
            normalized_value_layers.append(layer)
            if isinstance(layer, nn.Linear):
                normalized_value_layers.append(
                    nn.LayerNorm(layer.out_features, elementwise_affine=False)
                )
        self.value_net = nn.Sequential(*normalized_value_layers)


class ValueNormalizedMaskablePolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy with an isolated, normalized value MLP."""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ValueNormalizedMlpExtractor(
            self.features_dim,
            self.net_arch,
            self.activation_fn,
            self.device,
        )
