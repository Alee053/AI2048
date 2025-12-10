import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from typing import Tuple

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution layer."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int]):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class CustomCNN(BaseFeaturesExtractor):
    """2048-aware CNN with embedding and multi-pathway extraction."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        embedding_dim = 128
        n_tile_types = 17  # 0 for empty, plus 2^1 through 2^16

        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=n_tile_types, embedding_dim=embedding_dim)

        n_input_channels = embedding_dim
        n_pathway_channels = 128

        # Feature Pathways
        self.row_pathway = nn.Sequential(
            DepthwiseSeparableConv(n_input_channels, n_pathway_channels, kernel_size=(1, 4)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.col_pathway = nn.Sequential(
            DepthwiseSeparableConv(n_input_channels, n_pathway_channels, kernel_size=(4, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.grid_pathway = nn.Sequential(
            DepthwiseSeparableConv(n_input_channels, n_pathway_channels, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Feature Combination
        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).long()
            combined_size = self._get_combined_features_size(sample_obs)

        self.linear = nn.Sequential(
            nn.Linear(combined_size, features_dim),
            nn.ReLU()
        )

    def _embed_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Embed integer observations."""
        indices = observations.long()
        # Flatten for embedding
        embedded = self.embedding(indices.squeeze(1).view(-1, 16))
        # Reshape to (N, C, H, W)
        return embedded.permute(0, 2, 1).reshape(-1, self.embedding.embedding_dim, 4, 4)

    def _get_combined_features_size(self, sample_obs: torch.Tensor) -> int:
        """Calculate flattened feature size."""
        embedded_obs = self._embed_observations(sample_obs)
        n_flatten_row = self.row_pathway(embedded_obs).shape[1]
        n_flatten_col = self.col_pathway(embedded_obs).shape[1]
        n_flatten_grid = self.grid_pathway(embedded_obs).shape[1]
        return n_flatten_row + n_flatten_col + n_flatten_grid

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Embed
        embedded = self._embed_observations(observations)

        # Extract features
        row_features = self.row_pathway(embedded)
        col_features = self.col_pathway(embedded)
        grid_features = self.grid_pathway(embedded)

        # Concatenate
        combined_features = torch.cat((row_features, col_features, grid_features), dim=1)

        # Project
        return self.linear(combined_features)