import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from typing import Tuple

class DepthwiseSeparableConv(nn.Module):
    """
    A module for a single depthwise separable convolution layer.
    It's more parameter-efficient than a standard convolution.
    """
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
    """
    A "2048-Aware" CNN.

    This architecture uses three key concepts:
    1.  An Embedding layer to create a rich vector representation for each tile.
    2.  Depthwise Separable Convolutions for efficient feature extraction.
    3.  Multiple pathways to analyze rows, columns, and local grid patterns simultaneously.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        embedding_dim = 128
        n_tile_types = 17  # 0 for empty, plus 2^1 through 2^16

        # 1. The Embedding Layer: Maps each tile's integer value to a dense vector.
        self.embedding = nn.Embedding(num_embeddings=n_tile_types, embedding_dim=embedding_dim)

        n_input_channels = embedding_dim
        n_pathway_channels = 128

        # 2. The Three Pathways: Analyze the board in different ways.
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

        # 3. The Combiner: A final linear layer to merge features from all pathways.
        #    We compute the input size for this layer with a dummy forward pass.
        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).long()
            combined_size = self._get_combined_features_size(sample_obs)

        self.linear = nn.Sequential(
            nn.Linear(combined_size, features_dim),
            nn.ReLU()
        )

    def _embed_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Helper function to perform the embedding step."""
        # Input observations are (N, 1, 4, 4) with integer log2 values.
        indices = observations.long()
        # Reshape to (N, 16) to pass to the embedding layer
        embedded = self.embedding(indices.squeeze(1).view(-1, 16))
        # Reshape back to a grid format (N, embedding_dim, 4, 4) for the CNN
        return embedded.permute(0, 2, 1).reshape(-1, self.embedding.embedding_dim, 4, 4)

    def _get_combined_features_size(self, sample_obs: torch.Tensor) -> int:
        """Performs a dummy forward pass to calculate the flattened feature size."""
        embedded_obs = self._embed_observations(sample_obs)
        n_flatten_row = self.row_pathway(embedded_obs).shape[1]
        n_flatten_col = self.col_pathway(embedded_obs).shape[1]
        n_flatten_grid = self.grid_pathway(embedded_obs).shape[1]
        return n_flatten_row + n_flatten_col + n_flatten_grid

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """The forward pass of the network."""
        # Step 1: Convert tile indices to rich vectors
        embedded = self._embed_observations(observations)

        # Step 2: Process the embedded board through each parallel pathway
        row_features = self.row_pathway(embedded)
        col_features = self.col_pathway(embedded)
        grid_features = self.grid_pathway(embedded)

        # Step 3: Concatenate the features from all pathways
        combined_features = torch.cat((row_features, col_features, grid_features), dim=1)

        # Step 4: Pass through the final linear layer to get the final feature vector
        return self.linear(combined_features)