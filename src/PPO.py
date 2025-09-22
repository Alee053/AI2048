# PPO.py

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        embedding_dim = 128
        self.embedding = nn.Embedding(num_embeddings=17, embedding_dim=embedding_dim)

        n_input_channels = embedding_dim

        def create_conv_pathway(kernel_size, out_channels=128):
            return nn.Sequential(
                nn.Conv2d(n_input_channels, n_input_channels, kernel_size=kernel_size, groups=n_input_channels),
                nn.Conv2d(n_input_channels, out_channels, kernel_size=1),
                nn.ReLU(),
                nn.Flatten(),
            )

        self.row_pathway = create_conv_pathway(kernel_size=(1, 4))
        self.col_pathway = create_conv_pathway(kernel_size=(4, 1))
        self.grid_pathway = create_conv_pathway(kernel_size=(2, 2))

        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).long()

            embedded_obs = self.embedding(sample_obs.squeeze(1).view(-1, 16))
            embedded_obs = embedded_obs.permute(0, 2, 1).reshape(-1, embedding_dim, 4, 4)

            n_flatten_row = self.row_pathway(embedded_obs).shape[1]
            n_flatten_col = self.col_pathway(embedded_obs).shape[1]
            n_flatten_grid = self.grid_pathway(embedded_obs).shape[1]
            combined_size = n_flatten_row + n_flatten_col + n_flatten_grid

        self.linear = nn.Sequential(
            nn.Linear(combined_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        indices = observations.long()

        embedded = self.embedding(indices.squeeze(1).view(-1, 16))
        embedded = embedded.permute(0, 2, 1).reshape(-1, self.embedding.embedding_dim, 4, 4)

        row_features = self.row_pathway(embedded)
        col_features = self.col_pathway(embedded)
        grid_features = self.grid_pathway(embedded)

        combined_features = torch.cat((row_features, col_features, grid_features), dim=1)
        return self.linear(combined_features)