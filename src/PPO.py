# PPO.py

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # --- Pathway 1: Row Convolutions ---
        # Kernel size (1, 4) "sees" one full row at a time.
        self.row_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=(1, 4), stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.col_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=(4, 1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.grid_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten_row = self.row_cnn(sample_obs).shape[1]
            n_flatten_col = self.col_cnn(sample_obs).shape[1]
            n_flatten_grid = self.grid_cnn(sample_obs).shape[1]
            combined_flatten_size = n_flatten_row + n_flatten_col + n_flatten_grid

        self.linear = nn.Sequential(
            nn.Linear(combined_flatten_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        row_features = self.row_cnn(observations)
        col_features = self.col_cnn(observations)
        grid_features = self.grid_cnn(observations)

        combined_features = torch.cat((row_features, col_features, grid_features), dim=1)

        return self.linear(combined_features)