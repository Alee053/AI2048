import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from typing import Tuple

class DepthwiseSeparableConv(nn.Module):
    """A module for a single depthwise separable convolution layer.

    This type of convolution is more parameter-efficient than a standard
    convolution by splitting the operation into a depthwise convolution
    (which acts on each input channel separately) and a pointwise convolution
    (a 1x1 convolution that combines the outputs).

    Attributes:
        depthwise (nn.Conv2d): The depthwise convolution layer.
        pointwise (nn.Conv2d): The pointwise convolution layer.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int]):
        """Initializes the DepthwiseSeparableConv module.

        Args:
            in_channels (int): The number of channels in the input image.
            out_channels (int): The number of channels produced by the convolution.
            kernel_size (Tuple[int, int]): The size of the convolving kernel.
        """
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
        """Defines the forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying depthwise and
            pointwise convolutions.
        """
        return self.pointwise(self.depthwise(x))


class CustomCNN(BaseFeaturesExtractor):
    """A custom CNN feature extractor for the 2048 game.

    This architecture is designed to be "2048-Aware" by using three key concepts:
    1.  An Embedding layer to create a rich vector representation for each tile value.
    2.  Depthwise Separable Convolutions for efficient feature extraction.
    3.  Multiple parallel pathways to analyze rows, columns, and local 2x2 grid
        patterns simultaneously.

    The features from these three pathways are then concatenated and passed through
    a final linear layer to produce the ultimate feature representation.

    Attributes:
        embedding (nn.Embedding): Layer to embed tile values.
        row_pathway (nn.Sequential): Pathway to extract features from rows.
        col_pathway (nn.Sequential): Pathway to extract features from columns.
        grid_pathway (nn.Sequential): Pathway to extract features from 2x2 grids.
        linear (nn.Sequential): Final linear layer to combine pathway features.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """Initializes the CustomCNN feature extractor.

        Args:
            observation_space (gym.spaces.Box): The observation space of the environment.
            features_dim (int): The number of features to extract.
        """
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
        """Helper function to perform the embedding and reshaping step.

        Args:
            observations (torch.Tensor): The input observations from the
                environment, of shape (N, 1, 4, 4).

        Returns:
            torch.Tensor: The embedded and reshaped tensor, ready for the
            convolutional pathways, of shape (N, embedding_dim, 4, 4).
        """
        # Input observations are (N, 1, 4, 4) with integer log2 values.
        indices = observations.long()
        # Reshape to (N, 16) to pass to the embedding layer
        embedded = self.embedding(indices.squeeze(1).view(-1, 16))
        # Reshape back to a grid format (N, embedding_dim, 4, 4) for the CNN
        return embedded.permute(0, 2, 1).reshape(-1, self.embedding.embedding_dim, 4, 4)

    def _get_combined_features_size(self, sample_obs: torch.Tensor) -> int:
        """Performs a dummy forward pass to calculate the flattened feature size.

        This is necessary to determine the input size of the final linear layer
        dynamically based on the output of the convolutional pathways.

        Args:
            sample_obs (torch.Tensor): A sample observation tensor.

        Returns:
            int: The total number of flattened features from all pathways.
        """
        embedded_obs = self._embed_observations(sample_obs)
        n_flatten_row = self.row_pathway(embedded_obs).shape[1]
        n_flatten_col = self.col_pathway(embedded_obs).shape[1]
        n_flatten_grid = self.grid_pathway(embedded_obs).shape[1]
        return n_flatten_row + n_flatten_col + n_flatten_grid

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the feature extractor.

        Args:
            observations (torch.Tensor): The input observations from the
                environment.

        Returns:
            torch.Tensor: The extracted features tensor.
        """
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