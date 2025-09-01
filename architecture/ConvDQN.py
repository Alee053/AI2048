import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvDQN(nn.Module):
    def __init__(self, num_actions=4):
        super(ConvDQN, self).__init__()
        # Input shape will be (batch_size, 16, 4, 4)
        self.conv1 = nn.Conv2d(16, 128, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def board_to_tensor(board):
    log_board = np.zeros_like(board, dtype=np.float32)

    np.log2(board, out=log_board, where=(board != 0))

    board_int = log_board.astype(np.int64)

    tensor = np.zeros((16, 4, 4), dtype=np.float32)

    for i in range(16):
        tensor[i][board_int == i] = 1

    return tensor