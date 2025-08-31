import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 1028)
        self.layer2 = nn.Linear(1028, 1024)
        self.layer3 = nn.Linear(1024, 1028)
        self.layer4 = nn.Linear(1028, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
