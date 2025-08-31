import math
import random
from collections import namedtuple
from itertools import count
from os import mkdir

import torch
from torch import nn
from torch import optim
from tqdm import tqdm

import wandb
from Game2048Env import Game2048Env
from QNetwork import ConvDQN
from ReplayMemory import ReplayMemory

# CONFIG GLOBALS
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 1024
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000000
TAU = 0.005
LR = 1e-4
UPDATE_FREQUENCY = 4

DEVICE = "cuda"

MEMORY_SIZE = 100000

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Trainer:
    def __init__(self, name):
        self.name = name
        self.env = Game2048Env()
        self.env.reset()

        wandb.init(
            project="2048-rl",  # Name of the project where runs are stored
            name=name,  # Name of this specific run
            config={  # Log all your hyperparameters
                "learning_rate": LR,
                "gamma": GAMMA,
                "eps_start": EPS_START,
                "eps_end": EPS_END,
                "eps_decay": EPS_DECAY,
                "batch_size": BATCH_SIZE,
                "memory_size": MEMORY_SIZE,
                "tau": TAU,
                "update_frequency": UPDATE_FREQUENCY,
            }
        )

        self.policy_net = ConvDQN(self.env.action_space.n).to(DEVICE)
        self.target_net = ConvDQN(self.env.action_space.n).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=DEVICE, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
        with torch.no_grad():
            next_actions = self.policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states).gather(1, next_actions).squeeze(1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train_model(self, num_episodes, save_interval=1000):
        self.prefill_memory(10000)

        for i_episode in tqdm(range(num_episodes), desc="Training Episodes"):
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32,
                                 device=DEVICE).unsqueeze(0)

            total_episode_reward = 0

            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=DEVICE)
                done = terminated

                total_episode_reward += reward.item()

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                if self.steps_done % UPDATE_FREQUENCY == 0:
                    # Perform one step of the optimization (on the policy network)
                    self.optimize_model()

                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * \
                                                     TAU + target_net_state_dict[key] * (1 - TAU)
                    self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    wandb.log({
                        "reward": total_episode_reward,
                        "max_tile": 2 ** self.env.game.max_tile,
                        "episode_length": t + 1,
                        "epsilon": EPS_END + (EPS_START - EPS_END) * \
                                   math.exp(-1. * self.steps_done / EPS_DECAY)
                    })
                    if i_episode % save_interval == 0 and i_episode > 0:
                        self.save_model()
                    break

        print('Complete')

        self.save_model()
        wandb.finish()

    def prefill_memory(self, prefill_steps):
        print(f"Prefilling replay buffer with {prefill_steps} random steps...")
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        for _ in tqdm(range(prefill_steps)):
            action = torch.tensor([[self.env.action_space.sample()]], device=DEVICE, dtype=torch.long)
            observation, reward, terminated, _ = self.env.step(action.item())
            reward = torch.tensor([reward], device=DEVICE)

            if terminated:
                observation, info = self.env.reset()  # Reset env if done

            next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            self.memory.push(state, action, next_state, reward)
            state = next_state

    def debug(self, total_episode_reward, t):
        print("Max Tile:", 2 ** self.env.game.max_tile)
        print("Reward:", total_episode_reward)
        print("Steps:", t)
        self.env.render()
        print("--------------------")

    def save_model(self):
        try:
            mkdir("models/" + self.name)
        except:
            pass

        directory = "models/" + self.name + "/"
        torch.save(self.policy_net.state_dict(), directory + "model.pth", )
        torch.save(self.target_net.state_dict(), directory + "target_model.pth")
        print("Model has been saved")

    def load_model(self, name):
        directory = "models/" + name + "/"
        self.policy_net.load_state_dict(torch.load(directory + "model.pth", weights_only=True))
        self.target_net.load_state_dict(torch.load(directory + "target_model.pth", weights_only=True))

    def test_model(self):
        self.policy_net.eval()
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32,
                             device=DEVICE).unsqueeze(0)
        for _ in count():
            action = self.policy_net(state).max(1).indices.view(1, 1)
            observation, reward, terminated, _ = self.env.step(action.item())
            state = torch.tensor(observation, dtype=torch.float32,
                                 device=DEVICE).unsqueeze(0)
            self.env.render()
            if terminated:
                break
