from itertools import count
from multiprocessing.util import debug
from os import mkdir
from os.path import split
from QNetwork import DQN
from ReplayMemory import ReplayMemory
from torch import optim
from Game2048Env import Game2048Env
import random
import torch
import math
from collections import namedtuple
from torch import nn
from matplotlib import pyplot as plt

# CONFIG GLOBALS
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

DEVICE="cuda"

MEMORY_SIZE=100000

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Model:
    def __init__(self,name):
        self.name=name
        self.env=Game2048Env()
        state, info = self.env.reset()
        self.policy_net = DQN( len(state),self.env.action_space.n ).to(DEVICE)
        self.target_net = DQN( len(state),self.env.action_space.n).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self,state):
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
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states).max(1).values
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

    def train_model(self, num_episodes,save_interval=1000):
        episode_steps = []
        episode_reward = []
        episode_max_tile = []

        plt.ion()

        for i_episode in range(num_episodes):
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
                    self.debug(i_episode,total_episode_reward,t)
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * \
                                                 TAU + target_net_state_dict[key] * (1 - TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_max_tile.append(2 ** self.env.game.max_tile)
                    episode_reward.append(total_episode_reward)
                    episode_steps.append(t)
                    self.plot_info(show_result=False, info=(episode_max_tile, episode_reward, episode_steps))
                    break
            if i_episode%save_interval==0 and i_episode>0:
                self.save_model()
        print('Complete')
        self.plot_info(show_result=True, info=(episode_max_tile, episode_reward, episode_steps))
        plt.ioff()
        plt.show()

        self.save_model()

    def debug(self,i_episode,total_episode_reward,t):
        print("--------------------")
        print("Episode:", i_episode)
        """print("Max Tile:", 2 ** self.env.game.max_tile)
        print("Reward:", total_episode_reward)
        print("Steps:", t)"""
        self.env.render()


    def save_model(self):
        try:
            mkdir("models/" + self.name)
        except:
            pass

        dir="models/"+self.name+"/"
        torch.save(self.policy_net.state_dict(), dir+"model.pth", )
        torch.save(self.target_net.state_dict(), dir+"target_model.pth")
        print("Model has been saved")

    def load_model(self,name):
        dir="models/"+name+"/"
        self.policy_net.load_state_dict(torch.load(dir+"model.pth",weights_only=True))
        self.target_net.load_state_dict(torch.load(dir+"target_model.pth",weights_only=True))

    def test_model(self):
        self.policy_net.eval()
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32,
                             device=DEVICE).unsqueeze(0)
        for t in count():
            action = self.policy_net(state).max(1).indices.view(1, 1)
            observation, reward, terminated, _ = self.env.step(action.item())
            state = torch.tensor(observation, dtype=torch.float32,
                                 device=DEVICE).unsqueeze(0)
            self.env.render()
            if terminated:
                break

    def plot_info(self, show_result=False, info=None):
        episode_max_tile, episode_reward, episode_steps = info

        plt.figure(1)
        max_tile_t = torch.tensor(episode_max_tile, dtype=torch.float)
        reward_t = torch.tensor(episode_reward, dtype=torch.float)
        steps_t = torch.tensor(episode_steps, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Max Tile')

        plt.plot(max_tile_t.numpy())
        # 100 avg
        avg_length = 100
        if len(max_tile_t) >= avg_length:
            means_max_tile = max_tile_t.unfold(0, avg_length, 1).mean(1).view(-1)
            means_max_tile = torch.cat((torch.zeros(avg_length - 1), means_max_tile))
            plt.plot(means_max_tile.numpy(), color="red")

            means_reward = reward_t.unfold(0, avg_length, 1).mean(1).view(-1)
            means_reward = torch.cat((torch.zeros(avg_length - 1), means_reward))
            plt.plot(means_reward.numpy(), color="green")

            plt.twinx()
            plt.ylabel('Episode Steps')
            means_steps = steps_t.unfold(0, avg_length, 1).mean(1).view(-1)
            means_steps = torch.cat((torch.zeros(avg_length - 1), means_steps))
            plt.plot(means_steps.numpy(), color="blue")

        plt.pause(0.001)  # pause a bit so that plots are updated
