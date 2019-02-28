import numpy as np
import gym
import argparse

import torch

from train import train


parser = argparse.ArgumentParser(description='Categorical DQN')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--vmax', type=float, default=10)
parser.add_argument('--vmin', type=float, default=-10)
parser.add_argument('--atom', type=int, default=51)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--game', type=str, default='CartPole-v0')
parser.add_argument('--max_episode_length', type=int, default=int(50e6))
parser.add_argument('--memory_capacity', type=int, default=int(1e5))
parser.add_argument('--discount', type=float, default=0.99)
parser.add_argument('--target_update', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.0000625)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learn_start', type=int, default=1000)

args = parser.parse_args()
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
	print(' '* 26 + k + ': ' + str(v))
np.random.seed(args.seed)
torch.manual_seed(args.seed)

env = gym.make(args.game)

train(args, env)
