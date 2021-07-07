import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from scipy.integrate import solve_ivp

import gym_gyroscope_env
import spinup
import torch
from functools import partial

from custom_functions.custom_functions import env_fn 
from custom_functions.custom_functions import create_env
from custom_functions.custom_functions import load_agent
from custom_functions.custom_functions import test_agent
from custom_functions.custom_functions import plot_test
from custom_functions.custom_functions import evaluate_control

import random
from collections import deque
import time

env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'PE'
reward_args = {
    'qx1': 1, 
    'qx2': 0.2, 
    'qx3': 1, 
    'qx4': 0.2, 
    'pu1': 0.1, 
    'pu2': 0.1,
    'p': 0.1,
    'e': 40
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

spinup.ddpg_pytorch(env_fn_, 
                    ac_kwargs = dict(hidden_sizes=[128,32], activation=torch.nn.ReLU), 
                    seed = 0, 
                    steps_per_epoch = 1500, 
                    epochs = 5000, 
                    replay_size = 8000000, 
                    gamma = 0.96, 
                    polyak = 0.999, 
                    pi_lr = 0.002,
                    q_lr = 0.002,
                    batch_size = 100, 
                    start_steps = 15000,
                    act_noise = 0.1,
                    max_ep_len = 100, 
                    logger_kwargs = dict(output_dir='m0_005_v0', exp_name='m0_005_v0'),
                    save_file = True)
