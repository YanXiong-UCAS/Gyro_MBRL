import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from scipy.integrate import solve_ivp

import gym_gyroscope_env
import spinup
import torch
from custom_functions.custom_functions import env_fn 
from functools import partial

# test_training_with_clip
env_name = 'GyroscopeEnv-v0'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Quadratic'
reward_args = {
    'qx1': 9, 
    'qx2': 0.05, 
    'qx3': 9, 
    'qx4': 0.05, 
    'pu1': 0.1, 
    'pu2': 0.1
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

spinup.ddpg_pytorch(env_fn_, 
                    ac_kwargs = dict(hidden_sizes=[128,32], activation=torch.nn.ReLU), 
                    seed = 0, 
                    steps_per_epoch = 1500, 
                    epochs = 500, 
                    replay_size = 1000000, 
                    gamma = 0.995, 
                    polyak = 0.995, 
                    pi_lr = 0.0025,
                    q_lr = 0.0025,
                    batch_size = 100, 
                    start_steps = 10000,
                    act_noise = 0.1,
                    max_ep_len = 100, 
                    logger_kwargs = dict(output_dir='test_training_with_clip', exp_name='test_training_with_clip'))

# test_training_no_clip
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Quadratic'
reward_args = {
    'qx1': 9, 
    'qx2': 0.05, 
    'qx3': 9, 
    'qx4': 0.05, 
    'pu1': 0.1, 
    'pu2': 0.1
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

spinup.ddpg_pytorch(env_fn_, 
                    ac_kwargs = dict(hidden_sizes=[128,32], activation=torch.nn.ReLU), 
                    seed = 0, 
                    steps_per_epoch = 1500, 
                    epochs = 500, 
                    replay_size = 1000000, 
                    gamma = 0.995, 
                    polyak = 0.995, 
                    pi_lr = 0.0025,
                    q_lr = 0.0025,
                    batch_size = 100, 
                    start_steps = 10000,
                    act_noise = 0.1,
                    max_ep_len = 100, 
                    logger_kwargs = dict(output_dir='test_training_no_clip', exp_name='test_training_no_clip'))
