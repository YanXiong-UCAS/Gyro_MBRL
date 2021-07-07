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

# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Sparse with root'
reward_args = {
    'qx1': 0, 
    'qx2': 0, 
    'qx3': 0, 
    'qx4': 0, 
    'pu1': 1, 
    'pu2': 1,
    'r': 1,
    'bound': 0.01,
    'reward': 10,
    'velocity': True
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
spinup.ddpg_mher_pytorch(env_fn_, 
                            ac_kwargs = dict(hidden_sizes=[128,32], activation=torch.nn.ReLU), 
                            seed = 0, 
                            steps_per_epoch = 1500, 
                            epochs = 5000, 
                            replay_size = 1000000, 
                            gamma = 0.995, 
                            polyak = 0.995, 
                            pi_lr = 0.0025,
                            q_lr = 0.0025,
                            batch_size = 100, 
                            start_steps = 10000,
                            act_noise = 0.1,
                            max_ep_len = 100, 
                            logger_kwargs = dict(output_dir='ddpg_sr_r1b001v_mher_5000epochs', exp_name='ddpg_sr_r1b001v_mher_5000epochs'),
                            trained_ac = None,
                            her = True)