#!/usr/bin/env python
# coding: utf-8

# # Model training with quadratic, exponential, and other reward functions on Env-v1

# In[ ]:


# trained in reward_training.py


# In[ ]:


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


# #### Quadratic reward

# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Quadratic'
reward_args = {
    'qx1': 1, 
    'qx2': 0, 
    'qx3': 1, 
    'qx4': 0, 
    'pu1': 0, 
    'pu2': 0
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_q', exp_name='ddpg_q'))


# #### Absolute reward

# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Absolute'
reward_args = {
    'qx1': 1, 
    'qx2': 0, 
    'qx3': 1, 
    'qx4': 0, 
    'pu1': 0, 
    'pu2': 0
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_a', exp_name='ddpg_a'))


# #### Normalized reward

# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Normalized'
reward_args = {
    'k': 1,
    'qx2': 1, 
    'qx4': 1, 
    'pu1': 0,
    'pu2': 0
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_n', exp_name='ddpg_n'))

# Following parameters are used in previous project
# startsteps_b = 20000
# pilr_b = 0.001
# qlr_b = 0.001


# #### Quadratic reward with ending penalty

# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Quadratic with ending penalty'
reward_args = {
    'qx1': 1, 
    'qx2': 0, 
    'qx3': 1, 
    'qx4': 0, 
    'pu1': 0, 
    'pu2': 0,
    'sx1': 100, 
    'sx3': 100, 
    'end_horizon': 0
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_q_ep', exp_name='ddpg_q_ep'))


# #### Quadratic reward with penalty 

# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Quadratic with penalty'
reward_args = {
    'qx1': 1, 
    'qx2': 0, 
    'qx3': 1, 
    'qx4': 0, 
    'pu1': 0, 
    'pu2': 0,
    'bound': 0.2,
    'penalty': 10
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_q_p', exp_name='ddpg_q_p'))


# #### Quadratic reward with exponential term

# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Quadratic with exponential'
reward_args = {
    'qx1': 1,
    'qx2': 0,
    'qx3': 1,
    'qx4': 0,
    'pu1': 0,
    'pu2': 0,
    'eax1': 10,
    'ebx1': 10,
    'eax3': 10,
    'ebx3': 10
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_q_e', exp_name='ddpg_q_e'))


# #### Quadratic reward with bonus

# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Quadratic with bonus'
reward_args = {
    'qx1': 1,
    'qx2': 0,
    'qx3': 1,
    'qx4': 0,
    'pu1': 0,
    'pu2': 0,
    'bound': 0.05,
    'bonus': 2
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_q_b', exp_name='ddpg_q_b'))


# #### Normalized reward with bonus

# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Normalized with bonus'
reward_args = {
    'k': 1,
    'qx2': 1, 
    'qx4': 1, 
    'pu1': 0,
    'pu2': 0,
    'bound': 0.05, 
    'bonus': 2
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_n_b', exp_name='ddpg_n_b'))


# #### Sparse reward

# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Sparse'
reward_args = {
    'bx': 0.05,
    'rx': 1, 
    'bv': 0, 
    'rv': 0, 
    'bu': 0, 
    'ru': 0
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_s', exp_name='ddpg_s'))


# In[ ]:





# In[ ]:





# #### Power function reward

# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Power'
reward_args = {
    'qx1': 1, 
    'qx2': 0, 
    'qx3': 1, 
    'qx4': 0, 
    'pu1': 0, 
    'pu2': 0,
    'p': 0.05
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_p005', exp_name='ddpg_p005'))


# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Power'
reward_args = {
    'qx1': 1, 
    'qx2': 0, 
    'qx3': 1, 
    'qx4': 0, 
    'pu1': 0, 
    'pu2': 0,
    'p': 0.1
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_p01', exp_name='ddpg_p01'))


# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Power'
reward_args = {
    'qx1': 1, 
    'qx2': 0, 
    'qx3': 1, 
    'qx4': 0, 
    'pu1': 0, 
    'pu2': 0,
    'p': 0.5
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_p05', exp_name='ddpg_p05'))


# #### Exponential reward

# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Exponential'
reward_args = {
    'qx1': 1, 
    'qx2': 0, 
    'qx3': 1, 
    'qx4': 0, 
    'pu1': 0, 
    'pu2': 0,
    'e': 10
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_e10', exp_name='ddpg_e10'))


# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Exponential'
reward_args = {
    'qx1': 1, 
    'qx2': 0, 
    'qx3': 1, 
    'qx4': 0, 
    'pu1': 0, 
    'pu2': 0,
    'e': 20
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_e20', exp_name='ddpg_e20'))


# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Exponential'
reward_args = {
    'qx1': 1, 
    'qx2': 0, 
    'qx3': 1, 
    'qx4': 0, 
    'pu1': 0, 
    'pu2': 0,
    'e': 40
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_e40', exp_name='ddpg_e40'))


# #### PE reward

# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'PE'
reward_args = {
    'qx1': 1, 
    'qx2': 0, 
    'qx3': 1, 
    'qx4': 0, 
    'pu1': 0, 
    'pu2': 0,
    'p': 0.1,
    'e': 40
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_p01e40', exp_name='ddpg_p01e40'))


# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'PE'
reward_args = {
    'qx1': 1, 
    'qx2': 0, 
    'qx3': 1, 
    'qx4': 0, 
    'pu1': 0, 
    'pu2': 0,
    'p': 0.1,
    'e': 80
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
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
                    logger_kwargs = dict(output_dir='ddpg_p01e80', exp_name='ddpg_p01e80'))


# In[ ]:





# In[ ]:





# #### further study on p = 0.05, why it doesnt learn

# In[ ]:


# Env function
env_name = 'GyroscopeEnv-v1'
simu_args = {
    'dt': 0.05,
    'ep_len': 100,
    'seed': 2
}
reward_func = 'Power'
reward_args = {
    'qx1': 1, 
    'qx2': 0, 
    'qx3': 1, 
    'qx4': 0, 
    'pu1': 0, 
    'pu2': 0,
    'p': 0.05
}
env_fn_ = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

# Baseline 0 training
spinup.ddpg_pytorch(env_fn_, 
                    ac_kwargs = dict(hidden_sizes=[128,32], activation=torch.nn.ReLU), 
                    seed = 0, 
                    steps_per_epoch = 1500, 
                    epochs = 500, 
                    replay_size = 1000000, 
                    gamma = 0.95, 
                    polyak = 0.995, 
                    pi_lr = 0.0025,
                    q_lr = 0.0025,
                    batch_size = 100, 
                    start_steps = 10000,
                    act_noise = 0.1,
                    max_ep_len = 100, 
                    logger_kwargs = dict(output_dir='ddpg_p005_gamma095', exp_name='ddpg_p005_gamma095'))


# In[ ]:




