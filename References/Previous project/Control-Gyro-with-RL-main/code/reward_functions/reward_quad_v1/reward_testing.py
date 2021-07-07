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

# Env
env_name = 'GyroscopeEnv-v1'
env = create_env(env_name)

# num and set
num_test = 10000
states = np.genfromtxt('states10k.csv', delimiter=',')

# Init dataframe
agent_paths = ['ddpg_q','ddpg_q_b','ddpg_q_e','ddpg_q_ep','ddpg_q_p','ddpg_a','ddpg_n','ddpg_n_b','ddpg_s']
t_end = 10
ss_bound = 0.25

# Loop dataframe
for idx, agent_path in enumerate(agent_paths):
    print(agent_path)
    agent = load_agent(agent_path)
    if idx == 0:
        metrics = evaluate_control(env,agent,agent_path,t_end,ss_bound,num_test,states)
    else:
        new_metrics = evaluate_control(env,agent,agent_path,t_end,ss_bound,num_test,states)
        metrics = metrics.append(new_metrics)
        
metrics_man = metrics.transpose()
metrics_man = metrics_man.round(4)

metrics_man.to_csv('Metrics_QAN.csv',index=True)


# Init dataframe
agent_paths = ['ddpg_p005','ddpg_p01','ddpg_p05','ddpg_e10_2000epochs','ddpg_e20_2000epochs',
               'ddpg_e40_2000epochs','ddpg_p01e40_2000epochs','ddpg_p01e80_2000epochs']

t_end = 10
ss_bound = 0.25

# Loop dataframe
for idx, agent_path in enumerate(agent_paths):
    start_time = time.time()
    agent = load_agent(agent_path)
    if idx == 0:
        metrics = evaluate_control(env,agent,agent_path,t_end,ss_bound,num_test,states)
    else:
        new_metrics = evaluate_control(env,agent,agent_path,t_end,ss_bound,num_test,states)
        metrics = metrics.append(new_metrics)
    elapsed_time = time.time() - start_time
    print(agent_path, elapsed_time)
        
metrics_man = metrics.transpose()
metrics_man = metrics_man.round(4)

metrics_man.to_csv('Metrics_PE.csv',index=True)