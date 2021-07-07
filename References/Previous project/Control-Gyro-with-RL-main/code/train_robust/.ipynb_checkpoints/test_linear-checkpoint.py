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

import multiprocessing as mp

def mp_test(l, r):
    
    # Env function
    env_name = 'GyroscopeRobustEnv-v0'
    simu_args = {
        'dt': 0.05,
        'ep_len': 100,
        'seed': 2,
        'obs_noise': 0.001
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
    # Env
    env = create_env(env_name,simu_args,reward_func,reward_args,state=None)

    # Agent
    agent_path = 'linearized controller'
    agent = load_agent(agent_path)

    # num and set
    states = np.genfromtxt('states10k.csv', delimiter=',')
    states = states[l:r]
    num_test = len(states)
    
    t_end = 10
    ss_bound = 0.25

    start_time = time.time()
    print(agent_path + '_' + str(l) + '_' + str(r), 'start time:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))

    metrics = evaluate_control(env,agent,agent_path,t_end,ss_bound,num_test,states,print_unsteady=True)
    file_path = agent_path + '_' + str(l) + '_' + str(r) + '_metrics.csv'

    metrics_man = metrics.transpose()
#     metrics_man = metrics_man.round(4)
    metrics_man.to_csv(file_path,index=True)

    print(agent_path + '_' + str(l) + '_' + str(r), 'finish time:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))
    print(agent_path + '_' + str(l) + '_' + str(r), 'total time: ', time.time() - start_time)    
    
    return metrics_man

# num_cores = int(mp.cpu_count())
# print('num_cores = ', num_cores)
# pool = mp.Pool(num_cores)

# param_list = []
# num_per_test = 100
# for i in range(16):
#     param_list.append([num_per_test*i, num_per_test*(i+1)])
# results = [pool.apply_async(mp_test, args=(l,r)) for l, r in param_list]
# results = [p.get() for p in results]

mp_test(1100, 1200)

print('All done!')



# # Test paramaters
# env_name = 'GyroscopeRobustEnv-v0'
# # Initialization args
# simu_args = {
#     'dt': 0.05,
#     'ep_len': 100,
#     'seed': 2,
#     'obs_noise': 0.001
# }
# reward_func = 'PE'
# reward_args = {
#     'qx1': 1, 
#     'qx2': 0.2, 
#     'qx3': 1, 
#     'qx4': 0.2, 
#     'pu1': 0.1, 
#     'pu2': 0.1,
#     'p': 0.1,
#     'e': 40
# }
# # Env
# env = create_env(env_name)

# # num and set
# num_test = 2
# states = np.genfromtxt('states10k.csv', delimiter=',')

# # Init dataframe
# agent_path = 'linearized controller'
# t_end = 10
# ss_bound = 0.25

# start_time = time.time()
# print(agent_path, 'start time:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))
# agent = load_agent(agent_path)
# metrics = evaluate_control(env,agent,agent_path,t_end,ss_bound,num_test,states,print_unsteady=True)
# print(agent_path, 'finish time:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))
# print(agent_path, 'total time: ', time.time() - start_time)
# print()

# file_path = agent_path + '_metrics.csv'

# metrics_man = metrics.transpose()
# metrics_man = metrics_man.round(4)
# metrics_man.to_csv(file_path,index=True)