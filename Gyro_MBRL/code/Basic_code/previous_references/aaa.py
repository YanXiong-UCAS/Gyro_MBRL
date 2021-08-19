#%% md

# Model testing with quadratic, exponential, and other reward functions on Env-v1

#%%

# More tests in reward_testing.py

#%%

import gym
import param
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from scipy.integrate import solve_ivp
import pandas as pd
import xlrd   # 用来打开Excel表格数据的包，其实pandas中有pd.read_cvs也可以读取文件

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
from custom_functions.custom_functions import read_progress

import random
from collections import deque
import matplotlib.pyplot as plt
# %matplotlib inline

#%% md

## Plot rewards

#%%

plt.figure(figsize=(20,10))
plt.title('Average Epoch Return',fontsize=28)
plt.xlabel('Epoch',fontsize=24)
plt.xticks(fontsize=24)
plt.ylabel('Average Epoch Return',fontsize=24)
plt.yticks(fontsize=24)
plt.grid()

# agent_paths = ['sac_q/']
# agent_paths = ['ddpg_a','ddpg_n','ddpg_n_b','ddpg_q','ddpg_q_b','ddpg_q_e','ddpg_q_ep','ddpg_q_p','ddpg_s']
# agent_paths = ['ddpg_p005','ddpg_p01','ddpg_p05','ddpg_p005_gamma095','ddpg_p005_gamma1']
# agent_paths = ['ddpg_e10','ddpg_e20','ddpg_e40','ddpg_p01e40','ddpg_p01e80']
# agent_paths = ['ddpg_e10_2000epochs','ddpg_e20_2000epochs','ddpg_e40_2000epochs',
#                'ddpg_p01e40_2000epochs','ddpg_p01e80_2000epochs']
agent_paths = ["m0_005"]

for agent_path in agent_paths[:]:
    progress = read_progress(agent_path)
    plt.plot(np.arange(progress.shape[0]), progress[:,1]/abs(max(progress[:,1])))
#     plt.fill_between(np.arange(progress.shape[0]), progress[:,3], progress[:,4], alpha=0.5)
#     plt.fill_between(np.arange(progress.shape[0]), progress[:,1]+progress[:,2], progress[:,1]-progress[:,2], alpha=0.5)

# plt.xlim([0,500])
# plt.ylim([-2000,0])
plt.legend(agent_paths,fontsize=24)
# plt.legend(['A','N','NB','Q','QB','QE','QEP','QP'],fontsize=24)
# plt.legend([r'p=0.05,$\gamma$=0.995','p=0.1,$\gamma$=0.995','p=0.5,$\gamma$=0.995','p=0.05,$\gamma$=0.95'],fontsize=24)
# plt.legend(['c=-10','c=-20','c=-40'],fontsize=24)
# plt.savefig('quad_epoch.png')
# plt.savefig('power_epoch.png')
# plt.savefig('exp_epoch.png')
plt.savefig('tmp.png')

#%% md

## Test an agent

#%%

# Test paramaters
env_name = 'GyroscopeEnv-v1'

init_state = np.array([0,0,0,0,45/180*np.pi,-60/180*np.pi,200/60*2*np.pi])
env = create_env(env_name,state=init_state)

# agent_paths = ['ddpg_q','ddpg_q_b','ddpg_q_e','ddpg_q_ep','ddpg_q_p','ddpg_a','ddpg_n','ddpg_n_b','ddpg_s']
# agent_paths = ['ddpg_p005','ddpg_p01','ddpg_p05','ddpg_p005_gamma095']
# agent_paths = ['ddpg_e10','ddpg_e20','ddpg_e40','ddpg_p01e40','ddpg_p01e80']
# agent_paths = ['ddpg_e10_2000epochs','ddpg_e20_2000epochs','ddpg_e40_2000epochs',
#                'ddpg_p01e40_2000epochs','ddpg_p01e80_2000epochs']
agent_paths = ["m0_005"]

agent = load_agent(agent_paths[0])

# np.set_printoptions(torch.threshold == np.inf)


f = "results.txt"
with open(f, "w") as file:
    for name, param in agent.named_parameters():
        file.write(name)
        file.write(str(param.tolist()))

        print(name,' ', param)   # 显示具体数值，由于数据量过大，默认显示局部数据





# f = "results.txt"
#
# with open(f, "w") as file:
#     # for i in range(a):
#     file.write(name)
#     # a += 1

# print(agent.named_parameters)  # 只显示模型结构，不现实具体数值
t_end = 5

score, state_record, obs_record, action_record, reward_record = test_agent(env,agent,t_end)
plot_test(state_record, action_record, t_end, 4)

#%%

time = np.linspace(0, t_end, len(state_record))
n = 1
f, axs = plt.subplots(n,2,figsize=(20,6*n))

plt.subplot(n,2,1)
plt.title('Red gimbal angle',fontsize=24)
plt.xlabel('time [s]',fontsize=20)
plt.ylabel(r'$\theta$ [rad]',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()
plt.plot(time,env.angle_normalize(state_record[:,0]),'r-')
plt.plot(time,env.angle_normalize(state_record[:,4]),'g--')
# plt.plot(time,np.full(len(time),180), 'k-')
# plt.plot(time,np.full(len(time),-180), 'k-')
plt.legend(['Simulated','Reference'],fontsize=20)

plt.subplot(n,2,2)
plt.title('Blue gimbal angle',fontsize=24)
plt.xlabel('time [s]',fontsize=20)
plt.ylabel(r'$\phi$ [rad]',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()
plt.plot(time,env.angle_normalize(state_record[:,2]),'b-')
plt.plot(time,env.angle_normalize(state_record[:,5]),'g--')
# plt.plot(time,np.full(len(time),180), 'k-')
# plt.plot(time,np.full(len(time),-180), 'k-')
plt.legend(['Simulated','Reference'],fontsize=20)

plt.savefig('tmp.png')

