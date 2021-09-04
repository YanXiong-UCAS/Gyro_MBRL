# %% md

# Model testing with PE on Env-v1 and Export NN_weights_bias to file.txt
# 模型测试以及将模型参数（权重&偏差等）输出到指定文本文件

# %%

import gym
import numpy
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from scipy.integrate import solve_ivp
import pandas as pd

import gym_gyroscope_env
import spinup
import torch
from functools import partial

import random
from collections import deque
import matplotlib.pyplot as plt

from custom_functions.custom_functions import env_fn
from custom_functions.custom_functions import create_env
from custom_functions.custom_functions import load_agent
from custom_functions.custom_functions import test_agent
from custom_functions.custom_functions import plot_test
from custom_functions.custom_functions import evaluate_control
from custom_functions.custom_functions import read_progress

import sys

# %% md

# Plot rewards   绘制奖励函数变化曲线

# %%

plt.figure(figsize=(20, 10))
plt.title('Average Epoch Return', fontsize=28)
plt.xlabel('Epoch', fontsize=24)
plt.xticks(fontsize=24)
plt.ylabel('Average Epoch Return', fontsize=24)
plt.yticks(fontsize=24)
plt.grid()

# agent_paths = ['m0_005']  # 选择模型
agent_paths = ['td3_pe_opt_ing_25000']

for agent_path in agent_paths[:]:  # 如果选择多个模型，则将绘制多个模型的奖励函数的变化曲线
    progress = read_progress(agent_path)
    plt.plot(np.arange(progress.shape[0]), progress[:, 1] / abs(max(progress[:, 1])))

plt.legend(agent_paths, fontsize=24)
# plt.legend(['PE'],fontsize=24)   # 奖励函数类型
plt.savefig('reward_function_curve.png')

# %% md

# Test an agent   # 测试模型

# %%

env_name = 'GyroscopeEnv-v1'  # 指定测试环境
init_state = np.array([0, 0, 0, 0, 45 / 180 * np.pi, -60 / 180 * np.pi, 200 / 60 * 2 * np.pi])  # 初始化状态空间
env = create_env(env_name, state=init_state)  # 根据初始化环境参数设置环境
# agent_paths = ['m0_005']  # 选择模型
agent_paths = ['td3_pe_opt_ing_25000']
agent = load_agent(agent_paths[0])  # 加载模型

t_end = 5  # 测试步长

score, state_record, obs_record, action_record, reward_record = test_agent(env, agent, t_end)  # 指定环境下测试模型
plot_test(state_record, action_record, t_end, 4)  # 绘制测试效果

# %%

# Export NN_weights_bias to file.txt
# 将训练的模型（weights和Bias）输出到指定文本文件，并根据labview所需的C++程序要求修改数据格式

# %%

f = "TD3_NN_weights_bias_25000.txt"  # 打开指定文本文件
numpy.set_printoptions(threshold=sys.maxsize)  # 用于设置文本输出数据的显示长度

with open(f, "w") as file:
    file.write('const float w1[sz_hid_1][sz_obs] = ')
    text = numpy.array2string(agent.pi.pi[0].weight.detach().numpy(), separator=",")
    file.write(text.replace('[', '{').replace(']', '}') + ";\n\n")

    file.write("const float b1[sz_hid_1] = ")
    text = numpy.array2string(agent.pi.pi[0].bias.detach().numpy(), separator=",")
    file.write(text.replace('[', '{').replace(']', '}') + ";\n\n")

    file.write('const float w2[sz_hid_2][sz_hid_1] = ')
    text = numpy.array2string(agent.pi.pi[2].weight.detach().numpy(), separator=",")
    file.write(text.replace('[', '{').replace(']', '}') + ";\n\n")

    file.write('const float b2[sz_hid_2] = ')
    text = numpy.array2string(agent.pi.pi[2].bias.detach().numpy(), separator=",")
    file.write(text.replace('[', '{').replace(']', '}') + ";\n\n")

    file.write('const float w3[sz_act][sz_hid_2] = ')
    text = numpy.array2string(agent.pi.pi[4].weight.detach().numpy(), separator=",")
    file.write(text.replace('[', '{').replace(']', '}') + ";\n\n")

    file.write('const float b3[sz_act] = ')
    text = numpy.array2string(agent.pi.pi[4].bias.detach().numpy(), separator=",")
    file.write(text.replace('[', '{').replace(']', '}') + ";\n\n")

print(agent.named_parameters)  # 只显示模型结构，不现实具体数值

print('------Export Finished------')


