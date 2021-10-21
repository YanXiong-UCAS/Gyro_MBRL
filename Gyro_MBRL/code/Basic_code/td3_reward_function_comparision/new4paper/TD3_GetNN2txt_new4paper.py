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
import shutil
import os

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
agent_paths = ['td3_pe_opt_ing_45000']

for agent_path in agent_paths[:]:  # 如果选择多个模型，则将绘制多个模型的奖励函数的变化曲线
    progress = read_progress(agent_path)
    plt.plot(np.arange(progress.shape[0]), progress[:, 1] / abs(max(progress[:, 1])))

plt.legend(agent_paths, fontsize=24)
# plt.legend(['PE'],fontsize=24)   # 奖励函数类型
plt.savefig('reward_function_td3_pe_opt_ing_45000.png')

# %% md

# Test an agent   # 测试模型

# %%

env_name = 'GyroscopeEnvNew4Paper-v0'  # 指定测试环境，同样指向转为本研究设计的环境
init_state = np.array([0, 0, 0, 0, 45 / 180 * np.pi, -60 / 180 * np.pi, 200 / 60 * 2 * np.pi])  # 初始化状态空间
env = create_env(env_name, state=init_state)  # 根据初始化环境参数设置环境
# agent_paths = ['m0_005']  # 选择模型
agent_paths = ['td3_pe_opt_ing_45000']
agent = load_agent(agent_paths[0])  # 加载模型

t_end = 20  # 测试步长

# np.array([0] * 100)
# Set-point tracking仿真时间为25s，每个阶段5s，共分为四个阶段，分别为
# Red Gimbal[1 > -1 > 1 >-1], Blue Gimbal[-1 > 1 > -1 >1]，Disk[55 > 40 > 50 > 35]
# Reference tracking仿真时间为4s，正弦变化曲线，周期为2s，极大值1，极小值-1

# Disk转速控制 [rad/s]   >>>   建议修改成函数方程式，相对简单一些！也容易修改，如果的哦欧式数字的话，修改起来太复杂麻烦！
disk = [55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
       55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
       55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
       55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
       55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
       55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,

        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,

        50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
        50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
        50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
        50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
        50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
        50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,

       35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35,
       35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35,
       35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35,
       35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35,
       35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35,
       35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35
        ]

# Red Gimbal控制  [rad]
redg = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
       0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
       0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
       0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
       0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
       0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
       0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
       0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,

        -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
        -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
        -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
        -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
        -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
        -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
        -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
        -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
        -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
        -0.8,

        0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
        0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
        0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
        0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
        0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
        0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
        0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
        0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,

        -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
        -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
        -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
        -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
        -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
        -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
        -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
        -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
        -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
        -0.9,
         ]

# Blue Gimbal控制  [rad]
blueg = [-0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
       -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
       -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
       -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
       -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
       -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
       -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
       -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
       -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9,
       -0.9,

         0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
         0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
         0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
         0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
         0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
         0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
         0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
         0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,

         -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
         -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
         -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
         -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
         -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
         -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
         -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
         -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
         -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
         -0.8,

         0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
        ]


score, state_record, obs_record, action_record, reward_record = test_agent(env, agent, t_end, w_seq=disk, x1_ref_seq=redg, x3_ref_seq=blueg)  # 指定环境下测试模型
plot_test(state_record, action_record, t_end, 4)  # 绘制测试效果


# file:///media/xiongyan/Data_Repositories/Project_code/Gyro_MBRL/Gyro_MBRL/code/Basic_code/td3_reward_function_comparision/new4paper/test_data


# 保存测试数据
main_data_path = "/media/xiongyan/Data_Repositories/Project_code/Gyro_MBRL/Gyro_MBRL/code/Basic_code/td3_reward_function_comparision/new4paper/test_data"
shutil.rmtree(main_data_path)
os.mkdir(main_data_path)

state_record_numpy = state_record
action_record_numpy = action_record
state_record_dataframe = pd.DataFrame(state_record_numpy)   # 将Numpy转换为pandas,因为Numpy和Tensor都不支持to_csv
action_record_dataframe = pd.DataFrame(action_record_numpy)

state_record_dataframe.to_csv('/media/xiongyan/Data_Repositories/Project_code/Gyro_MBRL/Gyro_MBRL/code/Basic_code/td3_reward_function_comparision/new4paper/test_data/state_record_td3_pe_opt_ing_45000.csv')   # 创建CSV文件并存储到CSV中
action_record_dataframe.to_csv('/media/xiongyan/Data_Repositories/Project_code/Gyro_MBRL/Gyro_MBRL/code/Basic_code/td3_reward_function_comparision/new4paper/test_data/action_record_td3_pe_opt_ing_45000.csv')   # 创建CSV文件并存储到CSV中



# %%

# Export NN_weights_bias to file.txt
# 将训练的模型（weights和Bias）输出到指定文本文件，并根据labview所需的C++程序要求修改数据格式

# %%

f = "TD3_NN_weights_bias_td3_pe_opt_ing_45000.txt"  # 打开指定文本文件
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
