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
agent_paths = ['td3_pe_opt_ing_100000']

for agent_path in agent_paths[:]:  # 如果选择多个模型，则将绘制多个模型的奖励函数的变化曲线
    progress = read_progress(agent_path)
    plt.plot(np.arange(progress.shape[0]), progress[:, 1] / abs(max(progress[:, 1])))

plt.legend(agent_paths, fontsize=24)
# plt.legend(['PE'],fontsize=24)   # 奖励函数类型
plt.savefig('reward_function_td3_pe_opt_ing_100000.png')

# %% md

# Test an agent   # 测试模型

# %%

# 测试Set-point跟踪效果






# %%
# 测试Set-point跟踪效果
env_name = 'GyroscopeEnvNew4Paper-v0'  # 指定测试环境，同样指向转为本研究设计的环境
init_state = np.array([0, 0, 0, 0, 45 / 180 * np.pi, -60 / 180 * np.pi, 200 / 60 * 2 * np.pi])  # 初始化状态空间
env = create_env(env_name, state=init_state)  # 根据初始化环境参数设置环境
# agent_paths = ['m0_005']  # 选择模型
agent_paths = ['td3_pe_opt_ing_100000']
agent = load_agent(agent_paths[0])  # 加载模型

t_end = 100  # 测试步长

# np.array([0] * 100)
# Set-point tracking仿真时间为25s，每个阶段5s，共分为四个阶段，分别为
# Red Gimbal[0.9 > -0.8 > 0.8 > -0.9], Blue Gimbal[-0.9 > 0.8 > -0.8 > 0.9]，Disk[55 > 40 > 50 > 35]
# Disk转速控制 [rad/s]   >>>   建议修改成函数方程式，相对简单一些！也容易修改，如果的哦欧式数字的话，修改起来太复杂麻烦！
disk_setpoint = [55]*500 + [40]*500 + [50]*500 + [35]*500
# Red Gimbal控制  [rad]
redg_setpoint = [0.9]*500 + [-0.8]*500 + [0.8]*500 + [-0.9]*500
# Blue Gimbal控制  [rad]
blueg_setpoint = [-0.9]*500 + [0.8]*500 + [-0.8]*500 +[0.9]*500

score_setp, state_record_setp, obs_record_setp, action_record_setp, reward_record_setp = test_agent(env, agent, t_end, w_seq=disk_setpoint, x1_ref_seq=redg_setpoint, x3_ref_seq=blueg_setpoint)  # 指定环境下测试模型
plot_test(state_record_setp, action_record_setp, t_end, 4)  # 绘制测试效果

# %%

# 保存测试数据('score'、'state_record'、'obs_record'、'action_record'、'reward_record')
main_data_path = "/media/xiongyan/Data_Repositories/Project_code/Gyro_MBRL/Gyro_MBRL/code/Basic_code/td3_reward_function_comparision/new4paper/test_data/set-point"
shutil.rmtree(main_data_path)
os.mkdir(main_data_path)

state_record_numpy_setp = state_record_setp
obs_record_numpy_setp = obs_record_setp
action_record_numpy_setp = action_record_setp
reward_record_numpy_setp = reward_record_setp
state_record_dataframe_setp = pd.DataFrame(state_record_numpy_setp)   # 将Numpy转换为pandas,因为Numpy和Tensor都不支持to_csv
obs_record_dataframe_setp = pd.DataFrame(obs_record_numpy_setp)
action_record_dataframe_setp = pd.DataFrame(action_record_numpy_setp)
reward_record_dataframe_setp = pd.DataFrame(reward_record_numpy_setp)

# 保存'score'、'state_record'、'obs_record'、'action_record'、'reward_record'
state_record_dataframe_setp.to_csv('/media/xiongyan/Data_Repositories/Project_code/Gyro_MBRL/Gyro_MBRL/code/Basic_code/td3_reward_function_comparision/new4paper/test_data/set-point/state_record_setp_td3_pe_opt_ing_100000.csv')   # 创建CSV文件并存储到CSV中
obs_record_dataframe_setp.to_csv('/media/xiongyan/Data_Repositories/Project_code/Gyro_MBRL/Gyro_MBRL/code/Basic_code/td3_reward_function_comparision/new4paper/test_data/set-point/obs_record_setp_td3_pe_opt_ing_100000.csv')
action_record_dataframe_setp.to_csv('/media/xiongyan/Data_Repositories/Project_code/Gyro_MBRL/Gyro_MBRL/code/Basic_code/td3_reward_function_comparision/new4paper/test_data/set-point/action_record_setp_td3_pe_opt_ing_100000.csv')
reward_record_dataframe_setp.to_csv('/media/xiongyan/Data_Repositories/Project_code/Gyro_MBRL/Gyro_MBRL/code/Basic_code/td3_reward_function_comparision/new4paper/test_data/set-point/reward_record_setp_td3_pe_opt_ing_100000.csv')



# %%

# 测试Reference跟踪效果






# %%
# 测试Reference跟踪效果
env_name = 'GyroscopeEnvNew4Paper-v0'  # 指定测试环境，同样指向转为本研究设计的环境
init_state = np.array([0, 0, 0, 0, 45 / 180 * np.pi, -60 / 180 * np.pi, 200 / 60 * 2 * np.pi])  # 初始化状态空间
env = create_env(env_name, state=init_state)  # 根据初始化环境参数设置环境
# agent_paths = ['m0_005']  # 选择模型
agent_paths = ['td3_pe_opt_ing_100000']
agent = load_agent(agent_paths[0])  # 加载模型

t_end = 20  # 测试步长

# np.array([0] * 100)
ref_matrix = np.loadtxt(open("td3ref.csv","rb"),delimiter=",",skiprows=0)

# Reference tracking仿真时间为4s，正弦变化曲线，周期为2s，极大值1，极小值-1
# Disk转速控制 [rad/s]   >>>   建议修改成函数方程式，相对简单一些！也容易修改，如果的哦欧式数字的话，修改起来太复杂麻烦！   numpy.array2string(ref_matrix[:,0].tolist(),separator=",").replace('\n ','')
disk_ref = ref_matrix[:,2].tolist()
# Red Gimbal控制  [rad]
redg_ref = ref_matrix[:,1].tolist()
# Blue Gimbal控制  [rad]
blueg_ref = ref_matrix[:,0].tolist()

score_ref, state_record_ref, obs_record_ref, action_record_ref, reward_record_ref = test_agent(env, agent, t_end, w_seq=disk_ref, x1_ref_seq=redg_ref, x3_ref_seq=blueg_ref)  # 指定环境下测试模型
plot_test(state_record_ref, action_record_ref, t_end, 4)  # 绘制测试效果

# %%

# 保存测试数据('score'、'state_record'、'obs_record'、'action_record'、'reward_record')
main_data_path = "/media/xiongyan/Data_Repositories/Project_code/Gyro_MBRL/Gyro_MBRL/code/Basic_code/td3_reward_function_comparision/new4paper/test_data/ref"
shutil.rmtree(main_data_path)
os.mkdir(main_data_path)

state_record_numpy_ref = state_record_ref
obs_record_numpy_ref = obs_record_ref
action_record_numpy_ref = action_record_ref
reward_record_numpy_ref = reward_record_ref
state_record_dataframe_ref = pd.DataFrame(state_record_numpy_ref)   # 将Numpy转换为pandas,因为Numpy和Tensor都不支持to_csv
obs_record_dataframe_ref = pd.DataFrame(obs_record_numpy_ref)
action_record_dataframe_ref = pd.DataFrame(action_record_numpy_ref)
reward_record_dataframe_ref = pd.DataFrame(reward_record_numpy_ref)

# 保存'score'、'state_record'、'obs_record'、'action_record'、'reward_record'
state_record_dataframe_ref.to_csv('/media/xiongyan/Data_Repositories/Project_code/Gyro_MBRL/Gyro_MBRL/code/Basic_code/td3_reward_function_comparision/new4paper/test_data/ref/state_record_ref_td3_pe_opt_ing_100000.csv')   # 创建CSV文件并存储到CSV中
obs_record_dataframe_ref.to_csv('/media/xiongyan/Data_Repositories/Project_code/Gyro_MBRL/Gyro_MBRL/code/Basic_code/td3_reward_function_comparision/new4paper/test_data/ref/obs_record__ref_td3_pe_opt_ing_100000.csv')
action_record_dataframe_ref.to_csv('/media/xiongyan/Data_Repositories/Project_code/Gyro_MBRL/Gyro_MBRL/code/Basic_code/td3_reward_function_comparision/new4paper/test_data/ref/action_record_ref_td3_pe_opt_ing_100000.csv')
reward_record_dataframe_ref.to_csv('/media/xiongyan/Data_Repositories/Project_code/Gyro_MBRL/Gyro_MBRL/code/Basic_code/td3_reward_function_comparision/new4paper/test_data/ref/reward_record_ref_td3_pe_opt_ing_100000.csv')



# %%

# Export NN_weights_bias to file.txt
# 将训练的模型（weights和Bias）输出到指定文本文件，并根据labview所需的C++程序要求修改数据格式

# %%

f = "TD3_NN_weights_bias_td3_pe_opt_ing_100000.txt"  # 打开指定文本文件
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
