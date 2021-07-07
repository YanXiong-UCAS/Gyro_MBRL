# Double gimbal control moment gyroscope Gym environment

- To create new environments for Gym

https://github.com/openai/gym/blob/master/docs/creating-environments.md

- To import a specific environment, update the __init__.py file in gym_gyroscope_env/gym_gyroscope_env:

from gym.envs.registration import register

register(
    id = 'GyroscopeEnv-v0', 
    entry_point = 'gym_gyroscope_env.envs:GyroscopeEnvV0'
)

- Also update the __init__.py file in gym_gyroscope_env/gym_gyroscope_env/envs:

from gym_gyroscope_env.envs.gyroscope_env_v0 import GyroscopeEnvV0

- To use using the env_fn from the custom_functions module function that allows to pass arguments to the class:

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
env_fn = partial(env_fn, env_name, simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)

- To install: 

pip3 install -e .

