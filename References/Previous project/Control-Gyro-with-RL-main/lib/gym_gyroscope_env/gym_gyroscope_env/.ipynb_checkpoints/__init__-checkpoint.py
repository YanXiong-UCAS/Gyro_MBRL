from gym.envs.registration import register

register(
    id = 'GyroscopeEnv-v0', 
    entry_point = 'gym_gyroscope_env.envs:GyroscopeEnvV0'
)

register(
    id = 'GyroscopeEnv-v1', 
    entry_point = 'gym_gyroscope_env.envs:GyroscopeEnvV1'
)

register(
    id = 'GyroscopeDiscontinuousEnv-v0', 
    entry_point = 'gym_gyroscope_env.envs:GyroscopeDiscontinuousEnv'
)

register(
    id = 'GyroscopeIntegralEnv-v0', 
    entry_point = 'gym_gyroscope_env.envs:GyroscopeIntegralEnvV0'
)

register(
    id = 'GyroscopeIntegralEnv-v1', 
    entry_point = 'gym_gyroscope_env.envs:GyroscopeIntegralEnvV1'
)

register(
    id = 'GyroscopeRobustEnv-v0', 
    entry_point = 'gym_gyroscope_env.envs:GyroscopeRobustEnv'
)

register(
    id = 'GyroscopeRealEnv-v0', 
    entry_point = 'gym_gyroscope_env.envs:GyroscopeRealEnv'
)

