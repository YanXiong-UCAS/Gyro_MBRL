Test implementation of gym environment

Test loading and training gym environment package

Compare 2 environments v0 and v1
    GyroscopeEnv-v0: gimbal velocity clip to [-100,100] rpm
    GyroscopeEnv-v1: remove such constraits
    It's found that model trained on GyroscopeEnv-v0 can be unstable on GyroscopeEnv-v1

Validate NN implementation
