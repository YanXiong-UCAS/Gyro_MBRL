# Control-Gyro-with-RL

Control-Gyro-with-RL is the project folder for my master project *CONTROL OF A GYROSCOPE USING REINFORCEMENT LEARNING METHODS*
conducted during the 2020 fall semester at EPFL. 

For more information, please contact me at: <huang.zhitao@outlook.com>

## Installation

The project highly depends on two frameworks, namely OpenAI's *Gym* and *Spinning Up* frameworks, which can be installed as such:

- For Gym:

```pip install gym ```

- For Spinning Up:

```git clone https://github.com/openai/spinningup.git 
cd spinningup 
pip install -e . ```

## Files

The project folder consists of the following main directories:
```bash
├── code 'All code files of the project'
├── gyroscope datasheet 'Datasheet of gyro and its motors'
├── lib 'Required library'
└── report 'Report， slide, and a video of the report'
```

The *code* directory contains the utilities, experiments and results of the project. Not all directories and sub-directories are shown here, such as the many model folders that were generated during the experimentation process, but the most relevant ones are. To understand what the many model experimentation folders correspond to, please refer to the notebooks where the experiments are performed. 
```bash
├── data_report 'Figure and data for report'
│   ├── data_delay 'Show the effect of computational delay'
│   ├── data_discontinuity 'Show discontinuity of reference angle'
│   ├── data_integration 'Show discontinuity of integration function'
│   ├── data_validation 'Validate gyro model'
├── reward_funciton 'Test different reward functions'
│   ├── reward_int_v0 'test reward functions on GyroscopeIntegralEnv-v0'
│   ├── reward_int_v1 'test reward functions on GyroscopeIntegralEnv-v1'
│   ├── reward_quad_v0 'test reward functions on GyroscopeEnv-v0'
│   ├── reward_quad_v1 'test reward functions on GyroscopeEnv-v1'
│   ├── reward_sparse 'test sparse reward functions on GyroscopeEnv-v1'
├── simulation 'Initial simulation experiments'
│   ├── AgramFrame.png 'Figure of gyro'
│   └── gyroscope_simulation.ipynb 'Simulation coding and experiments'
├── ss_ananlysis_q_value 'Analysis the cause of steady state error from the view of Q function'
├── ss_ananlysis_return 'Analysis the cause of steady state error from the view of return'
├── test_gyro 'Implement and apply RL controller on real gyro'
├── test_implementation 'Test implementation of Gym environment'
├── test_set 'Generate a test set'
├── train_gyro 'Train RL controllers on gyro'
├── train_robust 'Train robust agents in virtual environment'
└── tune_hyper 'Hyperparameter searching' 
```

```python

```
