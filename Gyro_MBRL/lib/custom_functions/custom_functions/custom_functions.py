import gym
import numpy as np
import pandas as pd
import torch
from functools import partial
import matplotlib.pyplot as plt
import time

# To keep the angles between -lim and lim
def angle_normalize(x, lim = np.pi):
    return ((x + lim) % (2 * lim)) - lim 


# A callable function that builds a copy of the RL environment
def env_fn(env_name,simu_args={},reward_func='Quadratic',reward_args={}):
    env = gym.make(env_name)
    env.init(simu_args, reward_func, reward_args)
    return env


# Create an environment and initialize it to given state
def create_env(env_name,simu_args={},reward_func='Quadratic',reward_args={},state=None,param=None):
    env = gym.make(env_name)
    env.init(simu_args, reward_func, reward_args)
    env.reset(state)
    return env


# Load RL agent from pt file, or define a linearized controller
def load_agent(agent_path):
    if agent_path == 'linearized controller':
        agent = 'linearized controller'
    else:
        agent_fullpath = agent_path + '/pyt_save/model.pt'
        agent = torch.load(agent_fullpath)
    return agent


# Test the agent on a given enviroment
def test_agent(env,agent,t_end,w_seq=None,x1_ref_seq=None,x3_ref_seq=None):

    # Start storing
    time = np.arange(0, t_end, env.dt)
    state_record = np.zeros([time.shape[0], env.state.shape[0]])
    obs_record = np.zeros([time.shape[0], env.observation.shape[0]])
    action_record = np.zeros([time.shape[0], 2])
    reward_record = np.zeros([time.shape[0], 1])
    score = 0

    # Start simulation
    for i in range(len(time)):
        
        # Update state if references or w are not constant (i.e. a sequence)
        # state = env.state
        if x1_ref_seq:
            env.state[4] = x1_ref_seq[i]
        if x3_ref_seq:
            env.state[5] = x3_ref_seq[i]
        if w_seq:
            env.state[6] = w_seq[i]
        obs = env.observe()

        # Get action from agent
        if agent == 'linearized controller':
            action = lin_control(env.state, 3, 3, 3) # With poles at 5 like in Agram's report
        else:
            action = agent.act(torch.as_tensor(obs, dtype = torch.float32))
        
        # Perform step in environment
        obs, reward, done, info = env.step(action)
        
        # Store results
        state_record[i] = info['state']
        obs_record[i] = obs
        action_record[i] = action
        reward_record[i] = reward
        score += reward
#         if done:
#             break

    return score, state_record, obs_record, action_record, reward_record


# Plot position, velocity, tracking error, and motor imput
def plot_test(state_record, action_record, t_end, n=4):
    
    assert n in [1,2,3,4]
    
    time = np.linspace(0, t_end, len(state_record))
    
    f, axs = plt.subplots(n,2,figsize=(30,8*n))
    
    plt.subplot(n,2,1)
    plt.title('Red gimbal angle',fontsize=24)
    plt.xlabel('time [s]',fontsize=20)
    plt.ylabel(r'$\theta$ [rad]',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.plot(time,angle_normalize(state_record[:,0]),'r-')
    plt.plot(time,angle_normalize(state_record[:,4]),'g--')
    # plt.plot(time,np.full(len(time),180), 'k-')
    # plt.plot(time,np.full(len(time),-180), 'k-')

    plt.subplot(n,2,2)
    plt.title('Blue gimbal angle',fontsize=24)
    plt.xlabel('time [s]',fontsize=20)
    plt.ylabel(r'$\phi$ [rad]',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.plot(time,angle_normalize(state_record[:,2]),'b-')
    plt.plot(time,angle_normalize(state_record[:,5]),'g--')
    # plt.plot(time,np.full(len(time),180), 'k-')
    # plt.plot(time,np.full(len(time),-180), 'k-')
    
    if n > 1:
        plt.subplot(n,2,3)
        plt.title('Red gimbal speed',fontsize=24)
        plt.xlabel('time [s]',fontsize=20)
        plt.ylabel(r'$\dot \theta$ [rad/s]',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
        plt.plot(time,state_record[:,1],'r-')
        plt.plot(time,np.full(len(time),200/60*np.pi), 'k-')
        plt.plot(time,np.full(len(time),-200/60*np.pi), 'k-')

        plt.subplot(n,2,4)
        plt.title('Blue gimbal speed',fontsize=24)
        plt.xlabel('time [s]',fontsize=20)
        plt.ylabel(r'$\dot \phi$ [rad/s]',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
        plt.plot(time,state_record[:,3],'b-')
        plt.plot(time,np.full(len(time),200/60*np.pi), 'k-')
        plt.plot(time,np.full(len(time),-200/60*np.pi), 'k-')
    
    if n > 2:
        plt.subplot(n,2,5)
        plt.title('Red gimbal input',fontsize=24)
        plt.xlabel('time [s]',fontsize=20)
        plt.ylabel('voltage [V]',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
        plt.plot(time,action_record[:,0]*10,'r-')

        plt.subplot(n,2,6)
        plt.title('Blue gimbal input',fontsize=24)
        plt.xlabel('time [s]',fontsize=20)
        plt.ylabel('voltage [V]',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
        plt.plot(time,action_record[:,1]*10,'b-')

    if n > 3:
        plt.subplot(n,2,7)
        plt.title('Red gimbal tracking error',fontsize=24)
        plt.xlabel('time [s]',fontsize=20)
        plt.ylabel(r'$\theta$ error [rad]',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
        plt.plot(time,angle_normalize(state_record[:,0]-state_record[:,4]),'r-')

        plt.subplot(n,2,8)
        plt.title('Blue gimbal tracking error',fontsize=24)
        plt.xlabel('time [s]',fontsize=20)
        plt.ylabel(r'$\phi$ error [rad]',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
        plt.plot(time,angle_normalize(state_record[:,2]-state_record[:,5]),'b-')

    plt.show()

    return None


# Evaluate an agent for a given environment, generate a metrix
def evaluate_control(env,agent,agent_name,t_end,ss_bound,num_exp=200,states=None,print_unsteady=False):
    
    # Experiments setup
    df_header = ['Config.','$\theta$ MAE (rad)','$\phi$ MAE (rad)',
                 '$\theta$ MSSE (rad)','$\phi$ MSSE (rad)',
                 '$\theta$ in bounds (%)','$\phi$ in bounds (%)',
                 '$\theta$ unsteady (%)','$\phi$ unsteady (%)',
                 '$\theta$ rise time (s)','$\phi$ rise time (s)',
                 '$\theta$ settling time (s)','$\phi$ settling time (s)',
                 'u1 (V)','u2 (V)','u1 variation (V)','u2 variation (V)','Convergence time (min)']
    num_metrics = len(df_header)-1

    # Initialize mean metrics
    all_metrics = np.zeros((num_metrics,num_exp))

    # Experiments rollout
    for i in range(num_exp):
        
        s = time.time()
        
        # Generate random init state
        # seed = np.random.randint(0,20)
        # env.seed(seed)
        if states is None:
            env.reset()
        else:
            env.reset(states[i])
        initial_state = env.state
        # Run trial
        score, state_record, obs_record, action_record, reward_record = test_agent(env,agent,t_end)
#         print(state_record[-1])
        # Evaluate control metrics for that run
        # Don't forget that u are normalized
        metrics = control_metrics(score,state_record,action_record*env.maxVoltage,reward_record,t_end,ss_bound)

        # Put in all_metrics table
        for idx in range(len(metrics)):
            all_metrics[idx,i] = metrics[idx]
        
        # if unstable
        if (print_unsteady and metrics[6] + metrics[7] > 0):
            print(initial_state)
            
        print('test: ', i, ', takes: ', time.time() - s)
        
    # Extract metrics from logger
    # log = pd.read_csv(agent_logpath, sep="\t")
    conv_time = 0 # log['Time'].iloc[-1]/60

    # Compute mean of metrics arrays
    mean_metrics = {df_header[0]:[agent_name], # 'Config.'
                    df_header[1]:[np.mean(all_metrics[0,:])], # '$\theta$ MAE (rad)'
                    df_header[2]:[np.mean(all_metrics[1,:])], # '$\phi$ MAE (rad)'
                    df_header[3]:[np.nanmean(all_metrics[2,:])], # '$\theta$ MSSE (rad)'
                    df_header[4]:[np.nanmean(all_metrics[3,:])], # '$\phi$ MSSE (rad)'
                    df_header[5]:[(100*np.mean(all_metrics[4,:]))], # '$\theta$ in bounds (%)'
                    df_header[6]:[(100*np.mean(all_metrics[5,:]))], # '$\phi$ in bounds (%)'
                    df_header[7]:[(100*np.mean(all_metrics[6,:]))], # '$\theta$ unsteady (%)'
                    df_header[8]:[(100*np.mean(all_metrics[7,:]))], # '$\phi$ unsteady (%)',
                    df_header[9]:[np.nanmean(all_metrics[8,:])], # '$\theta$ rise time (s)'
                    df_header[10]:[np.nanmean(all_metrics[9,:])], # '$\phi$ rise time (s)'
                    df_header[11]:[np.nanmean(all_metrics[10,:])], # '$\theta$ settling time (s)'
                    df_header[12]:[np.nanmean(all_metrics[11,:])], # '$\phi$ settling time (s)'
                    df_header[13]:[np.mean(all_metrics[12,:])], # 'u1 (V)'
                    df_header[14]:[np.mean(all_metrics[13,:])], # 'u2 (V)'
                    df_header[15]:[np.mean(all_metrics[14,:])], # 'u1 variation (V)'
                    df_header[16]:[np.mean(all_metrics[15,:])], # 'u2 variation (V)'
                    df_header[17]:[conv_time]} # 'Convergence time (min)'

    # Put in panda dataframe
    df = pd.DataFrame(mean_metrics, columns = df_header)
    df.set_index('Config.', inplace=True)

    return df


# Compute the performance matrics, action_record has already been multiplied by 10
def control_metrics(score,state_record,action_record,reward_record,t_end,ss_bound):
        
    time = np.linspace(0, t_end, len(state_record))

    # Check convergence to steady state
    conv_thresh = 0.025
    x1_diff = np.abs(np.diff(state_record[:,0]))
    x3_diff = np.abs(np.diff(state_record[:,2]))
    x1_conv = np.all(x1_diff[int(0.75*len(x1_diff)):] <= conv_thresh)
    x3_conv = np.all(x3_diff[int(0.75*len(x3_diff)):] <= conv_thresh)
    
    # Initialize values that are dependent of convergence to steady state
    x1_in_bound,x3_in_bound = 0,0
    x1_msse,x3_msse = np.nan,np.nan # use nans and then np.nanmean to ignore the values not in bounds
    x1_us,x3_us = 1,1
    x1_rise,x1_settle,x3_rise,x3_settle = np.nan,np.nan,np.nan,np.nan
    
    # Check convergence on x1
    if x1_conv:
        # Since it does converge, replace true x1 ss error and put unsteady to 0
        x1_msse = np.abs(angle_normalize(state_record[-1,0]-state_record[-1,4]))
        x1_us = 0
        # Check if x1 ss error is in bounds, otherwise no use in computing the metrics below
        if x1_msse <= ss_bound:
            x1_in_bound = 1
            x1_rise,x1_settle = step_info(time,state_record[:,0],state_record[0,4],ss_bound)

    # Check convergence on x3
    if x3_conv:
        # Since it does converge, replace true x3 ss error and put unsteady to 0
        x3_msse = np.abs(angle_normalize(state_record[-1,2]-state_record[-1,5]))
        x3_us = 0
        # Check if x3 ss error is in bounds, otherwise no use in computing the metrics below
        if x3_msse <= ss_bound:
            x3_in_bound = 1
            x3_rise,x3_settle = step_info(time,state_record[:,2],state_record[0,5],ss_bound)

    # '$\theta$ MAE (rad)','$\phi$ MAE (rad)'
    x1_mae = np.mean(np.abs(angle_normalize(state_record[:,0]-state_record[:,4])))
    x3_mae = np.mean(np.abs(angle_normalize(state_record[:,2]-state_record[:,5])))
        
    # 'u1 (V)','u2 (V)','u1 variation (V)','u2 variation (V)'
    u1_mean = np.mean(np.abs(action_record[:,0]))
    u2_mean = np.mean(np.abs(action_record[:,1]))
    u1_var = np.mean(np.abs(np.diff(action_record[:,0])))
    u2_var = np.mean(np.abs(np.diff(action_record[:,1])))

    metrics = [x1_mae,x3_mae,x1_msse,x3_msse,
               x1_in_bound,x3_in_bound,x1_us,x3_us,
               x1_rise,x3_rise,x1_settle,x3_settle,
               u1_mean,u2_mean,u1_var,u2_var]
    
    return metrics


# Compute rising time and settling time in a test, x is array and x_ref is constant
def step_info(time,x,x_ref,ss_bound):
    
    # Rising time: starting from beginning, first to reach 0.9 of steady state value
    idx_rise = np.nan
    for i in range(0,len(x)-1):
        if abs(angle_normalize(x[i])-angle_normalize(x[0]))/abs(angle_normalize(x[-1])-angle_normalize(x[0]))>0.9:
            idx_rise = i
            break
    if idx_rise is np.nan:
        t_rise = np.nan
    else:
        t_rise = time[idx_rise]-time[0]

    # Settling time: starting from the end, first to be out of bounds
    idx_set = np.nan
    for i in range(2,len(x)-1):
        if abs(angle_normalize(x[-i]-x_ref))>ss_bound:
            idx_set = len(x)-i
            break
    if idx_set is np.nan:
        t_set = np.nan
    else:
        t_set = time[idx_set]-time[0]

    return t_rise,t_set

# Read training data from txt file
def read_progress(agent_path):
    
    # count lines
    file = open(agent_path+"/progress.txt", "r")
    count = len(file.readlines())
    data = np.empty([count-1, 23])
    file.seek(0)
    
    # read each line as a numpy array
    for row, x in enumerate(file):
        if row == 0:
            continue
        data[row-1] = np.array(x.split('\t')).astype(np.float)
    file.close()
    
    return data

# Linearized controller
def lin_control(x,sig_th,sig_phi,sig_psid):

    # Constants from the controller
    fvr = 0.002679
    fvb = 0.005308
    Jbx1 = 0.0019
    Jbx2 = 0.0008
    Jbx3 = 0.0012
    Jrx1 = 0.0179
    Jdx1 = 0.0028
    Jdx3 = 0.0056
    Kamp = 0.5
    Ktorque = 0.0704
    eff = 0.86
    nRed = 1.5
    nBlue = 1
    KtotRed = Kamp*Ktorque*eff*nRed
    KtotBlue = Kamp*Ktorque*eff*nBlue

    # Extract states
    x1,x2,x3,x4,x1_ref,x3_ref,x6 = x[:7]

    # Speed refs at 0
    x2_ref,x4_ref,x6_ref = 0,0,x6

    # Controls
    T1 = 0.5*((Jbx1+Jbx3+Jdx1+Jdx3+2*Jrx1+Jbx1*np.cos(2*x3)-Jbx3*np.cos(2*x3)+Jdx1*np.cos(2*x3)-Jdx3*np.cos(2*x3))*sig_th**2*x1_ref
              +2*x6_ref*Jdx3*sig_psid*np.sin(x3)-sig_th**2*(Jbx1+Jbx3+Jdx1+Jdx3+2*Jrx1+(Jbx1-Jbx3+Jdx1-Jdx3)*np.cos(2*x3))*x1
              +2*x2*(fvr-Jbx1*sig_th-Jbx3*sig_th-Jdx1*sig_th-Jdx3*sig_th-2*Jrx1*sig_th-(Jbx1-Jbx3+Jdx1-Jdx3)*sig_th*np.cos(2*x3)-(Jbx1-Jbx3+Jdx1-Jdx3)*np.sin(2*x3)*x4)
              -2*Jdx3*sig_psid*np.sin(x3)*x6+2*Jdx3*np.cos(x3)*x4*x6)
    T2 = -(Jbx2+Jdx1)*(-sig_phi*(x3_ref*sig_phi-sig_phi*x3-2*x4)+((-Jbx1+Jbx3-Jdx1+Jdx3)*np.cos(x3)*np.sin(x3)*x2*x2-fvb*x4+Jdx3*np.cos(x3)*x2*x6)/(Jbx2+Jdx1))

    # The above T1 and T2 are desired torque
    # Action (normalized to range [-1,1])
    action = np.array([T1/(10*KtotRed),T2/(10*KtotBlue)])

    return action


# Generate a prbs signal of order n
def prbs(n, p):
    
    # n: shift register length
    # p: number of periods in signal

    u = np.ones([2**n-1]).astype(int)
    q = np.array([0,1,2,1,2,1,1,0,4,3]);

    assert n <= 10, 'Maximum allowable value for n is 10'
    assert n >= 2, 'Minimum allowable value for n is 2'

    if n == 8:
        for i in range(n,2**n-1):
            u[i] = (u[i-n]^u[i-n+1])^(u[i-n+2]^u[i-n+7])
    else:
        for i in range(n,2**n-1):
            u[i] = u[i-n]^u[i-n+q[n-1]]

    u = 2*u - 1
    u = np.tile(u, p)

    return u