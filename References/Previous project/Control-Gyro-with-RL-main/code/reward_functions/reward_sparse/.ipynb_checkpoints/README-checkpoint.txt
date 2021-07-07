Test sparse reward functions on GyroscopeEnvV1

1.  For simply sparse reward functions, there is no gredient in the beginning. 
    The exploration can be extremely slow, so try to add an extra exponential term to speed up training
    Result: training is dominated by exponential term, but sparse part helps to improve the result
    
    agent_paths = ['ddpg_e10','ddpg_se_xe10b01','ddpg_se_xe10b001','ddpg_se_xe10b001_mher','ddpg_se_xe10b001_vher']

2.  To speed up training for merely sparse reward functions, try to continue training based on an existing agent.
    Not working, the return drops to 0 dramasticly. The sparse itself is not able to generate new stable trajectory

    Try to use HER algorithm, 2 ways to implement
    HER: only use HER when trajectory is stable
    HER2: use HER after each step, like in the HER paper
    Result: HER without pre-trained weight is promising.
    
    agent_paths = ['ddpg_s_b01_mher','ddpg_s_b01_mher_ac','ddpg_s_b01_vher','ddpg_s_b01_vher_ac']

3.  Add a sparse reward for velocity
    
    oscillating of position
    agent_paths = ['ddpg_s_b01v_mher','ddpg_s_b01v_mher_ac','ddpg_s_b01v_vher','ddpg_s_b01v_vher_ac']
    
4. 'ddpg_se_xe10b001_mher' is best choice.
    an alternative sparse reward is also tested: reward*(x1_diff<bound) + reward*(x3_diff<bound), leads to high steady state error
