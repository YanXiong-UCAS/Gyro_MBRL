Analysis the cause of steady state error with quadratic reward function. 

Also show that integral term doesn't help.

The main idea is from the paper 'On-line RL for nonlinear motion control: quadratic and non-quadaric reward functions'.

The idea is problematic. It does prove that the agent prefer a fast trajectory, but doesn't explain why the steay state remains in the end. Also, in DDPG, the idea of 'return' is not used, Q-learning is used instead. 

A more reasonable explaination can be found in 'ss_analysis-part2'