import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from scipy.integrate import solve_ivp

class GyroscopeDiscontinuousEnv(gym.Env):


    """
    GyroscopeDiscontinuousEnv is a double gimbal control moment gyroscope (DGCMG) with 2 input voltage u1 and u2
    on the two gimbals, and disk speed assumed constant (parameter w). Simulation is based on the
    Quanser 3-DOF gyroscope setup.

    Here, observation = state, meaning that the angles (normalized between -pi and pi)
    are fed to the ANN and not the cos and sin of the angles. This causes a discontinuity at -pi/pi where the boundary
    should normally be circular.


    **STATE:**
    The state consists of the angle and angular speed of the outer red gimbal (theta = x1, thetadot = x2),
    the angle and angular speed of the inner blue gimbal (phi = x3, phidot = x4), the reference
    for tracking on theta and phi (x1_ref and x3_ref), and the disk speed (disk speed = w):

    state = [x1, x2, x3, x4, x1_ref, x3_ref, w]

    **OBSERVATION:**

    observation = state (7 dimensions)

    **ACTIONS:**
    The actions are the input voltage to create the red and blue gimbal torque (red voltage = u1, blue voltage = u2),
    and are continuous in a range of -10 and 10V:

    action = [u1,u2]

    """


    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):

        # Inertias in Kg*m2
        self.Jbx1 = 0.0019
        self.Jbx2 = 0.0008
        self.Jbx3 = 0.0012
        self.Jrx1 = 0.0179
        self.Jdx1 = 0.0028
        self.Jdx3 = 0.0056

        # Combined inertias
        self.J1 = self.Jbx1 - self.Jbx3 + self.Jdx1 - self.Jdx3
        self.J2 = self.Jbx1 + self.Jdx1 + self.Jrx1
        self.J3 = self.Jbx2 + self.Jdx1

        # Motor constants
        self.Kamp = 0.5 # A/V
        self.Ktorque = 0.0704 # Nm/A
        self.eff = 0.86
        self.nRed = 1.5
        self.nBlue = 1
        self.KtotRed = self.Kamp*self.Ktorque*self.eff*self.nRed
        self.KtotBlue = self.Kamp*self.Ktorque*self.eff*self.nBlue

        # Time step in s
        self.dt = 0.05
        self.eval_per_dt = 5

        # Reward function by default for SpiningUp functions that cannot init using args_int
        self.reward_func = quad_reward
        self.reward_args = {'qx1':9,'qx2':0.04,'qx3':9,'qx4':0.04,'pu1':0.01,'pu2':0.01}

        # Action space
        self.maxVoltage = 10 # V
        self.highAct = np.array([self.maxVoltage,self.maxVoltage])
        self.action_space =  spaces.Box(low = -self.highAct, high = self.highAct, dtype=np.float32)

        # State and observation space
        self.maxSpeed = 100 * 2 * np.pi / 60
        self.maxAngle = np.pi
        self.maxdiskSpeed = 300 * 2 * np.pi / 60
        self.highState = np.array([self.maxAngle,self.maxSpeed,self.maxAngle,self.maxSpeed,self.maxAngle,self.maxAngle,self.maxdiskSpeed])
        self.highObs = self.highState # np.array([1.0, 1.0, self.maxSpeed, 1.0, 1.0, self.maxSpeed, self.maxAngle, self.maxAngle, self.maxdiskSpeed])
        self.observation_space = spaces.Box(low = -self.highObs, high = self.highObs, dtype=np.float32)

        # Seed for random number generation
        self.seed()

        self.viewer = None

    # method used to try different variants of the env
    def args_init(self, reward_type, reward_args, ep_len, is_noise=False):
        # Reward function type
        reward_dict = {
            'Quadratic': quad_reward,
            'Absolute': abs_reward,
            'Normalized': norm_reward
        }
        self.reward_func = reward_dict[reward_type]
        self.reward_args = reward_args

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def step(self,u):
        x1, x2, x3, x4, x1_ref, x3_ref, w = self.state
        u1,u2 = u

        # Simulate
        results = solve_ivp(fun = dxdt, t_span = (0, self.dt), y0 = [x1,x2,x3,x4], method='RK45',  t_eval = np.linspace(0,self.dt,self.eval_per_dt), args=(u1,u2,self))

        # for rendering
        x1_eval = angle_normalize_pi(results.y[0])
        x2_eval = np.clip(results.y[1],-self.maxSpeed,self.maxSpeed)
        x3_eval = angle_normalize_pi(results.y[2])
        x4_eval = np.clip(results.y[3],-self.maxSpeed,self.maxSpeed)

        # for state
        x1 = x1_eval[-1]
        x2 = x2_eval[-1]
        x3 = x3_eval[-1]
        x4 = x4_eval[-1]

        self.state = np.asarray([x1,x2,x3,x4,x1_ref, x3_ref,w])

        # Angle error (normalized between pi and -pi this time to get smallest distance)
        diff_x1 = angle_normalize_pi(x1 - x1_ref)
        diff_x3 = angle_normalize_pi(x3 - x3_ref)

        # Reward
        reward = self.reward_func(diff_x1, diff_x3, x2, x4, u1, u2, **self.reward_args)

        return (self._get_ob(), reward, False, {'state':self.state,'x1_eval':x1_eval,'x2_eval':x2_eval,'x3_eval':x3_eval,'x4_eval':x4_eval,'x1_ref_eval':np.full(self.eval_per_dt,x1_ref),'x3_ref_eval':np.full(self.eval_per_dt,x3_ref)})

    def _get_ob(self):
        s = self.state
        return s # np.array([np.cos(s[0]), np.sin(s[0]), s[1], np.cos(s[2]), np.sin(s[2]), s[3], s[4], s[5], s[6]])


    def reset(self):
        self.state = self.np_random.uniform(low=-self.highState, high=self.highState)
        return self._get_ob()

    # cannot be implemented in reset() in gym API, need to have separate method
    def reset2state(self, state):
        self.state = state
        return self._get_ob()

    def access_members(self):
        return self.dt,self.eval_per_dt,self.maxVoltage


    def render(self, mode='human'):
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def dxdt(t, x, u1, u2, gyro):

        # Rewrite constants shorter
        J1 = gyro.J1
        J2 = gyro.J2
        J3 = gyro.J3
        Jdx3 = gyro.Jdx3
        KtotRed = gyro.KtotRed
        KtotBlue = gyro.KtotBlue
        w = x[-1]

        # Convert input voltage to input torque
        u1,u2 = KtotRed*u1, KtotBlue*u2

        # Equations of motion
        dx_dt = [0, 0, 0, 0]
        dx_dt[0] = x[1]
        dx_dt[1] = (u1+J1*np.sin(2*x[2])*x[1]*x[3]-Jdx3*np.cos(x[2])*x[3]*w)/(J2 + J1*np.power(np.sin(x[2]),2))
        dx_dt[2] = x[3]
        dx_dt[3] = (u2 - J1*np.cos(x[2])*np.sin(x[2])*np.power(x[1],2)+Jdx3*np.cos(x[2])*x[1]*w)/J3
        return dx_dt

def quad_reward(diff_x1, diff_x3, x2, x4, u1, u2, qx1 = 1, qx2 = 0.01, qx3 = 1, qx4 = 0.01, pu1 = 0, pu2 = 0):
    return -(qx1*(diff_x1**2) + qx3*(diff_x3**2) + qx2*(x2**2) + qx4*(x4**2) + pu1*(u1**2) + pu2*(u2**2))

def abs_reward(diff_x1, diff_x3, x2, x4, u1, u2, qx1 = 1, qx2 = 0.01, qx3 = 1, qx4 = 0.01, pu1 = 0, pu2 = 0):
    return -(qx1*abs(diff_x1) + qx3*abs(diff_x3) + qx2*abs(x2) + qx4*abs(x4) + pu1*abs(u1) + pu2*abs(u2))

def norm_reward(diff_x1, diff_x3, x2, x4, u1, u2, k = 0.2):
    return -((abs(diff_x1)/k)/(1+ (abs(diff_x1)/k)) + (abs(diff_x3)/k)/(1+ (abs(diff_x3)/k)))

def angle_normalize_2pi(x):
        return (((x+2*np.pi) % (4*np.pi)) - 2*np.pi) # To keep the angles between -2pi and 2pi

def angle_normalize_pi(x):
        return (((x+np.pi) % (2*np.pi)) - np.pi) # To keep the angles between -pi and pi
