'''
OM-1: 外部方形框架固定，RL直接控制red Gimbal, Blue Gimbal和Golden Disk；
OM-2: Red Gimbal locked, RL直接控制Golden Disk和Blue Gimbal,进而间接控制外部方形框架转角，该工况为欠驱动；
Difference with other GyroscopeEnv:
-- Add a golden disk as a control object for RL
-- 后续测试的时候采用“正弦跟踪曲线”+“阶跃跟踪曲线”

Write by Yan
2021/10/13
'''

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from scipy.integrate import solve_ivp


class GyroscopeNewEnvV0(gym.Env):
    """
    GyroscopeEnv:
        GyroscopeEnv is a GYM environment for Quanser 3-DOF gyroscope. The gyroscope consists of a disk mounted
        inside an inner gimbal which in turn is mounted inside an outer gimbal.
        The <<<two gimbals and golden disk>>> are controlled by a RL controller.

    State:   增加“golden disk”的转速作为RL的控制对象
        state = [x1, x2, x3, x4, x1_ref, x3_ref, w, w_a, w_ref]
        (7 dimensions >>> 9 dimensions)  增加golden disk角加速度，以及转速目标值

        Outer red gimbal:
            x1, or theta: angular position [rad]   被控对象1 -- 转角
            x2, or dot(theta): angular velocity [rad/s]   被控对象1的转速
            x1_ref: angular position reference [rad]   被控对象1的转角目标值
            u1: motor voltage [V]   被控对象1的电压值
        Inner blue gimbal:
            x3, or phi: angular position [rad]   被控对象2 -- 转角
            x4, or dot(phi): angular velocity [rad/s]   被空对象2的转速
            x3_ref: angular position reference [rad]   被控对象2的转角目标值
            u2: motor voltage [V]   被控对象2的电压值
        Golden disk:
            w: angular velocity [rad/s]   被控对象3 -- 转速
            w_a: angular acceleration [rad/s2]   被控对象3的角加速度
            w_ref: angular velocity [rad/s]   被控对象3的转速目标值
            u3: motor voltage [V]   被控对象3的电压值
        Mechanical constraints: 阈值
            motor voltage: [-10, 10] [V]
            gimbal velocity: [-100, 100] [rpm]
            disk velocity: [-300, 300] [rpm]

    Observation:   增加“golden disk”的转速目标值到观测空间
        observation = [cos(x1), sin(x1), x2, cos(x3), sin(x3), x4, x1_ref, x3_ref, w, w_a, w_ref]
        (9 dimensions >>> 11 dimensions)

        The angles have been replaced with their cosine and sine to prevent the discontinuity at -pi and pi.
        The observation space is thus larger than the state space.

    Action:   增加“golden disk”的转速，即控制“golden disk”的电机转速的电压作为RL的控制对象
        action = [a1, a2, a3]
        Note: a1, a2, a3 are normalized voltages
              u1, u2, u3 = 10*a1, 10*a2, 10*a3 are actual voltages
              T1, T2, T3 = KtotRed*u1, KtotBlue*u2, KtotRed*u3 are motor torques   金色圆盘的转矩需要吗？？？仔细检查后续的程序后作出最终判断

    Initialization:
        Some versions of Gym may not support initialization with arguments, so initialize it manully with:
        # create env
        env = GyroscopeEnv()
        env.init(simu_args = simu_args, reward_func = reward_func, reward_args = reward_args)
        # simu_args, with optional simulation step (dt), episode length (ep_len), and random seed (seed)
        simu_args = {'dt': 0.05, 'ep_len': 100, 'seed': 2， ‘friction’: False}
        # reward_func, optional reward function, default value is 'Quadratic'
        reward_func = 'Quadratic'
        # reward_args, optional reward parameters
        reward_args = {'qx1': 1, 'qx2': 0.01, 'qx3': 1, 'qx4': 0.01, 'pu1': 0, 'pu2': 0}   是否需要增加参数数量，检查后最初最终判断！
        reward_args = {'qx1': 1, 'qx2': 0.01, 'qx3': 1, 'qx4': 0.01, 'pu1': 0, 'pu2': 0, 'qw': 1, 'qwa': 0.01, 'pu3': 0}
        仔细检查判断上述修订是否正确
    """

    # ---------------------------------------------------------------------------------------------------- #
    # ------------------------------------------ Initialization ------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------- #

    def init(self, simu_args={}, reward_func='Quadratic', reward_args={}):

        # Initialize mechanical parameters of the gyroscope
        self.init_gyro()

        # Initialize simulation parameters
        self.init_simu(**simu_args)

        # Initialize reward parameters
        self.init_reward(reward_func, reward_args)

        # State space, 7D  对应转换成9D
        self.state_bound = np.array([self.maxAngle, self.maxGimbalSpeed, self.maxAngle, self.maxGimbalSpeed,
                                     self.maxAngle, self.maxAngle, self.maxDiskSpeed, self.maxDiskSpeed], dtype=np.float32)
        self.state_space = spaces.Box(low=-self.state_bound, high=self.state_bound, dtype=np.float32)

        # Observation space (normalized), 9D   对应转换成11D
        self.observation_bound = np.array([1.0] * 11, dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.observation_bound, high=self.observation_bound,
                                            dtype=np.float32)

        # Action space (normalized), 2D   对应转换3D
        self.action_bound = np.array([1.0] * 3, dtype=np.float32)
        self.action_space = spaces.Box(low=-self.action_bound, high=self.action_bound, dtype=np.float32)

    # Initialize fixed parameters of the gyroscope
    def init_gyro(self):

        # Inertias in Kg*m2, from SP report page 23, table 2
        self.Jrx1 = 0.0179
        self.Jbx1 = 0.0019
        self.Jbx2 = 0.0008
        self.Jbx3 = 0.0012
        self.Jdx1 = 0.0028
        self.Jdx2 = 0.0056
        self.Jdx3 = 0.0056

        # Combined inertias to simplify equations, from SP report page 22, state space equations
        self.J1 = self.Jbx1 - self.Jbx3 + self.Jdx1 - self.Jdx3
        self.J2 = self.Jbx1 + self.Jdx1 + self.Jrx1
        self.J3 = self.Jbx2 + self.Jdx1

        # Motor constants, from SP report page 23, table 1
        self.Kamp = 0.5  # current gain, A/V
        self.Ktorque = 0.0704  # motor gain, Nm/A
        self.eff = 0.86  # motor efficiency
        self.nRed = 1.5  # red gearbox ratio
        self.nBlue = 1  # blue gearbox ratio
        self.nDisk = 1 # disk gearbox ratio
        self.KtotRed = self.Kamp * self.Ktorque * self.eff * self.nRed  # Nm/V   Red Gimbal
        self.KtotBlue = self.Kamp * self.Ktorque * self.eff * self.nBlue  # Nm/V   Blue Gimbal
        self.KtotDisk = self.Kamp * self.Ktorque * self.eff * self.nDisk  # Nm/V   Golden Disk

        # Mechanical constraints
        self.maxVoltage = 10  # V
        self.maxAngle = np.pi  # rad
        self.maxGimbalSpeed = 100 * 2 * np.pi / 60  # rad/s
        self.maxDiskSpeed = 300 * 2 * np.pi / 60  # rad/s
        self.maxDiskAcceleration = 300 * 2 * np.pi / 60  # rad/s2   加速度的上限应该是多少呢？？？

    # Initialize simulation parameters
    def init_simu(self, dt=0.05, ep_len=100, seed=2, friction=False):

        # Gyroscope state and observation
        self.state = np.array([0] * 8)   # 转换成8D
        self.observe()

        # Time step in s
        self.dt = dt
        self.eval_per_dt = int(dt / 0.01)  # run evaluation every 0.01s

        # Episode length and current episode
        self.ep_len = ep_len
        self.ep_cur = 0

        # Seed for random number generation
        self.seed(seed)
        self.viewer = None

        # Friction
        self.fvr = 0.002679 if friction else 0
        self.fcr = 0
        self.fvb = 0.005308 if friction else 0
        self.fcb = 0

    # Initialize reward parameters
    def init_reward(self, reward_func, reward_args):

        reward_dict = {
            # continuous reward functions, part one
            'Quadratic': self.quad_reward,
            'Quadratic with bonus': self.quad_bon_reward,
            'Quadratic with exponential': self.quad_exp_reward,
            'Quadratic with ending penalty': self.quad_end_pen_reward,
            'Quadratic with penalty': self.quad_pen_reward,
            'Absolute': self.abs_reward,
            'Normalized': self.norm_reward,
            'Normalized with bonus': self.norm_bon_reward,

            # continuous reward functions, part two
            'Power': self.power_reward,
            'Exponential': self.exp_reward,
            'PE': self.power_exp_reward,
            'PE_new_V0': self.power_exp_reward_new_V0,   # 根据新环境设置的奖励函数！！！等待验证可行后，考虑增加其他奖励函数的形式进行对比验证！！！

            # sparse reward functions
            'Sparse': self.sparse_reward,
            'Sparse with exp': self.sparse_reward_with_exp,
            'Sparse with exp 2': self.sparse_reward_with_exp_2
        }
        if reward_func in ['Sparse']:  # 'Sparse with exp'
            self.sparse = True
        else:
            self.sparse = False
        self.reward_func = reward_dict[reward_func]
        self.reward_args = reward_args

    # ---------------------------------------------------------------------------------------------------- #
    # ----------------------------------------------- Step ----------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------- #

    # Simulate the environment fot one step dt
    def step(self, a):

        # extract states and actions   对应增加golden disk转速的控制
        x1, x2, x3, x4, x1_ref, x3_ref, w, w_ref = self.state
        a1, a2, a3 = a
        u1, u2, u3 = self.maxVoltage * a1, self.maxVoltage * a2, self.maxVoltage * a3

        # Increment episode
        self.ep_cur += 1

        # For quad_end_pen_reward, check if terminal state is reached
        if self.reward_func == self.quad_end_pen_reward and self.ep_cur == self.ep_len:
            self.reward_args['end_horizon'] = 1

        # run simulation for a step
        results = solve_ivp(
            fun=self.dxdt,
            t_span=(0, self.dt),  # solver starts with t = 0 and integrates until it reaches t = self.dt
            y0=[x1, x2, x3, x4, w],  # initial state   初始化状态空间
            method='RK45',
            t_eval=np.linspace(0, self.dt, self.eval_per_dt),  # times at which to store the computed solution
            args=(u1, u2, u3)   # RL控制器对应三个电压输出值
        )

        # evaluated states, each contains eval_per_dt points
        x1_eval = results.y[0]
        x2_eval = results.y[1]
        x3_eval = results.y[2]
        x4_eval = results.y[3]
        w_eval = results.y[4]

        # change in velocity, or acceleration   速度/加速度的变化
        dx2 = x2_eval[-1] - x2
        dx4 = x4_eval[-1] - x4
        dw = w_eval[-1] - w


        # keep only the last evaluation value   只保留最后的评估值
        x1 = x1_eval[-1]
        x2 = x2_eval[-1]
        x3 = x3_eval[-1]
        x4 = x4_eval[-1]
        w = w_eval[-1]


        # Angle error (normalized between pi and -pi to get smallest distance)
        x1_diff = self.angle_normalize(x1 - x1_ref)
        x3_diff = self.angle_normalize(x3 - x3_ref)
        w_diff = self.diskspeed_normalize(w - w_ref)

        # update state and observation
        self.state = np.array([x1, x2, x3, x4, x1_ref, x3_ref, w, w_ref])   # 7D >>> 8D
        self.observe()

        # Reward(float), normalized everything in advance
        reward = self.reward_func(x1_diff / self.maxAngle, x3_diff / self.maxAngle, w_diff / self.maxDiskSpeed,
                                  x2 / self.maxGimbalSpeed, x4 / self.maxGimbalSpeed, w / self.maxDiskSpeed,
                                  dx2 / self.maxGimbalSpeed, dx4 / self.maxGimbalSpeed, dw / self.maxDiskSpeed,
                                  a1, a2, a3, **self.reward_args)   # 对应后续的奖励函数的“自变量排列顺序”也需要进行重新的编排！

        # Done(bool): whether it’s time to reset the environment again.
        if self.sparse:
            # in sparse reward functions, terminate the episode when the speed is too large
            # otherwise the exploration will happen mainly in high speed area, which is not desired
            done = self.ep_cur > self.ep_len or x2 > 2 * self.maxGimbalSpeed or x4 > 2 * self.maxGimbalSpeed or w > self.maxDiskSpeed
        else:
            # in other reward functions, terminating the episode early will encourage the agent to
            # speed up the gyroscope and end the episode, because the reward is negative
            done = self.ep_cur > self.ep_len

        # Info(dict): diagnostic information useful for debugging.
        info = {'state': self.state, 'observation': self.observation}

        return self.observation, reward, done, info

    # Compute the derivative of the state, here u is NOT normalized
    def dxdt(self, t, x, u1, u2, u3):

        J1, J2, J3, Jdx3 = self.J1, self.J2, self.J3, self.Jdx3
        w = self.state[-1]

        # Convert input voltage to input torque
        T1, T2 = self.KtotRed * u1, self.KtotBlue * u2

        # Friction
        T1 = T1 - self.fvr * x[1] - self.fcr * np.sign(x[1])
        T2 = T2 - self.fvb * x[3] - self.fcb * np.sign(x[3])

        # Equations of motion
        dx_dt = [0, 0, 0, 0]
        dx_dt[0] = x[1]
        dx_dt[1] = (T1 + J1 * np.sin(2 * x[2]) * x[1] * x[3] - Jdx3 * np.cos(x[2]) * x[3] * w) / (
                    J2 + J1 * np.power(np.sin(x[2]), 2))
        dx_dt[2] = x[3]
        dx_dt[3] = (T2 - J1 * np.cos(x[2]) * np.sin(x[2]) * np.power(x[1], 2) + Jdx3 * np.cos(x[2]) * x[1] * w) / J3

        return dx_dt

    # ---------------------------------------------------------------------------------------------------- #
    # ------------------------------------------ Reward Part I ------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------- #

    def abs_reward(self, x1_diff, x3_diff, x2, x4, dx2, dx4, u1, u2, qx1=1, qx2=0.01, qx3=1, qx4=0.01, pu1=0, pu2=0):
        return -(qx1 * abs(x1_diff) + qx3 * abs(x3_diff) + qx2 * abs(x2) + qx4 * abs(x4) + pu1 * abs(u1) + pu2 * abs(
            u2))

    def norm_reward(self, x1_diff, x3_diff, x2, x4, dx2, dx4, u1, u2, k=0.2, qx2=0, qx4=0, pu1=0, pu2=0):
        return -((abs(x1_diff) / k) / (1 + (abs(x1_diff) / k)) + (abs(x3_diff) / k) / (1 + (abs(x3_diff) / k)) + qx2 * (
                    x2 ** 2) + qx4 * (x4 ** 2) + pu1 * (u1 ** 2) + pu2 * (u2 ** 2))

    def norm_bon_reward(self, x1_diff, x3_diff, x2, x4, dx2, dx4, u1, u2, k=0.2, qx2=0, qx4=0, pu1=0, pu2=0,
                        bound=0.001, bonus=1):
        return -((abs(x1_diff) / k) / (1 + (abs(x1_diff) / k)) + (abs(x3_diff) / k) / (1 + (abs(x3_diff) / k)) + qx2 * (
                    x2 ** 2) + qx4 * (x4 ** 2) + pu1 * (u1 ** 2) + pu2 * (u2 ** 2)) + bonus * (
                           abs(x1_diff) <= bound or abs(x3_diff) <= bound)

    def quad_reward(self, x1_diff, x3_diff, x2, x4, dx2, dx4, u1, u2, qx1=1, qx2=0.01, qx3=1, qx4=0.01, pu1=0, pu2=0):
        return -(qx1 * (x1_diff ** 2) + qx3 * (x3_diff ** 2) + qx2 * (x2 ** 2) + qx4 * (x4 ** 2) + pu1 * (
                    u1 ** 2) + pu2 * (u2 ** 2))

    def quad_exp_reward(self, x1_diff, x3_diff, x2, x4, dx2, dx4, u1, u2, qx1=1, qx2=0.01, qx3=1, qx4=0.01, pu1=0,
                        pu2=0, eax1=10, ebx1=10, eax3=10, ebx3=10):
        return -(qx1 * (x1_diff ** 2) + qx3 * (x3_diff ** 2) + qx2 * (x2 ** 2) + qx4 * (x4 ** 2) + pu1 * (
                    u1 ** 2) + pu2 * (u2 ** 2) + eax1 * (1 - np.exp(-ebx1 * (x1_diff ** 2))) + eax3 * (
                             1 - np.exp(-ebx3 * (x3_diff ** 2))))

    def quad_end_pen_reward(self, x1_diff, x3_diff, x2, x4, dx2, dx4, u1, u2, qx1=1, qx2=0.01, qx3=1, qx4=0.01, pu1=0,
                            pu2=0, sx1=10, sx3=10, end_horizon=0):
        return -(qx1 * (x1_diff ** 2) + qx3 * (x3_diff ** 2) + qx2 * (x2 ** 2) + qx4 * (x4 ** 2) + pu1 * (
                    u1 ** 2) + pu2 * (u2 ** 2) + end_horizon * (sx1 * (x1_diff ** 2) + sx3 * (x3_diff ** 2)))

    def quad_pen_reward(self, x1_diff, x3_diff, x2, x4, dx2, dx4, u1, u2, qx1=1, qx2=0.01, qx3=1, qx4=0.01, pu1=0,
                        pu2=0, bound=0.1, penalty=50):
        return -(qx1 * (x1_diff ** 2) + qx3 * (x3_diff ** 2) + qx2 * (x2 ** 2) + qx4 * (x4 ** 2) + pu1 * (
                    u1 ** 2) + pu2 * (u2 ** 2)) - penalty * (abs(x1_diff) >= bound or abs(x3_diff) >= bound)

    def quad_bon_reward(self, x1_diff, x3_diff, x2, x4, dx2, dx4, u1, u2, qx1=1, qx2=0.01, qx3=1, qx4=0.01, pu1=0,
                        pu2=0, bound=0.1, bonus=5):
        return -(qx1 * (x1_diff ** 2) + qx3 * (x3_diff ** 2) + qx2 * (x2 ** 2) + qx4 * (x4 ** 2) + pu1 * (
                    u1 ** 2) + pu2 * (u2 ** 2)) + bonus * (abs(x1_diff) <= bound or abs(x3_diff) <= bound)

    # ---------------------------------------------------------------------------------------------------- #
    # ------------------------------------------ Reward Part II ------------------------------------------ #
    # ---------------------------------------------------------------------------------------------------- #

    def power_reward(self, x1_diff, x3_diff, x2, x4, dx2, dx4, u1, u2, qx1=1, qx2=1, qx3=1, qx4=1, pu1=0, pu2=0, p=0.5):
        return -(qx1 * abs(x1_diff) ** p + qx3 * abs(x3_diff) ** p + qx2 * abs(x2) ** p + qx4 * abs(
            x4) ** p + pu1 * abs(u1) ** p + pu2 * abs(u2) ** p)

    def exp_reward(self, x1_diff, x3_diff, x2, x4, dx2, dx4, u1, u2, qx1=1, qx2=1, qx3=1, qx4=1, pu1=0, pu2=0, e=10):
        return -(qx1 * (1 - np.exp(-e * abs(x1_diff))) + qx3 * (1 - np.exp(-e * abs(x3_diff))) + qx2 * (
                    1 - np.exp(-e * abs(x2))) + qx4 * (1 - np.exp(-e * abs(x4))) + pu1 * (
                             1 - np.exp(-e * abs(u1))) + pu2 * (1 - np.exp(-e * abs(u2))))

    def power_exp_reward(self, x1_diff, x3_diff, x2, x4, dx2, dx4, u1, u2, qx1=1, qx2=1, qx3=1, qx4=1, pu1=0, pu2=0,
                         p=0.1, e=10):
        return -(qx1 * abs(x1_diff) ** p + qx3 * abs(x3_diff) ** p + qx2 * abs(x2) ** p + qx4 * abs(
            x4) ** p + pu1 * abs(u1) ** p + pu2 * abs(u2) ** p) - (
                           qx1 * (1 - np.exp(-e * abs(x1_diff))) + qx3 * (1 - np.exp(-e * abs(x3_diff))) + qx2 * (
                               1 - np.exp(-e * abs(x2))) + qx4 * (1 - np.exp(-e * abs(x4))) + pu1 * (
                                       1 - np.exp(-e * abs(u1))) + pu2 * (1 - np.exp(-e * abs(u2))))

    def power_exp_reward_new_V0(self, x1_diff, x3_diff, w_diff, x2, x4, w, dx2, dx4, dw, u1, u2, u3, qx1=1, qx2=1, qx3=1, qx4=1, qw = 1, pu1=0, pu2=0, pu3 =0,
                         p=0.1, e=10):   # 需要修订！！！
        return -(qx1 * abs(x1_diff) ** p + qx3 * abs(x3_diff) ** p + qw * abs(w_diff) ** p + qx2 * abs(x2) ** p + qx4 * abs(
            x4) ** p + qw * abs(w) ** p + pu1 * abs(u1) ** p + pu2 * abs(u2) ** p + pu3 * abs(u3) ** p ) - (
                           qx1 * (1 - np.exp(-e * abs(x1_diff))) + qx3 * (1 - np.exp(-e * abs(x3_diff))) + qw * (1 - np.exp(-e * abs(w_diff))) + qx2 * (
                               1 - np.exp(-e * abs(x2))) + qx4 * (1 - np.exp(-e * abs(x4))) + qw * (1 - np.exp(-e * abs(w)))+ pu1 * (
                                       1 - np.exp(-e * abs(u1))) + pu2 * (1 - np.exp(-e * abs(u2))) + pu3 * (1 - np.exp(-e * abs(u3))))



    # ---------------------------------------------------------------------------------------------------- #
    # ------------------------------------------ Reward Part III ----------------------------------------- #
    # ---------------------------------------------------------------------------------------------------- #

    def sparse_reward(self, x1_diff, x3_diff, x2, x4, dx2, dx4, u1, u2, bx=0.01, rx=1, bv=0.01, rv=0, bu=0.01, ru=0):
        r = 0
        if abs(x1_diff) <= bx and abs(x3_diff) <= bx:
            if abs(x2) <= bv and abs(x4) <= bv:
                r += rv
            if abs(dx2) <= bu and abs(dx4) <= bu:
                r += ru
            r += rx
        return r

    def sparse_reward_with_exp(self, x1_diff, x3_diff, x2, x4, dx2, dx4, u1, u2, qx1=1, qx2=1, qx3=1, qx4=1, pu1=0,
                               pu2=0, e=10, bound=0.01, reward=1):
        return -(qx1 * (1 - np.exp(-e * abs(x1_diff))) + qx3 * (1 - np.exp(-e * abs(x3_diff))) + qx2 * (
                    1 - np.exp(-e * abs(x2))) + qx4 * (1 - np.exp(-e * abs(x4))) + pu1 * (
                             1 - np.exp(-e * abs(u1))) + pu2 * (1 - np.exp(-e * abs(u2)))) + reward * (
                           abs(x1_diff) <= bound and abs(x3_diff) <= bound)

    def sparse_reward_with_exp_2(self, x1_diff, x3_diff, x2, x4, dx2, dx4, u1, u2, qx1=1, qx2=1, qx3=1, qx4=1, pu1=0,
                                 pu2=0, e=10, bound=0.01, reward=1):
        return -(qx1 * (1 - np.exp(-e * abs(x1_diff))) + qx3 * (1 - np.exp(-e * abs(x3_diff))) + qx2 * (
                    1 - np.exp(-e * abs(x2))) + qx4 * (1 - np.exp(-e * abs(x4))) + pu1 * (
                             1 - np.exp(-e * abs(u1))) + pu2 * (1 - np.exp(-e * abs(u2)))) + reward * (
                           abs(x1_diff) <= bound) + reward * (abs(x3_diff) <= bound)

    # ---------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------- Helper ---------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------- #

    # reset system to a given or random initial state
    def reset(self, x_0=None):

        # reset state
        if x_0 is None:
            self.state = self.state_space.sample()
        else:
            self.state = x_0
        # update observation
        self.observe()
        # reset counter
        self.ep_cur = 0

        return self.observation

    # return normalized observation
    def observe(self):
        s = self.state
        self.observation = np.array([np.cos(s[0]), np.sin(s[0]), s[1] / self.maxGimbalSpeed,
                                     np.cos(s[2]), np.sin(s[2]), s[3] / self.maxGimbalSpeed,
                                     s[4] / self.maxAngle, s[5] / self.maxAngle, s[6] / self.maxDiskSpeed])
        return self.observation

    # Keep the angles between -lim and lim
    def angle_normalize(self, x, lim=np.pi):   # self.maxAngle = np.pi  [rad]
        return ((x + lim) % (2 * lim)) - lim

    def diskspeed_normalize(self, x, lim=300 * 2 * np.pi / 60):   # self.maxDiskSpeed = 300 * 2 * np.pi / 60  [rad/s]
        return ((x + lim) % (2 * lim)) - lim

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None