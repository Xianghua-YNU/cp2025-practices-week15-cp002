"""
双摆模拟
课程：计算物理
说明：请完成标记为 TODO 的部分，完成双摆运动的数值模拟和能量计算。

系统参数说明：
- m1, m2: 两个摆球的质量 (kg)
- l1, l2: 两个摆长 (m)
- G_CONST: 重力加速度 (m/s^2)
- 初始状态 y0 = [theta1, omega1, theta2, omega2]
    theta为摆角（弧度），omega为角速度（rad/s）

边界条件：
- 时间t为一维numpy数组，表示模拟的时间序列
- 摆角theta无特殊边界限制，可为任意实数（代表转过的角度）
- 初始角速度和角度根据实验或模拟需求给出

"""

import numpy as np
from scipy.integrate import odeint

G_CONST = 9.81  # 重力加速度 (m/s^2)

def derivatives(y, t, m1, m2, l1, l2):
    """
    计算双摆系统的导数，基于拉格朗日方程得到的运动方程。

    参数：
    y: 状态变量数组 [theta1, omega1, theta2, omega2]
    t: 时间（odeint需要，但该系统为自治系统，t无显式出现）
    m1, m2: 摆球质量
    l1, l2: 摆长

    返回：
    dydt: 状态变量的导数 [dtheta1/dt, domega1/dt, dtheta2/dt, domega2/dt]
    """
    theta1, omega1, theta2, omega2 = y
    delta = theta2 - theta1

    denom1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta) * np.cos(delta)
    denom2 = (l2 / l1) * denom1

    # 角加速度计算公式，来源于经典双摆动力学推导
    domega1 = (m2 * l1 * omega1**2 * np.sin(delta) * np.cos(delta) +
               m2 * G_CONST * np.sin(theta2) * np.cos(delta) +
               m2 * l2 * omega2**2 * np.sin(delta) -
               (m1 + m2) * G_CONST * np.sin(theta1)) / denom1

    domega2 = (-m2 * l2 * omega2**2 * np.sin(delta) * np.cos(delta) +
               (m1 + m2) * G_CONST * np.sin(theta1) * np.cos(delta) -
               (m1 + m2) * l1 * omega1**2 * np.sin(delta) -
               (m1 + m2) * G_CONST * np.sin(theta2)) / denom2

    return [omega1, domega1, omega2, domega2]

def solve_double_pendulum(y0, t, m1, m2, l1, l2):
    """
    利用 scipy.integrate.odeint 求解双摆运动的微分方程。

    参数：
    y0: 初始状态向量 [theta1_0, omega1_0, theta2_0, omega2_0]
    t: 时间数组，一维numpy数组，表示求解的时间点
    m1, m2: 摆球质量
    l1, l2: 摆长

    返回：
    sol: 数组，形状为(len(t), 4)，每行对应时刻t[i]的状态变量
    """
    sol = odeint(derivatives, y0, t, args=(m1, m2, l1, l2))
    return sol

def total_energy(y, m1, m2, l1, l2):
    """
    计算双摆系统的总能量（动能 + 势能）。

    参数：
    y: 状态变量向量 [theta1, omega1, theta2, omega2]
    m1, m2: 摆球质量
    l1, l2: 摆长

    返回：
    E: 标量，总能量（焦耳）
    """
    theta1, omega1, theta2, omega2 = y

    # 计算质点位置 (以悬点为坐标原点，向下为正y)
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    # 计算质点速度（角速度转化为线速度）
    v1x = l1 * omega1 * np.cos(theta1)
    v1y = l1 * omega1 * np.sin(theta1)
    v2x = v1x + l2 * omega2 * np.cos(theta2)
    v2y = v1y + l2 * omega2 * np.sin(theta2)

    KE1 = 0.5 * m1 * (v1x**2 + v1y**2)
    KE2 = 0.5 * m2 * (v2x**2 + v2y**2)

    # 势能取悬点为零势能参考点，y向下为正，注意加上摆长修正
    PE1 = m1 * G_CONST * (y1 + l1)
    PE2 = m2 * G_CONST * (y2 + l1 + l2)

    E = KE1 + KE2 + PE1 + PE2
    return E
