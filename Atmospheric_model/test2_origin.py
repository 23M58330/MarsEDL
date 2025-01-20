import numpy as np
import matplotlib.pyplot as plt

# 初始参数
he = 122000.0  # 初始高度，单位：米
R = 6378000  # 地球半径，单位：米
Ve = 7500.0  # 初始速度，单位：米/秒
beta = 0.1354 / 1000.0  # 空气密度常数，单位：(1/米)
rhos = 1.225  # 海平面空气密度，单位：千克/立方米
m = 1350.0  # 物体质量，单位：千克
CD = 1.5  # 阻力系数，无量纲
S = 2.8  # 再入飞行器的迎风面积，单位：平方米
gammae = -2.0 * np.pi / 180.0  # 再入角度，单位：弧度
dt = 0.1  # 时间步长，单位：秒

# 地球的常数
G = 6.67430e-11  # 引力常数，单位 m^3 kg^-1 s^-2
M = 5.972e24  # 地球质量，单位 kg
R_Earth = 6.371e6  # 地球半径，单位 m

# 更新重力加速度公式
def gravity(h):
    """
    根据高度计算地球上的重力加速度。

    参数:
        h: 高度，单位为米

    返回:
        重力加速度，单位 m/s^2
    """
    return G * M / (R_Earth + h) ** 2

# 导数函数
def Derivatives(state, t):
    V = state[0]  # 速度
    h = state[1]  # 高度
    gamma = state[2]  # 飞行路径角

    # 空气密度模型
    rho = rhos * np.exp(-beta * h)

    # 气动力模型
    D = 0.5 * rho * V**2 * S * CD  # 阻力

    # 重力加速度
    g = gravity(h)  # 根据高度计算的重力加速度
    r = R + h  # 从地心到物体的距离

    # 动力学方程
    dVdt = -D / m - g * np.sin(gamma)  # dV/dt
    dgammadt = (-g * np.cos(gamma) + V**2 / r) / V  # dγ/dt
    dhdt = V * np.sin(gamma)  # dh/dt

    return np.asarray([dVdt, dhdt, dgammadt])

# 龙格-库塔法（四阶）
def runge_kutta_step(f, state, t, dt):
    k1 = f(state, t)
    k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(state + dt * k3, t + dt)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# 动态积分
state = np.asarray([Ve, he, gammae])  # 初始状态
t = 0.0  # 初始时间
stateout = []  # 存储结果
tout = []  # 存储时间

while state[1] > 0:  # 当高度大于零时继续计算
    state = runge_kutta_step(Derivatives, state, t, dt)  # 使用龙格-库塔法计算一个时间步
    stateout.append(state)  # 保存当前状态
    tout.append(t)  # 保存当前时间
    t += dt  # 更新时间

stateout = np.array(stateout)
tout = np.array(tout)

Vnum = stateout[:, 0]  # 数值解的速度
hnum = stateout[:, 1]  # 数值解的高度
gammanum = stateout[:, 2]  # 数值解的飞行路径角
accelnum = (Vnum[1:] - Vnum[:-1]) / dt  # 数值解的加速度

# 绘制图像
# 速度-高度图
plt.figure()
plt.plot(hnum, Vnum, label='Numerical')
plt.xlabel('Altitude (m)')
plt.ylabel('Velocity (m/s)')
plt.grid()
plt.legend()

# 加速度-高度图
plt.figure()
plt.plot(hnum[:-1], accelnum / 9.81, label='Numerical')
plt.xlabel('Altitude (m)')
plt.ylabel('Gs')
plt.grid()
plt.legend()

# 飞行路径角-时间图
plt.figure()
plt.plot(tout, gammanum, 'm-')
plt.ylabel('Flight Path Angle (rad)')
plt.xlabel('Time (sec)')
plt.grid()

# 高度-时间图
plt.figure()
plt.plot(tout, hnum)
plt.xlabel('Time (sec)')
plt.ylabel('Altitude (m)')
plt.grid()

plt.show()
