# -*- coding: utf-8 -*-
"""
创建于2021年4月24日 10:26:47

作者：Dani
"""

import pandas as pd  # 导入处理数据的库
import numpy as np  # 导入处理数值计算的库
import matplotlib.pyplot as plt  # 导入用于绘制图形的库
import pylab as pl  # 导入pylab库，用于辅助绘图
import matplotlib.animation as animation  # 导入动画库
from matplotlib.animation import PillowWriter  # 导入PillowWriter，用于生成GIF动画

# =============================================================================
# 基本函数
# =============================================================================
def integra(y, q1, h):
    """
    使用欧拉法计算变量的下一个值。

    参数:
        y: 当前变量值。
        q1: 变量的导数。
        h: 时间步长。

    返回:
        下一个变量值。
    """
    y_siguiente = y + q1 * h  # 通过当前值、导数和步长计算下一个值
    return y_siguiente  # 返回计算结果


# =============================================================================
# 简化的大气模型
# =============================================================================

# 温度：使用Viking 1探测器的测量值进行多项式拟合。
#   高度单位为公里，温度单位为开尔文
df = pd.read_csv('viking1.csv', sep=';', header=None, decimal=',')  # 读取CSV文件，包含火星探测器测得的温度数据
df[0].values  # 提取高度值

# 火星上的气体常数
R_gas = 188.92  # 火星大气气体常数，单位为J/(kg*K)
gamma_gas = 1.2941  # 比热比

plt.style.use('default')  # 使用默认的绘图风格
plt.plot(df[1], df[0])  # 绘制原始数据中的温度与高度关系
z = np.polyfit(df[1][:], df[0][:], 7)  # 对温度数据进行7次多项式拟合
temperature = np.poly1d(z)  # 创建多项式函数来计算温度
xp_linear = np.linspace(0, 120, 100)  # 创建从0到120之间的线性空间，模拟高度，单位为公里
plt.plot(df[1], df[0], '.', label='Viking 1 Measurements')  # 用点状图表示原始测量值
plt.plot(xp_linear, temperature(xp_linear), '-', label='Simulated Model')  # 用线条绘制多项式拟合的温度模型
plt.plot(xp_linear, -23.4 + 273.15 - 0.00222 * 1000 * xp_linear, label='NASA Model Zone 1')  # NASA模型区域1
plt.plot(xp_linear, -31 + 273.15 - 0.000998 * 1000 * xp_linear, label='NASA Model Zone 2')  # NASA模型区域2
plt.legend(loc="lower left")  # 添加图例，显示在左下角
plt.xlabel("Altitude (km)")  # 设置x轴标签为高度
plt.ylabel("Temperature (K)")  # 设置y轴标签为温度
plt.title("Temperature Profile Models in the Martian Atmosphere")  # 设置图表标题


# 压力函数:
def presion_marte(z):
    """
    根据火星上的高度计算气压。

    参数:
        z: 高度，单位为公里。

    返回:
        气压，单位为帕斯卡。
    """
    h = z * 1000  # 将高度转换为米
    p = .699 * np.exp(-0.00009 * h) * 1000  # 计算气压，单位为Pa（帕斯卡）
    return p  # 返回计算得到的气压


# 密度函数:
def rho(z):
    """
    根据火星上的高度计算大气密度。

    参数:
        z: 高度，单位为公里。

    返回:
        大气密度，单位为kg/m^3。
    """
    r = presion_marte(z) / (R_gas * temperature(z))  # 通过气体方程计算密度
    return r  # 返回计算得到的密度


# 生成图形
fig = plt.figure(figsize=(20, 10))  # 创建大小为20x10的绘图区域
# 温度图
ax = fig.add_subplot(131)  # 在图形中添加子图，位置为第1行第3列的第1个
ax.set_xlabel('Altitude (km)')  # 设置x轴标签为高度
ax.set_ylabel('Temperature (K)')  # 设置y轴标签为温度
ax.plot(xp_linear, temperature(xp_linear), 'gs-')  # 绘制温度与高度的关系
ax.set_title('Temperature vs. Altitude')  # 设置子图标题

# 气压图
ax = fig.add_subplot(132)  # 添加第二个子图，位置为第1行第3列的第2个
ax.set_xlabel('Altitude (km)')  # 设置x轴标签为高度
ax.set_ylabel('Pressure (Pa)')  # 设置y轴标签为气压
ax.plot(xp_linear, presion_marte(xp_linear), 'cs-')  # 绘制气压与高度的关系
ax.set_title('Pressure vs. Altitude')  # 设置子图标题

# 密度图
ax2 = fig.add_subplot(133)  # 添加第三个子图，位置为第1行第3列的第3个
ax2.set_xlabel('Altitude (km)')  # 设置x轴标签为高度
ax2.set_ylabel('Density (kg/m^3)')  # 设置y轴标签为密度
ax2.set_title('Density vs. Altitude')  # 设置子图标题
ax2.plot(xp_linear, rho(xp_linear), 'rs-')  # 绘制密度与高度的关系
plt.show()  # 显示图形

# =============================================================================
# 火星着陆模拟
# =============================================================================

# 问题中的常量
g = 3.721  # 火星表面重力加速度，单位为m/s^2

# 配置参数
h0 = 30000  # 初始高度，单位为米
h1 = 12400  # 开伞时的高度，单位为米
h2 = 1500  # 减速完成后的高度，单位为米
s0 = 8  # 初始受力面积，单位为m^2
s1 = 0  # 最终伞打开时的面积，单位为m^2
m = 1293  # 着陆器质量，单位为kg
mf = 1255.63  # 剩余燃料质量，单位为kg
t0 = 0  # 初始时间，单位为秒
t_elapsed = 0  # 已经过的时间，单位为秒
t_combustion = 154  # 燃烧时间，单位为秒
v0 = -800  # 初始速度，单位为m/s
dt = 0.1  # 时间步长，单位为秒
Cd0 = 1.3  # 初始阻力系数
Cd1 = 0.39  # 最终阻力系数
v_e = 200  # 喷射速度，单位为m/s


def vel_sonido(h):
    """
    根据高度计算音速。

    参数:
        h: 高度，单位为米。

    返回:
        音速，单位为m/s。
    """
    velocidad_sonido = np.sqrt(gamma_gas * R_gas * temperature(h / 1000))  # 根据比热比和气体常数计算音速
    return velocidad_sonido  # 返回音速


def Mach(velocidad, h):
    """
    根据速度和高度计算马赫数。

    参数:
        velocidad: 速度，单位为m/s。
        h: 高度，单位为米。

    返回:
        马赫数。
    """
    mach_number = abs(velocidad) / vel_sonido(h)  # 计算马赫数
    return mach_number  # 返回马赫数


def factor_compresibilidad(mach):
    """
    根据马赫数计算可压缩性因子。

    参数:
        mach: 马赫数。

    返回:
        可压缩性因子。
    """
    if mach <= 0.7:
        factor = 1 / (np.sqrt(1 - (mach ** 2)))  # 亚音速时的可压缩性因子
    elif 0.7 < mach <= 1:
        factor = 1 / (np.sqrt(1 - (0.7 ** 2)))  # 音速时的可压缩性因子
    elif 1 < mach <= 1.3:
        factor = 1 / (np.sqrt((1.3 ** 2) - 1))  # 超音速时的可压缩性因子
    else:
        factor = 1 / (np.sqrt((mach ** 2) - 1))  # 其他情况下的可压缩性因子

    return factor  # 返回可压缩性因子


# 定义初始状态向量
u = np.array([[t0, h0, v0, vel_sonido(h0), Mach(v0, h0), factor_compresibilidad(Mach(v0, h0))]])
# u数组包含以下信息：
#   u[:, 0]: 时间，单位为秒
#   u[:, 1]: 高度，单位为米
#   u[:, 2]: 速度，单位为米/秒
#   u[:, 3]: 音速，单位为米/秒
#   u[:, 4]: 马赫数
#   u[:, 5]: 可压缩性因子

# =============================================================================
# 绘制着陆模拟图
# =============================================================================

# 创建绘图窗口和坐标轴
plt.rcParams.update({'font.size': 16})  # 设置字体大小
plt.style.use('dark_background')  # 使用黑色背景
fig = plt.figure(figsize=(16, 16))  # 创建绘图窗口，大小为16x16英寸
ax = fig.add_subplot(211)  # 创建子图，两行一列的第一个子图，用于绘制高度-时间图像
ax.set_xlabel('Time (s)')  # 设置x轴标签
ax.set_ylabel('Altitude (m)')  # 设置y轴标签
ax.set_title('Spacecraft Altitude vs. Time')  # 设置图表标题
ax.set_xlim([0, 160])  # 设置x轴范围
ax.set_ylim([0, 36000])  # 设置y轴范围

# 添加文本注释
time_text = ax.text(120, 32000, '', fontsize=18)  # 创建文本对象，用于显示时间信息
height_text = ax.text(120, 29000, '', fontsize=18)  # 创建文本对象，用于显示高度信息
speed_text = ax.text(120, 26000, '', fontsize=18)  # 创建文本对象，用于显示速度信息
mach_text = ax.text(120, 23000, '', fontsize=18)  # 创建文本对象，用于显示马赫数信息

# 绘制高度-时间曲线
line, = ax.plot(u[:, 0], u[:, 1], 'y-', lw=3, )  # 绘制高度-时间曲线，颜色为黄色，线宽为3

# 创建第二个子图，用于绘制速度-时间图像
ax2 = fig.add_subplot(212)
ax2.set_xlabel('Time (s)')  # 设置x轴标签
ax2.set_ylabel('Velocity (m/s)')  # 设置y轴标签
ax2.set_title('Spacecraft Velocity vs. Time')  # 设置图表标题
ax2.set_xlim([0, 160])  # 设置x轴范围
ax2.set_ylim([0, 1000])  # 设置y轴范围

# 绘制速度-时间曲线
line2, = ax2.plot(u[:, 0], -u[:, 2], lw=3)  # 绘制速度-时间曲线，线宽为3

# 模拟着陆过程
i = 0  # 初始化时间步
while u[i, 1] > 0:  # 当高度大于0时循环
    if i % 10 == 0:
        print(f"Time: {u[i, 0]:.2f} s, Altitude: {u[i, 1]:.2f} m, Velocity: {u[i, 2]:.2f} m/s, "
              f"Sound Speed: {u[i, 3]:.2f} m/s, Mach Number: {u[i, 4]:.2f}, Compressibility Factor: {u[i, 5]:.2f}")
    if u[i, 1] >= h1:  # 判断是否到达开伞高度
        s = s0  # 使用着陆器的横截面积
        fuerza = 0.5 * rho(u[i, 1] / 1000) * s0 * Cd0 * u[i, 5] * u[i, 2] ** 2  # 计算空气阻力
        s1 = 0  # 降落伞还未打开
        a_cohete = 0  # 火箭推力为0
    elif h2 < u[i, 1] < h1:  # 判断是否处于打开降落伞后的阶段
        if s1 < 330:  # 逐渐打开降落伞
            s1 += 10
            print(s1)
        fuerza = 0.5 * rho(u[i, 1] / 1000) * s0 * Cd0 * u[
            i, 2] ** 2 + 0.5 * rho(u[i, 1] / 1000) * s1 * Cd1 * u[
                     i, 5] * u[i, 2] ** 2  # 计算空气阻力，考虑降落伞的影响
        a_cohete = 0  # 火箭推力为0


    f = np.array([0, u[i, 2], (fuerza / m) - g + a_cohete, 0, 0, 0])  # 计算状态量的变化率
    u = np.vstack([u, integra(u[i, :], f, dt)])  # 使用欧拉法更新状态量
    u[i + 1, 0] = u[i, 0] + dt  # 更新时间
    u[i + 1, 3] = vel_sonido(u[i, 1])  # 更新音速
    u[i + 1, 4] = Mach(u[i, 2], u[i, 1])  # 更新马赫数
    u[i + 1, 5] = factor_compresibilidad(u[i + 1, 4])  # 更新可压缩性因子

    i += 1  # 进入下一个时间步

i_final = i  # 记录最终时间步

# ----------------------------------
# ----------------------------------

# ----------------------------------


# 动态调整坐标轴范围，并留出空白
def add_margin(min_val, max_val, margin=0.1):
    """计算范围，添加一定比例的空白"""
    delta = max_val - min_val
    return min_val - margin * delta, max_val + margin * delta

time_min, time_max = add_margin(np.min(u[:, 0]), np.max(u[:, 0]))
altitude_min, altitude_max = add_margin(np.min(u[:, 1]), np.max(u[:, 1]))
velocity_min, velocity_max = add_margin(np.min(-u[:, 2]), np.max(-u[:, 2]))
sound_min, sound_max = add_margin(np.min(u[:, 3]), np.max(u[:, 3]))
mach_min, mach_max = add_margin(np.min(u[:, 4]), np.max(u[:, 4]))
compressibility_min, compressibility_max = add_margin(np.min(u[:, 5]), np.max(u[:, 5]))

# 绘制高度和速度随时间变化曲线
plt.rcParams.update({'font.size': 15})
plt.style.use('default')  # 使用默认的绘图风格
fig = plt.figure(figsize=(16, 8))  # 调整大小为横向

# 绘制高度随时间变化曲线
ax = fig.add_subplot(121)  # 1行2列的第一个子图
ax.set_xlabel('Time (s)')
ax.set_ylabel('Altitude (m)')
ax.set_title('Spacecraft Altitude vs. Time')
ax.set_xlim([time_min, time_max])  # 时间范围
ax.set_ylim([altitude_min, altitude_max])  # 高度范围
ax.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
line, = ax.plot(u[:, 0], u[:, 1], 'r-', lw=3)  # 绘制高度-时间曲线，颜色为红色，线宽为3

# 绘制速度随时间变化曲线
ax2 = fig.add_subplot(122)  # 1行2列的第二个子图
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_title('Spacecraft Velocity vs. Time')
ax2.set_xlim([time_min, time_max])  # 时间范围
ax2.set_ylim([velocity_min, velocity_max])  # 速度范围
ax2.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
line2, = ax2.plot(u[:, 0], -u[:, 2], lw=3)  # 绘制速度-时间曲线，线宽为3

# -----------------------------------

# 绘制声速、马赫数、可压缩性因子随时间变化曲线
fig = plt.figure(figsize=(20, 10))

# 声速-时间曲线
ax = fig.add_subplot(131)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Speed of Sound (m/s)')
ax.set_title('Speed of Sound vs. Time')
ax.set_xlim([time_min, time_max])  # 时间范围
ax.set_ylim([sound_min, sound_max])  # 声速范围
ax.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
line, = ax.plot(u[:, 0], u[:, 3], 'r-', lw=3)  # 绘制声速-时间曲线，颜色为红色，线宽为3

# 马赫数-时间曲线
ax2 = fig.add_subplot(132)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Mach Number')
ax2.set_title('Spacecraft Mach Number vs. Time')
ax2.set_xlim([time_min, time_max])  # 时间范围
ax2.set_ylim([mach_min, mach_max])  # 马赫数范围
ax2.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
line2, = ax2.plot(u[:, 0], u[:, 4], 'b-', lw=3)  # 绘制马赫数-时间曲线，颜色为蓝色，线宽为3

# 可压缩性因子-时间曲线
ax3 = fig.add_subplot(133)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Compressibility Factor')
ax3.set_title('Compressibility Factor vs. Time')
ax3.set_xlim([time_min, time_max])  # 时间范围
ax3.set_ylim([compressibility_min, compressibility_max])  # 可压缩性因子范围
ax3.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
line3, = ax3.plot(u[:, 0], u[:, 5], 'g-', lw=3)  # 绘制可压缩性因子-时间曲线，颜色为绿色，线宽为3

plt.show()  # 显示图形

# 绘制原始数据中的速度与高度关系
plt.style.use('default')  # 使用默认的绘图风格
velocity_min, velocity_max = add_margin(np.min(np.abs(u[:, 2])), np.max(np.abs(u[:, 2])))
altitude_min, altitude_max = add_margin(np.min(u[:, 1] / 1000), np.max(u[:, 1] / 1000))
plt.plot(np.abs(u[:, 2]), u[:, 1] / 1000)  # 绘制速度与高度关系
plt.xlabel("Velocity (m/s)")  # 设置x轴标签为速度
plt.ylabel("Altitude (km)")  # 设置y轴标签为高度
plt.title("Atmospheric entry and descent trajectory")  # 设置图表标题
plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
plt.xlim([velocity_min, velocity_max])  # 动态调整速度范围
plt.ylim([altitude_min, altitude_max])  # 动态调整高度范围


plt.show()  # 显示图形
