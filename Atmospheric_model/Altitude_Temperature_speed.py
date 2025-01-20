import pandas as pd  # 导入处理数据的库
import numpy as np  # 导入处理数值计算的库
import matplotlib.pyplot as plt  # 导入用于绘制图形的库
from scipy import interpolate  # 导入插值库


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
    y_next = y + q1 * h  # 通过当前值、导数和步长计算下一个值
    return y_next  # 返回计算结果


# =============================================================================
# 简化的大气模型
# =============================================================================


# 读取CSV文件，并处理千分位分隔符
df = pd.read_csv('NASA_simple.csv', sep=';', header=None, decimal=',', thousands=',')  # 处理千分位

# 确保所有的数值列是浮动类型
df[0] = pd.to_numeric(df[0], errors='coerce')  # 高度列
df[1] = pd.to_numeric(df[1], errors='coerce')  # 温度列
df[2] = pd.to_numeric(df[2], errors='coerce')  # 压力列
df[3] = pd.to_numeric(df[3], errors='coerce')  # 密度列

altitude = df[0].values  # 高度（单位：米）
temperature = df[1].values  # 温度（单位：K）
pressure = df[2].values  # 压强（单位：Pa）
density = df[3].values  # 密度（单位：kg/m^3）

# 创建插值函数
altitude_interp = interpolate.interp1d(altitude, altitude, kind='linear', fill_value="extrapolate")
temperature_interp = interpolate.interp1d(altitude, temperature, kind='linear', fill_value="extrapolate")
pressure_interp = interpolate.interp1d(altitude, pressure, kind='linear', fill_value="extrapolate")
density_interp = interpolate.interp1d(altitude, density, kind='linear', fill_value="extrapolate")

# 创建从0到410000米之间的线性空间，高度以米为单位
xp_linear = np.linspace(0, 410000, 100)

# 绘制图形
fig = plt.figure(figsize=(20, 10))

# 温度图
ax1 = fig.add_subplot(131)
ax1.plot(temperature / 1000, altitude / 1000, 'o', label='Data Points')  # 温度在横坐标，高度在纵坐标
ax1.plot(temperature_interp(xp_linear) / 1000, xp_linear / 1000, '-', label='Interpolation')  # 交换数据点和插值
ax1.set_xlabel('Temperature (K)')
ax1.set_ylabel('Altitude (km)')
ax1.set_title('Temperature vs. Altitude')
ax1.legend()
ax1.grid(True)  # 启用网格
ax1.set_axisbelow(True)  # 网格线在数据点下方

# 压强图
ax2 = fig.add_subplot(132)
ax2.plot(pressure, altitude / 1000, 'o', label='Data Points')  # 高度单位转换为公里
ax2.plot(pressure_interp(xp_linear), xp_linear / 1000, '-', label='Interpolation')
ax2.set_xscale('log')  # 对数横坐标
ax2.set_xlabel('Pressure (Pa)')
ax2.set_ylabel('Altitude (km)')
ax2.set_title('Pressure vs. Altitude')
ax2.legend()
ax2.grid(True)  # 启用网格
ax2.set_axisbelow(True)  # 网格线在数据点下方

# 密度图
ax3 = fig.add_subplot(133)
ax3.plot(density, altitude / 1000, 'o', label='Data Points')  # 高度单位转换为公里
ax3.plot(density_interp(xp_linear), xp_linear / 1000, '-', label='Interpolation')
ax3.set_xscale('log')  # 对数横坐标
ax3.set_xlabel('Density (kg/m^3)')
ax3.set_ylabel('Altitude (km)')
ax3.set_title('Density vs. Altitude')
ax3.legend()
ax3.grid(True)  # 启用网格
ax3.set_axisbelow(True)  # 网格线在数据点下方

plt.tight_layout()
plt.show()

# =============================================================================
# 火星着陆模拟
# =============================================================================

# 问题中的常量
g = 3.721  # 火星表面重力加速度，单位为m/s^2
R_gas = 188.92  # 火星大气气体常数，单位为J/(kg*K)
gamma_gas = 1.2941  # 比热比

# 配置参数
h0 = 30000  # 初始高度，单位为米
h1 = 12400  # 开伞时的高度，单位为米
s0 = 8  # 初始受力面积，单位为m^2
s1 = 0  # 降落伞的实时面积（会从零展开），单位为m^2
m = 1293  # 着陆器质量，单位为kg
t0 = 0  # 初始时间，单位为秒
t_elapsed = 0  # 已经过的时间，单位为秒
v0 = -800  # 初始速度，单位为m/s
dt = 0.1  # 时间步长，单位为秒
Cd0 = 1.3  # 初始阻力系数
Cd1 = 0.39  # 降落伞的阻力系数（这个数回去再确认）


def velocity_sound(h):
    temp = temperature_interp(h) / 1000
    sound_velocity = np.sqrt(gamma_gas * R_gas * temp)
    return sound_velocity

def Mach(velocity, h):
    """
    根据速度和高度计算马赫数。

    参数:
        velocity: 速度，单位为m/s。
        h: 高度，单位为米。

    返回:
        马赫数。
    """

    mach_number = abs(velocity) / velocity_sound(h)  # 计算马赫数
    return mach_number  # 返回马赫数


def compressibility_factor(mach):
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
u = np.array([[t0, h0, v0, velocity_sound(h0), Mach(v0, h0), compressibility_factor(Mach(v0, h0))]])
# u数组包含以下信息：
#   u[:, 0]: 时间，单位为秒
#   u[:, 1]: 高度，单位为米
#   u[:, 2]: 速度，单位为米/秒
#   u[:, 3]: 音速，单位为米/秒
#   u[:, 4]: 马赫数
#   u[:, 5]: 可压缩性因子

# 模拟着陆过程
i = 0  # 初始化时间步
while u[i, 1] > 0:  # 当高度大于0时循环
    # 打印每个时间步的相关信息
    if i % 10 == 0:
        print(f"Time: {u[i, 0]:.2f} s, Altitude: {u[i, 1]:.2f} m, Velocity: {u[i, 2]:.2f} m/s, "
              f"Sound Speed: {u[i, 3]:.2f} m/s, Mach Number: {u[i, 4]:.2f}, Compressibility Factor: {u[i, 5]:.2f}")

    if u[i, 1] >= h1:  # 判断是否到达开伞高度
        s = s0  # 使用着陆器的横截面积
        force = 0.5 * density_interp(u[i, 1] / 1000) * s0 * Cd0 * u[i, 5] * u[i, 2] ** 2  # 计算空气阻力
        s1 = 0  # 降落伞还未打开

    else:  # 判断是否处于打开降落伞后的阶段
        if s1 < 330:  # 逐渐打开降落伞
            s1 += 10
            print(f"Time: {u[i, 0]} s,Parachute area: {s1} m²")  # 打印降落伞的当前面积
        force = 0.5 * density_interp(u[i, 1] / 1000) * s0 * Cd0 * u[i, 5] * u[i, 2] ** 2 + 0.5 * density_interp(
            u[i, 1] / 1000) * s1 * Cd1 * u[i, 5] * u[i, 2] ** 2  # 计算空气阻力，考虑降落伞的影响
    f = np.array([0, u[i, 2], (force / m) - g, 0, 0, 0])  # 计算状态量的变化率
    u = np.vstack([u, integra(u[i, :], f, dt)])  # 使用欧拉法更新状态量
    u[i + 1, 0] = u[i, 0] + dt  # 更新时间
    u[i + 1, 3] = velocity_sound(u[i, 1])  # 更新音速
    u[i + 1, 4] = Mach(u[i, 2], u[i, 1])  # 更新马赫数
    u[i + 1, 5] = compressibility_factor(u[i + 1, 4])  # 更新可压缩性因子

    i += 1  # 进入下一个时间步

i_final = i  # 记录最终时间步
print(f"Time: {u[i, 0]} s, Acceleration: {(force / m) - g} m/s²")
print(f"Final velocity: {u[i, 2]} m/s")
print(f"The loop ran {i} times.")

# ----------------------------------

# 将所有数据绘制在一个图表中
# 绘制高度随时间变化曲线
plt.rcParams.update({'font.size': 15})
plt.style.use('default')  # 使用默认的绘图风格
fig = plt.figure(figsize=(16, 8))  # 调整大小为横向

# 绘制高度随时间变化曲线
ax = fig.add_subplot(121)  # 1行2列的第一个子图
ax.set_xlabel('Time (s)')
ax.set_ylabel('Altitude (m)')
ax.set_title('Spacecraft Altitude vs. Time')
ax.set_xlim([0, 140])  # 修改为0到130的时间跨度
ax.set_ylim([0, 36000])
line, = ax.plot(u[:, 0], u[:, 1], 'r-', lw=3)  # 绘制高度-时间曲线，颜色为红色，线宽为3

# 绘制速度随时间变化曲线
ax2 = fig.add_subplot(122)  # 1行2列的第二个子图
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_title('Spacecraft Velocity vs. Time')
ax2.set_xlim([0, 140])  # 修改为0到130的时间跨度
ax2.set_ylim([0, 1000])
line2, = ax2.plot(u[:, 0], -u[:, 2], lw=3)  # 绘制速度-时间曲线，线宽为3

# -----------------------------------

# 绘制声速、马赫数、可压缩性因子随时间变化曲线
plt.rcParams.update({'font.size': 15})
plt.style.use('default')  # 使用默认的绘图风格
fig = plt.figure(figsize=(20, 10))

# 声速-时间曲线
ax = fig.add_subplot(131)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Speed of Sound (m/s)')
ax.set_title('Speed of Sound vs. Time')
ax.set_xlim([0, 140])  # 修改为0到130的时间跨度
ax.set_ylim([0, 260])
line, = ax.plot(u[:, 0], u[:, 3], 'r-', lw=3)  # 绘制声速-时间曲线，颜色为红色，线宽为3

# 马赫数-时间曲线
ax2 = fig.add_subplot(132)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Mach Number')
ax2.set_title('Spacecraft Mach Number vs. Time')
ax2.set_xlim([0, 140])  # 修改为0到130的时间跨度
ax2.set_ylim([0, 4])
line2, = ax2.plot(u[:, 0], u[:, 4], 'b-', lw=3)  # 绘制马赫数-时间曲线，颜色为蓝色，线宽为3

# 可压缩性因子-时间曲线
ax3 = fig.add_subplot(133)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Compressibility Factor')
ax3.set_title('Compressibility Factor vs. Time')
ax3.set_xlim([0, 140])  # 修改为0到130的时间跨度
ax3.set_ylim([0, 2])
line3, = ax3.plot(u[:, 0], u[:, 5], 'g-', lw=3)  # 绘制可压缩性因子-时间曲线，颜色为绿色，线宽为3

plt.show()  # 显示图形

# 绘制原始数据中的温度与高度关系
plt.style.use('default')  # 使用默认的绘图风格
plt.plot(np.abs(u[:, 2]), u[:, 1] / 1000)  # 绘制原始数据中的温度与高度关系
plt.xlabel("Velocity (m/s)")  # 设置x轴标签为高度
plt.ylabel("Altitude (km)")  # 设置y轴标签为温度
plt.title("Atmospheric entry and descent trajectory")  # 设置图表标题
plt.grid(True, linestyle='--', alpha=0.7)  # 网格线为虚线，透明度为0.7
plt.show()  # 显示图形
