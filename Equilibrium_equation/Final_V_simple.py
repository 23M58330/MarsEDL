import math
import matplotlib.pyplot as plt


def estimate_velocity_precise(mass, gravity, rho, C_d_parachute, S_parachute, C_d_weight, S_weight):
    # 计算重力 mg
    mg = mass * gravity

    # 计算总面积 S_eq
    S_eq = S_parachute  # 计算等效面积

    # 计算等效阻力系数 C_d_eq
    C_d_eq = (C_d_parachute * S_parachute + C_d_weight * S_weight) / S_eq

    # 根据总阻力计算速度 v
    v = math.sqrt((2 * mg) / (rho * C_d_eq * S_eq))

    return v


# 示例输入
mass = 2.0  # kg
gravity = 3.73  # m/s²
rho = 0.0126  # kg/m³
C_d_weight = 1.3  # Drag coefficient for weight
S_weight = 0.1  # Area of the weight in m²

# 曲线1：固定C_d_parachute = 0.75，S_parachute从5到60变化
C_d_parachute_fixed = 0.75
S_parachute_values = [i for i in range(5, 61)]  # 从5到60连续变化
velocities_S = [
    estimate_velocity_precise(mass, gravity, rho, C_d_parachute_fixed, S, C_d_weight, S_weight)
    for S in S_parachute_values
]

# 曲线2：固定S_parachute = 60.0，C_d_parachute从0.5到0.75变化
S_parachute_fixed = 60.0
C_d_parachute_values = [i / 100 for i in range(50, 76)]  # 从0.5到0.75连续变化
velocities_Cd = [
    estimate_velocity_precise(mass, gravity, rho, C_d, S_parachute_fixed, C_d_weight, S_weight)
    for C_d in C_d_parachute_values
]

# 绘制结果
plt.figure(figsize=(12, 6))

# 子图1
plt.subplot(1, 2, 1)
plt.plot(S_parachute_values, velocities_S, label="v vs S_parachute", color="blue")
plt.xlabel("S_parachute (m²)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity vs S_parachute (C_d_parachute = 0.75)")
plt.grid(True)
plt.legend()

# 子图2
plt.subplot(1, 2, 2)
plt.plot(C_d_parachute_values, velocities_Cd, label="v vs Cd_parachute", color="green")
plt.xlabel("C_d_parachute")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity vs C_d_parachute (S_parachute = 60.0)")
plt.grid(True)
plt.legend()

# 显示图像
plt.tight_layout()
plt.show()
