"""
模块：双摆模拟解决方案
作者：由 Trae Assistant 生成
说明：完整的双摆动力学模拟、能量计算及动画展示解决方案。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation

# 常量
G_CONST = 9.81  # 重力加速度 (m/s^2)
L_CONST = 0.4   # 摆杆长度 (米)
M_CONST = 1.0   # 摆球质量 (千克)

def derivatives(y, t, L1, L2, m1, m2, g_param):
    """
    返回双摆系统状态变量 y 关于时间的导数。

    参数：
        y (列表或np.array)：当前状态向量 [theta1, omega1, theta2, omega2]。
        t (浮点数)：当前时间（该自治方程不显式依赖时间，但 odeint 要求提供）。
        L1 (浮点数)：第一个摆杆长度。
        L2 (浮点数)：第二个摆杆长度。
        m1 (浮点数)：第一个摆球质量。
        m2 (浮点数)：第二个摆球质量。
        g_param (浮点数)：重力加速度。

    返回：
        列表：时间导数 [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]。
    
    运动方程（简化版本，假设 L1=L2=L，m1=m2=m）：
    dtheta1_dt = omega1
    dtheta2_dt = omega2
    domega1_dt = (-omega1**2*np.sin(2*theta1-2*theta2) - 2*omega2**2*np.sin(theta1-theta2) - 
                  (g/L) * (np.sin(theta1-2*theta2) + 3*np.sin(theta1))) / (3 - np.cos(2*theta1-2*theta2))
    domega2_dt = (4*omega1**2*np.sin(theta1-theta2) + omega2**2*np.sin(2*theta1-2*theta2) + 
                  2*(g/L) * (np.sin(2*theta1-theta2) - np.sin(theta2))) / (3 - np.cos(2*theta1-2*theta2))
    """
    theta1, omega1, theta2, omega2 = y

    # 本题中假设 L1=L2=L，m1=m2=M，直接使用题目给出的简化公式。
    # 若 m1, m2, L1, L2 不同，则需要更一般的方程。
    # 这里按照题目简化条件处理。
    
    dtheta1_dt = omega1
    dtheta2_dt = omega2

    # domega1_dt 分子和分母
    num1 = -omega1**2 * np.sin(2*theta1 - 2*theta2) \
           - 2 * omega2**2 * np.sin(theta1 - theta2) \
           - (g_param/L1) * (np.sin(theta1 - 2*theta2) + 3*np.sin(theta1))
    den1 = 3 - np.cos(2*theta1 - 2*theta2)  # 假设 m1=m2，L1=L2
    # 若质量和长度不同，更通用的分母公式如下（未使用）：
    # den1_general = (m1 + m2) * L1 - m2 * L1 * np.cos(theta1 - theta2)**2 
    # 本题使用简化版本。
    
    domega1_dt = num1 / den1

    # domega2_dt 分子和分母
    num2 = 4 * omega1**2 * np.sin(theta1 - theta2) \
           + omega2**2 * np.sin(2*theta1 - 2*theta2) \
           + 2 * (g_param/L1) * (np.sin(2*theta1 - theta2) - np.sin(theta2))
    den2 = 3 - np.cos(2*theta1 - 2*theta2)  # 假设 m1=m2，L1=L2
    # 通用分母公式（未使用）：
    # den2_general = (m1 + m2) * L2 - m2 * L2 * np.cos(theta1 - theta2)**2
    # 依然使用简化公式。
    
    domega2_dt = num2 / den2
    
    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

# 将函数参数名中 g 改为 g_param 以避免与全局 G_CONST 冲突。
# 但原函数中 g 作为参数名也是可以的。
# 关键是使用全局变量时应明确传递或指定默认参数。

def solve_double_pendulum(initial_conditions, t_span, t_points, L_param=L_CONST, g_param=G_CONST):
    """
    求解双摆微分方程。

    参数：
        initial_conditions (字典)：{'theta1': 值, 'omega1': 值, 'theta2': 值, 'omega2': 值}，单位为弧度和弧度/秒。
        t_span (元组)：(起始时间, 结束时间)。
        t_points (整数)：时间点数量。
        L_param (浮点数)：摆杆长度。
        g_param (浮点数)：重力加速度。

    返回：
        元组：(时间数组, 解数组)
               时间数组：一维 numpy 数组
               解数组：二维 numpy 数组，包含每个时间点对应的状态 [theta1, omega1, theta2, omega2]
    """
    y0 = [initial_conditions['theta1'], initial_conditions['omega1'], 
          initial_conditions['theta2'], initial_conditions['omega2']]
    t_arr = np.linspace(t_span[0], t_span[1], t_points)
    
    # 使用 L_param 作为摆长，假设 L1=L2=L_param，质量为全局 M_CONST
    sol_arr = odeint(derivatives, y0, t_arr, args=(L_param, L_param, M_CONST, M_CONST, g_param), rtol=1e-9, atol=1e-9)
    return t_arr, sol_arr

def calculate_energy(sol_arr, L_param=L_CONST, m_param=M_CONST, g_param=G_CONST):
    """
    计算双摆系统的总能量。

    参数：
        sol_arr (np.array)：odeint 的结果数组（行对应时间点，列依次为 [theta1, omega1, theta2, omega2]）。
        L_param (浮点数)：摆杆长度。
        m_param (浮点数)：摆球质量。
        g_param (浮点数)：重力加速度。

    返回：
        np.array：每个时间点对应的总能量（标量数组）。
    """
    theta1 = sol_arr[:, 0]
    omega1 = sol_arr[:, 1]
    theta2 = sol_arr[:, 2]
    omega2 = sol_arr[:, 3]

    # 势能 V
    # V = -m*g*L*(2*cos(theta1) + cos(theta2))
    V = -m_param * g_param * L_param * (2 * np.cos(theta1) + np.cos(theta2))

    # 动能 T
    # T = m*L^2 * (omega1^2 + 0.5*omega2^2 + omega1*omega2*cos(theta1-theta2))
    T = m_param * L_param**2 * (omega1**2 + 0.5 * omega2**2 + omega1 * omega2 * np.cos(theta1 - theta2))
    
    return T + V

def animate_double_pendulum(t_arr, sol_arr, L_param=L_CONST, skip_frames=10):
    """
    创建双摆运动动画。

    参数：
        t_arr (np.array)：时间数组。
        sol_arr (np.array)：odeint 计算得到的状态数组。
        L_param (浮点数)：摆杆长度。
        skip_frames (整数)：动画每帧跳过的计算点数量。

    返回：
        matplotlib.animation.FuncAnimation：动画对象。
    """
    theta1_all = sol_arr[:, 0]
    theta2_all = sol_arr[:, 2]

    # 选择动画帧
    theta1_anim = theta1_all[::skip_frames]
    theta2_anim = theta2_all[::skip_frames]
    t_anim = t_arr[::skip_frames]

    # 计算笛卡尔坐标
    x1 = L_param * np.sin(theta1_anim)
    y1 = -L_param * np.cos(theta1_anim)
    x2 = x1 + L_param * np.sin(theta2_anim)
    y2 = y1 - L_param * np.cos(theta2_anim)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2*L_param - 0.1, 2*L_param + 0.1), ylim=(-2*L_param - 0.1, 0.1))
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlabel('x (米)')
    ax.set_ylabel('y (米)')
    ax.set_title('双摆动画')

    line, = ax.plot([], [], 'o-', lw=2, markersize=8, color='blue')  # 摆杆和摆球
    time_template = '时间 = %.1f秒'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % t_anim[i])
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=len(t_anim),
                                  interval=25, blit=True, init_func=init)
    return ani

if __name__ == "__main__":
    # 初始条件
    initial_conditions_rad = {
        'theta1': np.pi/2,  # 90度
        'omega1': 0.0,
        'theta2': np.pi/2,  # 90度
        'omega2': 0.0
    }
    t_start = 0
    t_end = 100
    t_points_sim = 2000  # 增加时间点数，提高能量守恒精度，默认odeint公差足够
    
    # 1. 求解微分方程
    print(f"求解 t = {t_start}s 到 {t_end}s 的微分方程...")
    t_solution, sol_solution = solve_double_pendulum(initial_conditions_rad, (t_start, t_end), t_points_sim, L_param=L_CONST, g_param=G_CONST)
    print("微分方程求解完成。")

    # 2. 计算能量
    print("计算系统能量...")
    energy_solution = calculate_energy(sol_solution, L_param=L_CONST, m_param=M_CONST, g_param=G_CONST)
    print("能量计算完成。")

    # 3. 绘制能量随时间变化图
    plt.figure(figsize=(10, 5))
    plt.plot(t_solution, energy_solution, label='总能量')
    plt.xlabel('时间 (秒)')
    plt.ylabel('能量 (焦耳)')
    plt.title('双摆系统总能量随时间变化')
    plt.grid(True)
    plt.legend()
    # 检查能量守恒
    initial_energy = energy_solution[0]
    final_energy = energy_solution[-1]
    energy_variation = np.max(energy_solution) - np.min(energy_solution)
    print(f"初始能量: {initial_energy:.7f} J")
    print(f"最终能量: {final_energy:.7f} J")
    print(f"最大能量变化: {energy_variation:.7e} J")
    if energy_variation < 1e-5:
        print("能量守恒目标（< 1e-5 J）达成。")
    else:
        print(f"能量守恒目标（< 1e-5 J）未达成。变化量: {energy_variation:.2e} J。可尝试增加时间点数或调整odeint公差。")
    plt.ylim(initial_energy - 5*energy_variation if energy_variation > 1e-7 else initial_energy - 1e-5, 
             initial_energy + 5*energy_variation if energy_variation > 1e-7 else initial_energy + 1e-5)
    plt.show()

    # 4. （可选）动画展示
    # 设置为 True 则运行动画，False 则跳过（动画可能较慢）
    run_animation = True 
    if run_animation:
        print("生成动画中... 这可能需要一点时间。")
        # 动画时通常使用较少的帧数以保证流畅播放
        anim_object = animate_double_pendulum(t_solution, sol_solution, L_param=L_CONST, skip_frames=max(1, t_points_sim // 1000) * 5) 
        
        # 若需保存动画，需安装 ffmpeg 或其他支持的写入器
        # 示例：anim_object.save('double_pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
        # print("动画已保存为 double_pendulum.mp4（需有写入器支持）")
        
        plt.show()  # 显示动画窗口
        print("动画展示完成。")
    else:
        print("跳过动画。")

    print("双摆模拟结束。")

"""
关于能量守恒与 odeint 参数说明：
为达到能量守恒误差 < 1e-5 J，可能需要：
1. 在 solve_double_pendulum 中增加时间点数（例如 5000、10000 或更多），
   这将减少 odeint 的平均时间步长。
2. 明确设置 odeint 的相对误差容限（rtol）和绝对误差容限（atol）为较小值，
   例如 rtol=1e-8, atol=1e-8，默认是约 1.49e-8。
   题目中默认步长可能不够严格，增加时间点和减小误差容限有助于提高精度。
本代码中默认使用 rtol=1e-9, atol=1e-9 以提高准确性。
"""
