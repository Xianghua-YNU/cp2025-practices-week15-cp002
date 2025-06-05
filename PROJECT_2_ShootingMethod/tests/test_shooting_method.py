import time  # 添加在文件开头

def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    """
    Compare shooting method and scipy.solve_bvp, generate comparison plot.
    """
    print("Solving BVP using both methods...")
    try:
        # 统计打靶法计算时间
        print("Running shooting method...")
        t0 = time.time()
        x_shoot, y_shoot = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
        t1 = time.time()
        shooting_time = t1 - t0

        # 统计scipy.solve_bvp计算时间
        print("Running scipy.solve_bvp...")
        t2 = time.time()
        x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points//2)
        t3 = time.time()
        scipy_time = t3 - t2

        # 插值对比
        y_scipy_interp = np.interp(x_shoot, x_scipy, y_scipy)
        max_diff = np.max(np.abs(y_shoot - y_scipy_interp))
        rms_diff = np.sqrt(np.mean((y_shoot - y_scipy_interp)**2))

        plt.figure(figsize=(12, 8))
        # 主解对比图
        plt.subplot(2, 1, 1)
        plt.plot(x_shoot, y_shoot, 'b-', linewidth=2, label='Shooting Method')
        plt.plot(x_scipy, y_scipy, 'r--', linewidth=2, label='scipy.solve_bvp')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('Comparison of BVP Solution Methods')
        plt.plot([x_span[0], x_span[1]], [boundary_conditions[0], boundary_conditions[1]],
                 'ko', markersize=8, label='Boundary Conditions')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 差值图
        plt.subplot(2, 1, 2)
        plt.plot(x_shoot, y_shoot - y_scipy_interp, 'g-', linewidth=2, label='Difference')
        plt.xlabel('x')
        plt.ylabel('Difference (Shooting - scipy)')
        plt.title(f'Solution Difference (Max: {max_diff:.2e}, RMS: {rms_diff:.2e})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(r'C:\Users\31025\OneDrive\桌面\t\compare_methods.png')
        # plt.show()

        print(f"\nShooting method time: {shooting_time:.6f} s")
        print(f"scipy.solve_bvp time: {scipy_time:.6f} s")
        print("\nSolution Analysis:")
        print(f"Maximum difference: {max_diff:.2e}")
        print(f"RMS difference: {rms_diff:.2e}")
        print(f"Shooting method points: {len(x_shoot)}")
        print(f"scipy.solve_bvp points: {len(x_scipy)}")
        print(f"\nBoundary condition verification:")
        print(f"Shooting method: u({x_span[0]}) = {y_shoot[0]:.6f}, u({x_span[1]}) = {y_shoot[-1]:.6f}")
        print(f"scipy.solve_bvp: u({x_span[0]}) = {y_scipy[0]:.6f}, u({x_span[1]}) = {y_scipy[-1]:.6f}")
        print(f"Target: u({x_span[0]}) = {boundary_conditions[0]}, u({x_span[1]}) = {boundary_conditions[1]}")
        return {
            'x_shooting': x_shoot,
            'y_shooting': y_shoot,
            'x_scipy': x_scipy,
            'y_scipy': y_scipy,
            'max_difference': max_diff,
            'rms_difference': rms_diff,
            'boundary_error_shooting': [abs(y_shoot[0] - boundary_conditions[0]),
                                        abs(y_shoot[-1] - boundary_conditions[1])],
            'boundary_error_scipy': [abs(y_scipy[0] - boundary_conditions[0]),
                                     abs(y_scipy[-1] - boundary_conditions[1])],
            'shooting_time': shooting_time,
            'scipy_time': scipy_time
        }
    except Exception as e:
        print(f"Error in method comparison: {str(e)}")
        raise
import time  # 在文件开头导入

def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    print("Solving BVP using both methods...")
    try:
        # 统计打靶法计算时间
        print("Running shooting method...")
        t0 = time.time()
        x_shoot, y_shoot = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
        t1 = time.time()
        shooting_time = t1 - t0

        # 统计scipy.solve_bvp计算时间
        print("Running scipy.solve_bvp...")
        t2 = time.time()
        x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points//2)
        t3 = time.time()
        scipy_time = t3 - t2

        # ...后续代码不变...

        print(f"\nShooting method time: {shooting_time:.6f} s")
        print(f"scipy.solve_bvp time: {scipy_time:.6f} s")

        return {
            'x_shooting': x_shoot,
            'y_shooting': y_shoot,
            'x_scipy': x_scipy,
            'y_scipy': y_scipy,
            'max_difference': max_diff,
            'rms_difference': rms_diff,
            'boundary_error_shooting': [abs(y_shoot[0] - boundary_conditions[0]),
                                        abs(y_shoot[-1] - boundary_conditions[1])],
            'boundary_error_scipy': [abs(y_scipy[0] - boundary_conditions[0]),
                                     abs(y_scipy[-1] - boundary_conditions[1])],
            'shooting_time': shooting_time,
            'scipy_time': scipy_time
        }
    except Exception as e:
        print(f"Error in method comparison: {str(e)}")
        raise
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，确保图片可以保存
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')


def ode_system_shooting(y, t=None):
    if isinstance(y, (int, float)) and hasattr(t, '__len__'):
        t, y = y, t
    elif t is None:
        pass
    return [y[1], -np.pi*(y[0]+1)/4]


def boundary_conditions_scipy(ya, yb):
    return np.array([ya[0] - 1, yb[0] - 1])


def ode_system_scipy(x, y):
    return np.vstack((y[1], -np.pi*(y[0]+1)/4))


def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=10, tolerance=1e-6):
    if len(x_span) != 2 or x_span[1] <= x_span[0]:
        raise ValueError("x_span must be a tuple (x_start, x_end) with x_end > x_start")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be a tuple (u_left, u_right)")
    if n_points < 10:
        raise ValueError("n_points must be at least 10")
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    x = np.linspace(x_start, x_end, n_points)
    m1 = -1.0
    y0 = [u_left, m1]
    sol1 = odeint(ode_system_shooting, y0, x)
    u_end_1 = sol1[-1, 0]
    if abs(u_end_1 - u_right) < tolerance:
        return x, sol1[:, 0]
    m2 = m1 * u_right / u_end_1 if abs(u_end_1) > 1e-12 else m1 + 1.0
    y0[1] = m2
    sol2 = odeint(ode_system_shooting, y0, x)
    u_end_2 = sol2[-1, 0]
    if abs(u_end_2 - u_right) < tolerance:
        return x, sol2[:, 0]
    for iteration in range(max_iterations):
        if abs(u_end_2 - u_end_1) < 1e-12:
            m3 = m2 + 0.1
        else:
            m3 = m2 + (u_right - u_end_2) * (m2 - m1) / (u_end_2 - u_end_1)
        y0[1] = m3
        sol3 = odeint(ode_system_shooting, y0, x)
        u_end_3 = sol3[-1, 0]
        if abs(u_end_3 - u_right) < tolerance:
            return x, sol3[:, 0]
        m1, m2 = m2, m3
        u_end_1, u_end_2 = u_end_2, u_end_3
    print(f"Warning: Shooting method did not converge after {max_iterations} iterations.")
    print(f"Final boundary error: {abs(u_end_3 - u_right):.2e}")
    return x, sol3[:, 0]


def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    if len(x_span) != 2 or x_span[1] <= x_span[0]:
        raise ValueError("x_span must be a tuple (x_start, x_end) with x_end > x_start")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions must be a tuple (u_left, u_right)")
    if n_points < 5:
        raise ValueError("n_points must be at least 5")
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    x_init = np.linspace(x_start, x_end, n_points)
    y_init = np.zeros((2, x_init.size))
    y_init[0] = u_left + (u_right - u_left) * (x_init - x_start) / (x_end - x_start)
    y_init[1] = (u_right - u_left) / (x_end - x_start)
    sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x_init, y_init)
    if not sol.success:
        raise RuntimeError(f"scipy.solve_bvp failed: {sol.message}")
    x_fine = np.linspace(x_start, x_end, 100)
    y_fine = sol.sol(x_fine)[0]
    return x_fine, y_fine


def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    print("Solving BVP using both methods...")
    try:
        print("Running shooting method...")
        x_shoot, y_shoot = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
        print("Running scipy.solve_bvp...")
        x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points//2)
        y_scipy_interp = np.interp(x_shoot, x_scipy, y_scipy)
        max_diff = np.max(np.abs(y_shoot - y_scipy_interp))
        rms_diff = np.sqrt(np.mean((y_shoot - y_scipy_interp)**2))

        plt.figure(figsize=(12, 8))
        # Main comparison plot
        plt.subplot(2, 1, 1)
        plt.plot(x_shoot, y_shoot, 'b-', linewidth=2, label='Shooting Method')
        plt.plot(x_scipy, y_scipy, 'r--', linewidth=2, label='scipy.solve_bvp')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('Comparison of BVP Solution Methods')
        plt.plot([x_span[0], x_span[1]], [boundary_conditions[0], boundary_conditions[1]],
                 'ko', markersize=8, label='Boundary Conditions')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Difference plot
        plt.subplot(2, 1, 2)
        plt.plot(x_shoot, y_shoot - y_scipy_interp, 'g-', linewidth=2, label='Difference')
        plt.xlabel('x')
        plt.ylabel('Difference (Shooting - scipy)')
        plt.title(f'Solution Difference (Max: {max_diff:.2e}, RMS: {rms_diff:.2e})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        # 保存图片到指定路径
        plt.savefig(r'C:\Users\31025\OneDrive\桌面\t\compare_methods.png')
        # plt.show()  # 如需显示可取消注释

        print("\nSolution Analysis:")
        print(f"Maximum difference: {max_diff:.2e}")
        print(f"RMS difference: {rms_diff:.2e}")
        print(f"Shooting method points: {len(x_shoot)}")
        print(f"scipy.solve_bvp points: {len(x_scipy)}")

        print(f"\nBoundary condition verification:")
        print(f"Shooting method: u({x_span[0]}) = {y_shoot[0]:.6f}, u({x_span[1]}) = {y_shoot[-1]:.6f}")
        print(f"scipy.solve_bvp: u({x_span[0]}) = {y_scipy[0]:.6f}, u({x_span[1]}) = {y_scipy[-1]:.6f}")
        print(f"Target: u({x_span[0]}) = {boundary_conditions[0]}, u({x_span[1]}) = {boundary_conditions[1]}")
        return {
            'x_shooting': x_shoot,
            'y_shooting': y_shoot,
            'x_scipy': x_scipy,
            'y_scipy': y_scipy,
            'max_difference': max_diff,
            'rms_difference': rms_diff,
            'boundary_error_shooting': [abs(y_shoot[0] - boundary_conditions[0]),
                                        abs(y_shoot[-1] - boundary_conditions[1])],
            'boundary_error_scipy': [abs(y_scipy[0] - boundary_conditions[0]),
                                     abs(y_scipy[-1] - boundary_conditions[1])]
        }
    except Exception as e:
        print(f"Error in method comparison: {str(e)}")
        raise


def test_ode_system():
    print("Testing ODE system...")
    t_test = 0.5
    y_test = np.array([1.0, 0.5])
    dydt = ode_system_shooting(y_test, t_test)
    expected = [0.5, -np.pi*(1.0+1)/4]
    print(f"ODE system (shooting): dydt = {dydt}")
    print(f"Expected: {expected}")
    assert np.allclose(dydt, expected), "Shooting ODE system test failed"
    dydt_scipy = ode_system_scipy(t_test, y_test)
    expected_scipy = np.array([[0.5], [-np.pi*2/4]])
    print(f"ODE system (scipy): dydt = {dydt_scipy.flatten()}")
    print(f"Expected: {expected_scipy.flatten()}")
    assert np.allclose(dydt_scipy, expected_scipy), "Scipy ODE system test failed"
    print("ODE system tests passed!")


def test_boundary_conditions():
    print("Testing boundary conditions...")
    ya = np.array([1.0, 0.5])
    yb = np.array([1.0, -0.3])
    bc_residual = boundary_conditions_scipy(ya, yb)
    expected = np.array([0.0, 0.0])
    print(f"Boundary condition residuals: {bc_residual}")
    print(f"Expected: {expected}")
    assert np.allclose(bc_residual, expected), "Boundary conditions test failed"
    print("Boundary conditions test passed!")


def test_shooting_method():
    print("Testing shooting method...")
    x_span = (0, 1)
    boundary_conditions = (1, 1)
    x, y = solve_bvp_shooting_method(x_span, boundary_conditions, n_points=50)
    assert abs(y[0] - boundary_conditions[0]) < 1e-6, "Left boundary condition not satisfied"
    assert abs(y[-1] - boundary_conditions[1]) < 1e-6, "Right boundary condition not satisfied"
    print(f"Shooting method: u(0) = {y[0]:.6f}, u(1) = {y[-1]:.6f}")
    print("Shooting method test passed!")


def test_scipy_method():
    print("Testing scipy.solve_bvp wrapper...")
    x_span = (0, 1)
    boundary_conditions = (1, 1)
    x, y = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=20)
    assert abs(y[0] - boundary_conditions[0]) < 1e-6, "Left boundary condition not satisfied"
    assert abs(y[-1] - boundary_conditions[1]) < 1e-6, "Right boundary condition not satisfied"
    print(f"scipy.solve_bvp: u(0) = {y[0]:.6f}, u(1) = {y[-1]:.6f}")
    print("scipy.solve_bvp wrapper test passed!")


if __name__ == "__main__":
    print("项目2：打靶法与scipy.solve_bvp求解边值问题 - 参考答案")
    print("=" * 60)
    print("Running unit tests...")
    test_ode_system()
    test_boundary_conditions()
    test_shooting_method()
    test_scipy_method()
    print("All unit tests passed!\n")
    print("Running method comparison...")
    results = compare_methods_and_plot()
    print("\n项目2完成！所有功能已实现并测试通过。")
