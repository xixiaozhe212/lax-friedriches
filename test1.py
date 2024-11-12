import numpy as np
import math


# 定义常量
pi = 3.1415926
nx = 100
L = 1.0
T_end = 1.0
dx = L / nx
dt = 0.1 * dx
r = 0.5 * dt / dx

# 初始化计算步数
cal_step = 0


def LF_Flux(rho_final, u_final):
    """
    计算通量（Flux - L - F）函数

    Args:
        rho_final (numpy.ndarray): 密度数组
        u_final (numpy.ndarray): 速度数组

    Returns:
        rho_flux (numpy.ndarray): 密度通量数组
        u_flux (numpy.ndarray): 速度通量数组
    """
    rho_flux = np.zeros_like(rho_final)
    u_flux = np.zeros_like(u_final)

    for i in range(len(rho_final)):
        rho_flux[i] = rho_final[i] * u_final[i]
        u_flux[i] = rho_final[i] * u_final[i] ** 2 + 0.5 * rho_final[i] ** 2

    return rho_flux, u_flux


def flux(rho_final, u_final):
    """
    更新函数

    Args:
        rho_final (numpy.ndarray): 密度数组
        u_final (numpy.ndarray): 速度数组

    Returns:
        rho_final (numpy.ndarray): 更新后的密度数组
        u_final (numpy.ndarray): 更新后的速度数组
    """
    rho_tmp = np.zeros_like(rho_final)
    u_tmp = np.zeros_like(u_final)

    for i in range(1, nx):
        rho_tmp[i] = 0.5 * (rho_final[i - 1] + rho_final[i + 1]) - r * (rho_flux[i + 1] - rho_flux[i - 1])
        u_tmp[i] = 0.5 * (u_final[i - 1] + u_final[i + 1]) - r * (u_flux[i + 1] - u_flux[i - 1])

    rho_final[:] = rho_tmp[:]
    u_final[:] = u_tmp[:]

    rho_final[0] = rho_final[1]
    rho_final[nx] = rho_final[nx - 1]
    u_final[0] = u_final[1]
    u_final[nx] = u_final[nx - 1]

    return rho_final, u_final


def initial_condition():
    """
    初始化条件函数

    Returns:
        rho_final (numpy.ndarray): 初始化后的密度数组
        u_final (numpy.ndarray): 初始化后的速度数组
    """
    rho_final = np.zeros(nx + 1)
    u_final = np.zeros(nx + 1)

    for i in range(nx + 1):
        x = i * dx

        if math.fabs(x) < 0.5:
            rho_final[i] = 1.0
            u_final[i] = 0.0
        else:
            rho_final[i] = 0.1
            u_final[i] = 0.0

    return rho_final, u_final


def write_plot(u_final, type_str):
    """
    将最终数据写成Tecplot文件格式的函数

    Args:
        u_final (numpy.ndarray): 要写入的数组（可以是密度或速度数组）
        type_str (str): 数据类型标识（如"rho"或"u"）
    """
    global cal_step
    filename = f"data_{type_str}{cal_step}.plt"

    print(f"Writing {filename}....")

    with open(filename, 'w') as fout1:
        fout1.write("TITLE = \"  1-dimension linearize gas dynamic equation system \"\n")
        fout1.write("VARIABLES = \"x\", \"f-cal\"\n")
        fout1.write("ZONE  T=\"S_time\", F=POINT\n")

        for j in range(nx + 1):
            m = j * dx
            fout1.write(f"{m}\t{u_final[j]}\n")

    print("finished!")


if __name__ == "__main__":
    # 初始化用于计算的数组
    rho_final, u_final = initial_condition()

    # 计算并输出可视化文件
    for t in np.arange(0, T_end + dt, dt):
        # 获取界面通量
        rho_flux, u_flux = LF_Flux(rho_final, u_final)

        # 计算数值解
        rho_final, u_final = flux(rho_final, u_final)

        if cal_step % 100 == 0:
            print(f"Current step is: {cal_step}      Current time is :{t}      Next step is:{cal_step + 1}")

            write_plot(rho_final, "rho")
            write_plot(u_final, "u")

        cal_step += 1

    write_plot(rho_final, "rho")
    write_plot(u_final, "u")