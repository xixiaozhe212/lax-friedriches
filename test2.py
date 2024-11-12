import numpy as np
import matplotlib.pyplot as plt


# 定义常量
PI = 3.1415926
NUM_MESH_POINTS = 100
SPACE_LENGTH = 1.0
TIME_END = 2.0
MESH_LENGTH = SPACE_LENGTH / NUM_MESH_POINTS
TIME_STEP = 0.1 * MESH_LENGTH
RATIO = 0.5 * TIME_STEP / MESH_LENGTH

# 初始化计算步数
calculation_step = 0


def calculate_lf_flux(rho_final, u_final):
    """
    计算通量（Flux - L - F）函数

    Args:
        rho_final (numpy.ndarray): 密度数组
        u_final (numpy.ndarray): 速度数组

    Returns:
        rho_flux (numpy.ndarray): 密度通量数组
        u_flux (numpy.ndarray): 速度通量数组
    """
    rho_flux = rho_final * u_final
    u_flux = rho_final * u_final ** 2 + 0.5 * rho_final ** 2

    return rho_flux, u_flux


def update_flux(rho_final, u_final):
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

    rho_flux, u_flux = calculate_lf_flux(rho_final, u_final)

    for i, _ in enumerate(rho_final[1:-1], start=1):
        rho_tmp[i] = 0.5 * (rho_final[i - 1] + rho_final[i + 1]) - RATIO * (rho_flux[i + 1] - rho_flux[i - 1])
        u_tmp[i] = 0.5 * (u_final[i - 1] + u_final[i + 1]) - RATIO * (u_flux[i + 1] - u_flux[i - 1])

    rho_final[:] = rho_tmp[:]
    u_final[:] = u_tmp[:]

    rho_final[0] = rho_final[1]
    rho_final[-1] = rho_final[-2]
    u_final[0] = u_final[1]
    u_final[-1] = u_final[-2]

    return rho_final, u_final


def initialize_conditions():
    """
    初始化条件函数

    Returns:
        rho_final (numpy.ndarray): 初始化后的密度数组
        u_final (numpy.ndarray): 初始化后的速度数组
    """
    x_values = np.linspace(0, SPACE_LENGTH, NUM_MESH_POINTS + 1)

    rho_final = np.where(np.abs(x_values) < 0.5, 1.0, 0.1)
    u_final = np.zeros(NUM_MESH_POINTS + 1)

    return rho_final, u_final


if __name__ == "__main__":
    # 初始化用于计算的数组
    rho_final, u_final = initialize_conditions()

    # 创建用于绘制图形的Figure和Axes对象
    fig, ax = plt.subplots()

    # 设置y轴初始范围为0到1
    ax.set_ylim(0, 1)

    # 计算并绘制图片
    for t in np.arange(0, TIME_END + TIME_STEP, TIME_STEP):
        # 获取界面通量
        rho_flux, u_flux = calculate_lf_flux(rho_final, u_final)

        # 计算数值解
        rho_final, u_final = update_flux(rho_final, u_final)

        

        if calculation_step % 100 == 0:
            ax.clear()
            rho_max = np.max([rho_final, u_final])
            rho_min = np.min([rho_final, u_final])

            # 根据数据的最大最小值以及预设的y轴范围，动态调整y轴显示范围
            if rho_max > 1:
                ax.set_ylim(-0.01, rho_max)
            elif rho_min < 0:
                ax.set_ylim(rho_min, 1.1)
            else:
                ax.set_ylim(-0.01, 1.1)

            ax.plot(np.linspace(0, SPACE_LENGTH, NUM_MESH_POINTS + 1), rho_final, label='rho')
            ax.plot(np.linspace(0, SPACE_LENGTH, NUM_MESH_POINTS + 1), u_final, label='u')
            ax.set_xlabel('x')
            ax.set_ylabel('Value')
            ax.set_title(f'Current step: {calculation_step}, Current time: {t}')
            ax.legend()
            plt.draw()
            plt.pause(0.5)
            # 保存每一步的图片
            fig.savefig(f'step_{calculation_step}.png')
            
        calculation_step += 1

    plt.show()

    # 保存最终图片
    fig.savefig('final_result.png')