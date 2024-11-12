import numpy as np
import matplotlib.pyplot as plt

# 定义常量
NUM_MESH_POINTS = 200
SPACE_LENGTH = 1.0
TIME_END = 1.0
MESH_LENGTH = SPACE_LENGTH / NUM_MESH_POINTS
TIME_STEP = 0.1 * MESH_LENGTH
RATIO = 0.5 * TIME_STEP / MESH_LENGTH

# 初始化计算步数
calculation_step = 0

def calculate_lf_flux(rho_final, u_final):
    rho_flux = rho_final * u_final
    u_flux = rho_final * u_final ** 2 + 0.5 * rho_final ** 2

    return rho_flux, u_flux

def update_flux(rho_final, u_final):
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
    x_values = np.linspace(0, SPACE_LENGTH, NUM_MESH_POINTS + 1)

    rho_final = np.where(np.abs(x_values) < 0.5, 1.0, 0.1)
    u_final = np.zeros(NUM_MESH_POINTS + 1)

    return rho_final, u_final


if __name__ == "__main__":
    rho_final, u_final = initialize_conditions()

    fig, ax = plt.subplots()

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

            ax.set_ylim(-0.01, 1.1)

            ax.plot(np.linspace(0, SPACE_LENGTH, NUM_MESH_POINTS + 1), rho_final, label='rho')
            ax.plot(np.linspace(0, SPACE_LENGTH, NUM_MESH_POINTS + 1), u_final, label='u')
            ax.set_xlabel('x')
            ax.set_ylabel('Value')
            ax.set_title(f'Current step: {calculation_step}, Current time: {t}')
            ax.legend()
            plt.draw()
            plt.pause(0.5)
            fig.savefig(f'step_{calculation_step}.png')
            
        calculation_step += 1

    plt.show()