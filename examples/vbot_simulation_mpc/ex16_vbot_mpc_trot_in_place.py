"""
VBot validation 16: MPC trot in place.

VBot 验证脚本 16：使用 MPC 控制机器人原地 trot。

This is the first bring-up target for the VBot MJCF adaptation:
zero commanded body velocity, zero yaw rate, and a fixed base height.
MuJoCo uses the full VBot model, while Pinocchio uses the simplified
MJCF at models/MJCF/vbot/vbot_pinocchio.xml.

这是 VBot 适配后的第一个闭环验证目标：
- 期望前进/侧向速度为 0
- 期望 yaw 角速度为 0
- 期望机身高度固定

注意这里有两个模型：
- MuJoCo：加载完整 VBot 模型，用于物理仿真、接触和可视化。
- Pinocchio：加载精简 MJCF，只保留 base 和四条腿，用于运动学/动力学计算。
"""
import os

# 避免无显示环境下 matplotlib 写到不可写的 home 配置目录。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# 默认不弹图形窗口；需要画图时用 VBOT_MPC_PLOTS=1。
os.environ.setdefault("MPLBACKEND", "Agg")

import time
from dataclasses import dataclass, field

import mujoco as mj
import numpy as np

from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.gait import Gait
from convex_mpc.leg_controller import LegController
from convex_mpc.mujoco_vbot_model import MuJoCo_VBot_Model
from convex_mpc.plot_helper import (
    hold_until_all_fig_closed,
    plot_mpc_result,
    plot_solve_time,
    plot_swing_foot_traj,
)
from convex_mpc.vbot_robot_data import PinVBotModel


# -----------------------------
# 仿真/回放设置
# -----------------------------

# 总仿真时间。当前用于原地 trot 初步验证，先短时间跑通闭环。
RUN_SIM_LENGTH_S = 5.0

# 回放采样频率。控制和仿真会更高频，回放只需要记录较少帧。
RENDER_HZ = 120.0
RENDER_DT = 1.0 / RENDER_HZ
REALTIME_FACTOR = 1 # 播放倍率 1.0 表示尽量实时回放，<1.0 表示慢放，>1.0 表示快放。

# 是否打开 MuJoCo viewer 回放。默认关闭，避免脚本在服务器/终端环境卡住。
# 使用方式：VBOT_MPC_REPLAY=1 python -m examples.ex16_vbot_mpc_trot_in_place
REPLAY = os.environ.get("VBOT_MPC_REPLAY", "0") == "1"

# 是否画 MPC/力矩/足端轨迹图。默认关闭，使用方式同上。
PLOTS = os.environ.get("VBOT_MPC_PLOTS", "0") == "1"

# -----------------------------
# 三层频率设置
# -----------------------------

# MuJoCo 物理积分频率：1000 Hz。
SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ

# 腿部低层控制频率：200 Hz。每个控制 tick 计算一次关节力矩。
CTRL_HZ = 200
CTRL_DT = 1.0 / CTRL_HZ

# -----------------------------
# 步态和 MPC 预测周期
# -----------------------------

# trot 频率和支撑相占比。duty=0.6 表示每条腿一个周期中 60% 时间在支撑相。
GAIT_HZ = 3.0
GAIT_DUTY = 0.6
GAIT_T = 1.0 / GAIT_HZ

# MPC 预测一个完整步态周期，并把这个周期切成 16 段。
# 因此 horizon N=16，MPC_DT 约为 0.0208s，MPC 频率约 48Hz。
MPC_DT = GAIT_T / 16
MPC_HZ = 1.0 / MPC_DT

# 控制循环每隔多少个 tick 更新一次 MPC。
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))

# -----------------------------
# 原地 trot 指令
# -----------------------------

# 初始平面位置。
INITIAL_X_POS = 0.0
INITIAL_Y_POS = 0.0

# 期望机身高度。VBot 当前模型默认站姿 base 高度约 0.462m。
Z_POS_DES_BODY = 0.462

# 原地 trot：期望前进速度、侧向速度、yaw 角速度全为 0。
X_VEL_DES_BODY = 0.0
Y_VEL_DES_BODY = 0.0
YAW_RATE_DES_BODY = 0.0

# 仿真循环和控制循环的降采样关系。
CTRL_DECIM = SIM_HZ // CTRL_HZ
SIM_STEPS = int(RUN_SIM_LENGTH_S * SIM_HZ)
CTRL_STEPS = int(RUN_SIM_LENGTH_S * CTRL_HZ)

# -----------------------------
# 力矩限幅
# -----------------------------

# VBot XML 中 hip/thigh 电机约 17 Nm，calf 电机约 34 Nm。
# SAFETY=0.9 表示实际命令只使用 90% 限幅，给仿真/实物保留余量。
SAFETY = 0.9
TAU_LIM = SAFETY * np.array([
    # 力矩限幅
    17.0, 17.0, 34.0,  # FL
    17.0, 17.0, 34.0,  # FR
    17.0, 17.0, 34.0,  # RL
    17.0, 17.0, 34.0,  # RR
])

# 控制器内部的 12 维向量顺序固定为：
# [FL_hip, FL_thigh, FL_calf, FR_..., RL_..., RR_...]
LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}


@dataclass
class FootTraj:
    """控制频率下的足端轨迹日志，用于调试摆腿轨迹和实际足端运动。"""

    pos_des: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
    pos_now: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
    vel_des: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
    vel_now: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))


def main():
    # PinVBotModel：Pinocchio 模型，用于质心、惯量、雅可比、动力学项计算。
    vbot = PinVBotModel()

    # MuJoCo_VBot_Model：MuJoCo 模型，用于物理积分、接触、力矩下发和回放。
    mujoco_vbot = MuJoCo_VBot_Model()

    # 腿部控制器：
    # - 支撑腿：将 MPC 接触力通过 J^T 映射成关节力矩。
    # - 摆动腿：笛卡尔空间 PD 跟踪足端摆腿轨迹。
    leg_controller = LegController()

    # 质心参考轨迹生成器和步态调度器。
    traj = ComTraj(vbot)
    gait = Gait(GAIT_HZ, GAIT_DUTY)

    # 设置机器人初始构型，并同步到 MuJoCo。
    q_init = vbot.current_config.get_q()
    q_init[0], q_init[1] = INITIAL_X_POS, INITIAL_Y_POS
    mujoco_vbot.update_with_q_pin(q_init)

    # 设置 MuJoCo 物理积分步长。
    mujoco_vbot.model.opt.timestep = SIM_DT

    # 先生成一次参考轨迹，用于初始化 MPC 的稀疏矩阵结构。
    traj.generate_traj(
        vbot,
        gait,
        0.0,
        X_VEL_DES_BODY,
        Y_VEL_DES_BODY,
        Z_POS_DES_BODY,
        YAW_RATE_DES_BODY,
        time_step=MPC_DT,
    )

    # 构建 MPC 求解器。这里会打印 QP 的 H/A 矩阵规模、变量数、约束数等信息。
    mpc = CentroidalMPC(vbot, traj)

    # MPC 输出的接触力序列，形状为 12 x N。
    # 每一列对应预测时域中的一个未来时间步。
    u_opt = np.zeros((12, traj.N), dtype=float)

    # -----------------------------
    # 控制频率日志
    # -----------------------------

    # 12 维质心状态：[px, py, pz, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    x_vec = np.zeros((12, CTRL_STEPS))

    # MPC 接触力日志，世界坐标系下四足各 3 维力。
    mpc_force_world = np.zeros((12, CTRL_STEPS))

    # tau_raw：未限幅关节力矩；tau_cmd：限幅后实际下发的力矩。
    tau_raw = np.zeros((12, CTRL_STEPS))
    tau_cmd = np.zeros((12, CTRL_STEPS))

    # 足端期望/实际位置速度日志。
    foot_traj = FootTraj()

    # MPC 单次更新矩阵和求解耗时，用于判断是否满足实时控制频率。
    mpc_update_time_ms = []
    mpc_solve_time_ms = []

    # -----------------------------
    # 回放日志
    # -----------------------------

    # 这些日志按 RENDER_HZ 采样，回放时比控制频率日志更轻。
    time_log_render = []
    q_log_render = []
    tau_log_render = []
    next_render_t = 0.0

    print(f"Running VBot MPC trot-in-place validation for {RUN_SIM_LENGTH_S:.1f}s")
    sim_start_time = time.perf_counter()
    ctrl_i = 0

    # tau_hold 保存最近一次控制 tick 计算出的力矩。
    # 在 1000Hz 仿真积分中，两个 200Hz 控制 tick 之间持续下发这个力矩。
    tau_hold = np.zeros(12, dtype=float)

    # -----------------------------
    # 主仿真循环：1000 Hz
    # -----------------------------
    for k in range(SIM_STEPS):
        time_now_s = float(mujoco_vbot.data.time)

        # 每隔 CTRL_DECIM 个 MuJoCo step 进入一次 200Hz 控制循环。
        if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:
            # 从 MuJoCo 读取当前真实/仿真状态，并同步到 Pinocchio。
            mujoco_vbot.update_pin_with_mujoco(vbot)

            # 计算并记录当前 12 维质心状态。
            x_vec[:, ctrl_i] = vbot.compute_com_x_vec().reshape(-1)

            # 每隔 STEPS_PER_MPC 个控制 tick 更新一次 MPC。
            # 原地 trot 中参考速度为 0，但 MPC 仍会根据接触时序分配支撑力。
            if (ctrl_i % STEPS_PER_MPC) == 0:
                print(f"\rSimulation Time: {time_now_s:.3f} s", end="", flush=True)

                # 根据当前状态、步态相位和期望速度生成未来 N 步参考轨迹。
                traj.generate_traj(
                    vbot,
                    gait,
                    time_now_s,
                    X_VEL_DES_BODY,
                    Y_VEL_DES_BODY,
                    Z_POS_DES_BODY,
                    YAW_RATE_DES_BODY,
                    time_step=MPC_DT,
                )

                # 求解 QP，得到未来 N 步的状态和接触力。
                sol = mpc.solve_QP(vbot, traj, False)
                mpc_solve_time_ms.append(mpc.solve_time)
                mpc_update_time_ms.append(mpc.update_time)

                # 决策变量 w = [X_0...X_N-1, U_0...U_N-1]。
                # 这里只取 U 部分，也就是四足接触力序列。
                n_horizon = traj.N
                w_opt = sol["x"].full().flatten()
                u_opt = w_opt[12 * n_horizon :].reshape((12, n_horizon), order="F")

            # 实际控制只用 MPC 预测序列的第一列，也就是当前时刻应施加的接触力。
            mpc_force_world[:, ctrl_i] = u_opt[:, 0]

            # 对四条腿分别计算关节力矩。
            for leg, leg_slice in LEG_SLICE.items():
                out = leg_controller.compute_leg_torque(
                    leg,
                    vbot,
                    gait,
                    mpc_force_world[leg_slice, ctrl_i],
                    time_now_s,
                )
                tau_raw[leg_slice, ctrl_i] = out.tau
                foot_traj.pos_des[leg_slice, ctrl_i] = out.pos_des
                foot_traj.pos_now[leg_slice, ctrl_i] = out.pos_now
                foot_traj.vel_des[leg_slice, ctrl_i] = out.vel_des
                foot_traj.vel_now[leg_slice, ctrl_i] = out.vel_now

            # 力矩限幅，防止命令超过 VBot 电机能力。
            tau_cmd[:, ctrl_i] = np.clip(tau_raw[:, ctrl_i], -TAU_LIM, TAU_LIM)

            # 保存本控制 tick 的力矩，供高频仿真积分阶段持续使用。
            tau_hold = tau_cmd[:, ctrl_i].copy()
            ctrl_i += 1

        # MuJoCo 分两步积分：
        # step1 计算当前状态相关项，set_joint_torque 写入控制量，step2 完成积分。
        mj.mj_step1(mujoco_vbot.model, mujoco_vbot.data)
        mujoco_vbot.set_joint_torque(tau_hold)
        mj.mj_step2(mujoco_vbot.model, mujoco_vbot.data)

        # 按回放频率记录 qpos 和力矩，避免每个 1000Hz step 都保存造成日志过大。
        t_after = float(mujoco_vbot.data.time)
        if t_after + 1e-12 >= next_render_t:
            time_log_render.append(t_after)
            q_log_render.append(mujoco_vbot.data.qpos.copy())
            tau_log_render.append(tau_hold.copy())
            next_render_t += RENDER_DT

    print(
        f"\nSimulation ended."
        f"\nElapsed time: {time.perf_counter() - sim_start_time:.3f}s"
        f"\nControl ticks: {ctrl_i}/{CTRL_STEPS}"
    )

    # 默认不画图；需要时设置 VBOT_MPC_PLOTS=1。
    if PLOTS:
        t_vec = np.arange(ctrl_i) * CTRL_DT
        plot_swing_foot_traj(t_vec, foot_traj, False)
        plot_mpc_result(t_vec, mpc_force_world, tau_cmd, x_vec, block=False)
        plot_solve_time(mpc_solve_time_ms, mpc_update_time_ms, MPC_DT, MPC_HZ, block=False)

    # 默认不打开 viewer；需要时设置 VBOT_MPC_REPLAY=1。
    if REPLAY:
        mujoco_vbot.replay_simulation(
            np.asarray(time_log_render, dtype=float),
            np.asarray(q_log_render, dtype=float),
            np.asarray(tau_log_render, dtype=float),
            RENDER_DT,
            REALTIME_FACTOR,
        )
        hold_until_all_fig_closed()


if __name__ == "__main__":
    main()
