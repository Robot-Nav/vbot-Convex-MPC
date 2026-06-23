"""
VBot validation 17: keyboard-controlled MPC in MuJoCo.

VBot 验证脚本 17：在 MuJoCo viewer 中用键盘实时修改 MPC 速度指令。

这个脚本和 ex16 的主要区别：
- ex16 是先跑完仿真，再按需回放。
- ex17 是打开 MuJoCo viewer 实时仿真，并通过键盘改变期望速度。

控制方式：
- W/S：增加/减小前进速度 x_vel
- A/D：增加/减小侧向速度 y_vel
- Q/E：增加/减小 yaw 角速度
- Space：速度和 yaw_rate 全部归零
- R：重置机器人姿态和控制器状态
- Esc：退出 viewer
"""
import os

# viewer 脚本一般不需要画 matplotlib 图；这里仍设置可写缓存目录，避免导入绘图工具时报权限警告。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import time
from dataclasses import dataclass

import mujoco as mj
import mujoco.viewer
import numpy as np

from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.gait import Gait
from convex_mpc.leg_controller import LegController
from convex_mpc.mujoco_vbot_model import MuJoCo_VBot_Model
from convex_mpc.vbot_robot_data import PinVBotModel


# -----------------------------
# 实时仿真设置
# -----------------------------

# MuJoCo 物理积分频率：1000 Hz。
SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ

# 低层腿控频率：200 Hz，每 5ms 计算一次 12 个关节力矩。
CTRL_HZ = 200
CTRL_DT = 1.0 / CTRL_HZ
CTRL_DECIM = SIM_HZ // CTRL_HZ

# viewer 同步频率。这个频率只影响画面刷新，不影响控制频率。
VIEWER_HZ = 60.0
VIEWER_DT = 1.0 / VIEWER_HZ

# -----------------------------
# 步态和 MPC 设置
# -----------------------------

# trot 频率和 duty。先用和 ex16 一样的保守参数。
GAIT_HZ = 3.0
GAIT_DUTY = 0.6
GAIT_T = 1.0 / GAIT_HZ

# 一个步态周期切 16 段，因此 MPC 每次预测约 0.333s。
MPC_DT = GAIT_T / 16
MPC_HZ = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))

# 初始位置和期望机身高度。
INITIAL_X_POS = 0.0
INITIAL_Y_POS = 0.0
DEFAULT_Z_POS = 0.462

# -----------------------------
# 键盘速度指令设置
# -----------------------------

# 每次按键对速度指令的增量。第一版故意设置得小一些，方便调试。
X_VEL_STEP = 0.05
Y_VEL_STEP = 0.03
YAW_RATE_STEP = 0.15

# 速度限幅。VBot MPC 参数还没充分整定，先不要给太激进的速度。
X_VEL_LIMIT = 0.30
Y_VEL_LIMIT = 0.15
YAW_RATE_LIMIT = 0.80


# -----------------------------
# 力矩限幅
# -----------------------------

# VBot XML 中 hip/thigh 约 17 Nm，calf 约 34 Nm。乘 0.9 留安全余量。
SAFETY = 0.9
TAU_LIM = SAFETY * np.array([
    17.0, 17.0, 34.0,  # FL
    17.0, 17.0, 34.0,  # FR
    17.0, 17.0, 34.0,  # RL
    17.0, 17.0, 34.0,  # RR
])

# 控制器内部 12 维向量顺序：[FL, FR, RL, RR]，每条腿 [hip, thigh, calf]。
LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}


@dataclass
class KeyboardCommand:
    """键盘实时修改的 body-frame 速度指令。"""

    x_vel: float = 0.0
    y_vel: float = 0.0
    yaw_rate: float = 0.0
    z_pos: float = DEFAULT_Z_POS
    reset_requested: bool = False
    exit_requested: bool = False

    def clamp(self):
        """把速度限制在安全范围内，避免一次按键后指令过大。"""
        self.x_vel = float(np.clip(self.x_vel, -X_VEL_LIMIT, X_VEL_LIMIT))
        self.y_vel = float(np.clip(self.y_vel, -Y_VEL_LIMIT, Y_VEL_LIMIT))
        self.yaw_rate = float(np.clip(self.yaw_rate, -YAW_RATE_LIMIT, YAW_RATE_LIMIT))

    def stop(self):
        """停止移动，但保持 trot 步态和机身高度。"""
        self.x_vel = 0.0
        self.y_vel = 0.0
        self.yaw_rate = 0.0

    def summary(self):
        return f"x={self.x_vel:+.2f} m/s, y={self.y_vel:+.2f} m/s, yaw={self.yaw_rate:+.2f} rad/s"


def make_key_callback(cmd: KeyboardCommand):
    """创建 MuJoCo viewer 的键盘回调函数。"""

    def key_callback(keycode: int):
        # MuJoCo 传入的是整数 keycode。普通字母键可以转成字符。
        try:
            key = chr(keycode).lower()
        except ValueError:
            key = ""

        if key == "w":
            cmd.x_vel += X_VEL_STEP
        elif key == "s":
            cmd.x_vel -= X_VEL_STEP
        elif key == "a":
            cmd.y_vel += Y_VEL_STEP
        elif key == "d":
            cmd.y_vel -= Y_VEL_STEP
        elif key == "q":
            cmd.yaw_rate += YAW_RATE_STEP
        elif key == "e":
            cmd.yaw_rate -= YAW_RATE_STEP
        elif key == "r":
            cmd.reset_requested = True
            print("\n[Keyboard] reset requested")
            return
        elif keycode == 32:  # Space
            cmd.stop()
            print(f"\n[Keyboard] stop -> {cmd.summary()}")
            return
        elif keycode == 256:  # Esc
            cmd.exit_requested = True
            print("\n[Keyboard] exit requested")
            return
        else:
            return

        cmd.clamp()
        print(f"\n[Keyboard] command -> {cmd.summary()}")

    return key_callback


def initialize_robot(vbot: PinVBotModel, mujoco_vbot: MuJoCo_VBot_Model):
    """把 Pinocchio 默认站姿同步到 MuJoCo，并清空速度/力矩。"""
    q_init = vbot.current_config.get_q()
    q_init[0], q_init[1] = INITIAL_X_POS, INITIAL_Y_POS
    mujoco_vbot.update_with_q_pin(q_init)
    mujoco_vbot.data.qvel[:] = 0.0
    mujoco_vbot.data.ctrl[:] = 0.0
    mj.mj_forward(mujoco_vbot.model, mujoco_vbot.data)
    mujoco_vbot.update_pin_with_mujoco(vbot)


def compute_control_tick(
    vbot: PinVBotModel,
    mujoco_vbot: MuJoCo_VBot_Model,
    leg_controller: LegController,
    traj: ComTraj,
    gait: Gait,
    mpc: CentroidalMPC,
    cmd: KeyboardCommand,
    ctrl_i: int,
    u_opt: np.ndarray,
):
    """执行一次 200Hz 控制 tick，返回新的力矩和 MPC 接触力序列。"""
    time_now_s = float(mujoco_vbot.data.time)

    # 1. 读取 MuJoCo 当前状态，同步到 Pinocchio。
    mujoco_vbot.update_pin_with_mujoco(vbot)

    # 2. 每隔若干个控制 tick 求一次 MPC。
    if (ctrl_i % STEPS_PER_MPC) == 0:
        traj.generate_traj(
            vbot,
            gait,
            time_now_s,
            cmd.x_vel,
            cmd.y_vel,
            cmd.z_pos,
            cmd.yaw_rate,
            time_step=MPC_DT,
        )
        sol = mpc.solve_QP(vbot, traj, False)
        n_horizon = traj.N
        w_opt = sol["x"].full().flatten()
        u_opt = w_opt[12 * n_horizon :].reshape((12, n_horizon), order="F")

    # 3. 只执行 MPC 预测接触力序列的第一列。
    mpc_force_now = u_opt[:, 0]

    # 4. 四条腿分别把接触力/摆腿轨迹转换成关节力矩。
    tau_raw = np.zeros(12, dtype=float)
    for leg, leg_slice in LEG_SLICE.items():
        out = leg_controller.compute_leg_torque(
            leg,
            vbot,
            gait,
            mpc_force_now[leg_slice],
            time_now_s,
        )
        tau_raw[leg_slice] = out.tau

    # 5. 力矩限幅，得到真正下发给 MuJoCo 的命令。
    tau_cmd = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
    return tau_cmd, u_opt


def main():
    cmd = KeyboardCommand()

    # Pinocchio 用于计算运动学/动力学，MuJoCo 用于实时仿真和 viewer。
    vbot = PinVBotModel()
    mujoco_vbot = MuJoCo_VBot_Model()
    mujoco_vbot.model.opt.timestep = SIM_DT
    initialize_robot(vbot, mujoco_vbot)

    leg_controller = LegController()
    traj = ComTraj(vbot)
    gait = Gait(GAIT_HZ, GAIT_DUTY)

    # 先用零速度生成一次轨迹，初始化 MPC 稀疏结构。
    traj.generate_traj(
        vbot,
        gait,
        0.0,
        cmd.x_vel,
        cmd.y_vel,
        cmd.z_pos,
        cmd.yaw_rate,
        time_step=MPC_DT,
    )
    mpc = CentroidalMPC(vbot, traj)
    u_opt = np.zeros((12, traj.N), dtype=float)

    print("VBot keyboard MPC control")
    print("Keys: W/S forward, A/D lateral, Q/E yaw, Space stop, R reset, Esc exit")
    print(f"Initial command -> {cmd.summary()}")

    ctrl_i = 0
    sim_i = 0
    tau_hold = np.zeros(12, dtype=float)
    next_viewer_sync_t = 0.0

    with mujoco.viewer.launch_passive(
        mujoco_vbot.model,
        mujoco_vbot.data,
        key_callback=make_key_callback(cmd),
    ) as viewer:
        viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco_vbot.base_bid
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90
        viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        last_wall_time = time.perf_counter()

        while viewer.is_running() and not cmd.exit_requested:
            loop_start = time.perf_counter()

            if cmd.reset_requested:
                initialize_robot(vbot, mujoco_vbot)
                leg_controller = LegController()
                traj = ComTraj(vbot)
                traj.generate_traj(
                    vbot,
                    gait,
                    float(mujoco_vbot.data.time),
                    cmd.x_vel,
                    cmd.y_vel,
                    cmd.z_pos,
                    cmd.yaw_rate,
                    time_step=MPC_DT,
                )
                mpc = CentroidalMPC(vbot, traj)
                u_opt = np.zeros((12, traj.N), dtype=float)
                tau_hold[:] = 0.0
                ctrl_i = 0
                sim_i = 0
                cmd.reset_requested = False

            # 200Hz 控制 tick：计算一次新关节力矩。
            if (sim_i % CTRL_DECIM) == 0:
                tau_hold, u_opt = compute_control_tick(
                    vbot,
                    mujoco_vbot,
                    leg_controller,
                    traj,
                    gait,
                    mpc,
                    cmd,
                    ctrl_i,
                    u_opt,
                )
                ctrl_i += 1

            # 1000Hz MuJoCo 物理积分。
            mj.mj_step1(mujoco_vbot.model, mujoco_vbot.data)
            mujoco_vbot.set_joint_torque(tau_hold)
            mj.mj_step2(mujoco_vbot.model, mujoco_vbot.data)
            sim_i += 1

            # viewer 只需要 60Hz 同步，过高会拖慢实时仿真。
            sim_time = float(mujoco_vbot.data.time)
            if sim_time + 1e-12 >= next_viewer_sync_t:
                viewer.sync()
                next_viewer_sync_t += VIEWER_DT

            # 简单实时限速：让仿真时间尽量接近真实时间。
            elapsed_wall = loop_start - last_wall_time
            sleep_time = SIM_DT - elapsed_wall
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_wall_time = loop_start


if __name__ == "__main__":
    main()
