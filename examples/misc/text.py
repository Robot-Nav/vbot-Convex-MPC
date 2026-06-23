#!/usr/bin/env python3
from pathlib import Path
import time
import argparse
import mujoco
import mujoco.viewer


REPO = Path(__file__).resolve().parents[2]
XML_PATH = REPO / "models" / "MJCF" / "vbot" / "vbot_mpc_scene.xml"

FLOATING_JOINT = "joint_fixed_world"

DDS_JOINT_ORDER = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)


def joint_qpos_addr(model, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"joint not found: {joint_name}")
    return int(model.jnt_qposadr[jid])


def base_qpos_addr(model) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FLOATING_JOINT)
    if jid < 0:
        raise RuntimeError(f"base joint not found: {FLOATING_JOINT}")
    return int(model.jnt_qposadr[jid])


def read_pose(model, data):
    return {
        name: float(data.qpos[joint_qpos_addr(model, name)])
        for name in DDS_JOINT_ORDER
    }


def print_pose_yaml(name, pose):
    print(f"{name}:")
    for joint_name in DDS_JOINT_ORDER:
        print(f"  {joint_name}: {pose[joint_name]:+.6f}")


def set_initial_pose(model, data, base_height: float):
    """
    给机器人一个初始姿态，然后让它自由下落。
    这里不是目标 down pose，只是初始条件。
    """
    qadr = base_qpos_addr(model)

    # floating base: x, y, z, qw, qx, qy, qz
    data.qpos[qadr:qadr + 7] = [
        0.0, 0.0, base_height,
        1.0, 0.0, 0.0, 0.0,
    ]

    # 初始关节姿态：类似站姿，但不要太僵硬
    init_pose = {
        "FR_hip_joint": 0.0,
        "FR_thigh_joint": 0.9,
        "FR_calf_joint": -1.8,
        "FL_hip_joint": 0.0,
        "FL_thigh_joint": 0.9,
        "FL_calf_joint": -1.8,
        "RR_hip_joint": 0.0,
        "RR_thigh_joint": 0.9,
        "RR_calf_joint": -1.8,
        "RL_hip_joint": 0.0,
        "RL_thigh_joint": 0.9,
        "RL_calf_joint": -1.8,
    }

    for joint_name, value in init_pose.items():
        data.qpos[joint_qpos_addr(model, joint_name)] = value

    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def zero_actuators(data):
    """
    不给控制力矩。
    如果模型有 actuator，这里全部置 0。
    """
    if data.ctrl is not None and len(data.ctrl) > 0:
        data.ctrl[:] = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-height", type=float, default=0.45)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--print-every", type=float, default=0.5)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    set_initial_pose(model, data, args.base_height)

    print("Initial pose:")
    print_pose_yaml("initial", read_pose(model, data))

    sim_time = 0.0
    next_print = 0.0

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and sim_time < args.duration:
                zero_actuators(data)
                mujoco.mj_step(model, data)

                sim_time = float(data.time)

                if sim_time >= next_print:
                    print(f"\ntime = {sim_time:.3f}, base_z = {data.qpos[base_qpos_addr(model) + 2]:.4f}")
                    next_print += args.print_every

                viewer.sync()
                time.sleep(model.opt.timestep)
    else:
        while sim_time < args.duration:
            zero_actuators(data)
            mujoco.mj_step(model, data)
            sim_time = float(data.time)

    final_pose = read_pose(model, data)

    print("\nFinal free-fall / collapsed pose:")
    print_pose_yaml("down_freefall", final_pose)

    print("\nBase height:")
    print(f"  base_z: {data.qpos[base_qpos_addr(model) + 2]:+.6f}")


if __name__ == "__main__":
    main()
