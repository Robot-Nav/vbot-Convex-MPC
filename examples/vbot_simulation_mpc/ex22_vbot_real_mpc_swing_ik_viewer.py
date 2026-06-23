"""
VBot validation 22: visualize EX34-style swing IK targets in MuJoCo.

This is a geometry/debug helper. It does not run MPC, integrate dynamics, or
publish DDS commands. It reproduces the EX34 target path:

    gait mask -> swing foot trajectory -> body-frame IK -> q_stand +/- limit

Use it to inspect whether the swing-leg joint targets look reasonable in the
same MuJoCo model used by the simulation examples.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import mujoco.viewer
import numpy as np


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from convex_mpc.gait import Gait  # noqa: E402
from convex_mpc.leg_controller import LegController  # noqa: E402
from convex_mpc.mujoco_vbot_model import MuJoCo_VBot_Model  # noqa: E402
from convex_mpc.vbot_robot_data import JOINTS, PIN_LEG_ORDER, PinVBotModel  # noqa: E402


DEFAULT_POSE_FILE = REPO / "configs" / "vbot_model_poses.yaml"
DEFAULT_XML = REPO / "models" / "MJCF" / "vbot" / "vbot_mpc_scene.xml"
LEG_INDEX = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
LEG_ORDER = ("FL", "FR", "RL", "RR")
JOINT_ORDER = tuple(joint for leg in LEG_ORDER for joint in JOINTS[leg])


@dataclass
class Frame:
    time_s: float
    mask: str
    pose_kind: str
    q_pin: np.ndarray
    max_delta_joint: str
    max_delta: float


class WarmupGait:
    def __init__(self, gait: Gait, warmup_s: float):
        self.gait = gait
        self.warmup_s = max(0.0, float(warmup_s))

    def __getattr__(self, name):
        return getattr(self.gait, name)

    def _active_time(self, time_s: float) -> float:
        return max(0.0, float(time_s) - self.warmup_s)

    def compute_current_mask(self, time_s):
        if float(time_s) < self.warmup_s:
            return np.ones(4, dtype=np.int32)
        return self.gait.compute_current_mask(self._active_time(time_s))

    def compute_contact_table(self, t0, dt, n):
        table = np.ones((4, int(n)), dtype=np.int32)
        for k in range(int(n)):
            t = float(t0) + float(dt) * (k + 0.5)
            if t >= self.warmup_s:
                table[:, k] = self.gait.compute_current_mask(self._active_time(t)).reshape(4)
        return table

    def compute_swing_traj_and_touchdown(self, vbot, leg: str):
        return self.gait.compute_swing_traj_and_touchdown(vbot, leg)


class StepTrotGait:
    def __init__(self, gait):
        self.gait = gait
        self.swing_time = float(gait.swing_time)
        self.stance_time = float(gait.stance_time)
        self.overlap_time = max(self.stance_time - 0.5 * float(gait.gait_period), 0.0)
        self.step_period = self.swing_time + self.overlap_time
        self.gait_period = 2.0 * self.step_period
        self.gait_duty = float(gait.gait_duty)
        self.gait_hz = 1.0 / max(self.gait_period, 1.0e-9)

    def __getattr__(self, name):
        return getattr(self.gait, name)

    def compute_current_mask(self, time_s):
        t = float(time_s) % max(self.gait_period, 1.0e-9)
        if t < self.swing_time:
            return np.array([0, 1, 1, 0], dtype=np.int32)
        if t < self.step_period:
            return np.ones(4, dtype=np.int32)
        if t < self.step_period + self.swing_time:
            return np.array([1, 0, 0, 1], dtype=np.int32)
        return np.ones(4, dtype=np.int32)

    def compute_contact_table(self, t0, dt, n):
        table = np.ones((4, int(n)), dtype=np.int32)
        for k in range(int(n)):
            t = float(t0) + float(dt) * (k + 0.5)
            table[:, k] = self.compute_current_mask(t).reshape(4)
        return table

    def compute_swing_traj_and_touchdown(self, vbot, leg: str):
        return self.gait.compute_swing_traj_and_touchdown(vbot, leg)


class PreserveTakeoffZGait:
    def __init__(self, gait):
        self.gait = gait

    def __getattr__(self, name):
        return getattr(self.gait, name)

    def compute_swing_traj_and_touchdown(self, vbot, leg: str):
        traj, touchdown = self.gait.compute_swing_traj_and_touchdown(vbot, leg)
        try:
            foot_pos, _foot_vel = vbot.get_single_foot_state_in_world(leg)
            foot_pos = np.asarray(foot_pos, dtype=float).reshape(3)
            touchdown = np.asarray(touchdown, dtype=float).reshape(3).copy()
            touchdown[2] = float(foot_pos[2])
            return self.gait.make_swing_trajectory(foot_pos, touchdown, self.swing_time, self.swing_height), touchdown
        except Exception:
            return traj, touchdown


def load_pose(path: Path, pose_name: str) -> dict[str, float]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load vbot_model_poses.yaml") from exc
    with path.open("r", encoding="utf-8") as f:
        poses = yaml.safe_load(f)
    if pose_name not in poses:
        raise RuntimeError(f"pose {pose_name!r} not found in {path}")
    pose = {joint: float(value) for joint, value in poses[pose_name].items()}
    missing = sorted(set(JOINT_ORDER) - set(pose))
    if missing:
        raise RuntimeError(f"pose {pose_name!r} is missing joints: {', '.join(missing)}")
    return pose


def pose_dict_to_pin_q(pose: dict[str, float], base_height: float) -> np.ndarray:
    joint_q = []
    for leg in PIN_LEG_ORDER:
        joint_q.extend(float(pose[joint]) for joint in JOINTS[leg])
    return np.concatenate(
        [
            np.array([0.0, 0.0, float(base_height)], dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            np.asarray(joint_q, dtype=float),
        ]
    )


def pin_q_to_pose_dict(q_pin: np.ndarray) -> dict[str, float]:
    q_pin = np.asarray(q_pin, dtype=float).reshape(-1)
    pose = {}
    offset = 7
    for leg in PIN_LEG_ORDER:
        for i, joint in enumerate(JOINTS[leg]):
            pose[joint] = float(q_pin[offset + i])
        offset += 3
    return pose


def pose_dict_to_mujoco_qpos(mujoco_vbot: MuJoCo_VBot_Model, pose: dict[str, float], base_height: float):
    q_pin = pose_dict_to_pin_q(pose, base_height)
    data = mj.MjData(mujoco_vbot.model)
    data.qpos[:] = mujoco_vbot.data.qpos
    mujoco_vbot.data = data
    mujoco_vbot.update_with_q_pin(q_pin)
    return mujoco_vbot.data.qpos.copy(), q_pin.copy()


def solve_leg_ik_body_near(vbot: PinVBotModel, leg: str, foot_des_body, q_seed, max_iters=12):
    q = np.asarray(q_seed, dtype=float).reshape(3).copy()
    target = np.asarray(foot_des_body, dtype=float).reshape(3)
    damping = 1.0e-4
    max_step = 0.08
    for _ in range(int(max_iters)):
        pos = np.asarray(vbot.calc_leg_fk_body(leg, q), dtype=float).reshape(3)
        err = target - pos
        if float(np.linalg.norm(err)) < 1.0e-5:
            break
        jac = np.asarray(vbot.calc_leg_jacobian_body(leg, q), dtype=float).reshape(3, 3)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), err)
        step_norm = float(np.linalg.norm(dq))
        if step_norm > max_step:
            dq *= max_step / step_norm
        q += dq
    return q


def swing_target_from_leg_output(vbot: PinVBotModel, leg: str, out, q_seed):
    foot_des_world = np.asarray(out.pos_des, dtype=float).reshape(3)
    foot_vel_des_world = np.asarray(out.vel_des, dtype=float).reshape(3)
    foot_des_body = vbot.R_world_to_body @ (foot_des_world - vbot.current_config.base_pos)
    foot_vel_des_body = vbot.R_world_to_body @ (foot_vel_des_world - vbot.current_config.base_vel)
    q_des = solve_leg_ik_body_near(vbot, leg, foot_des_body, q_seed)
    qd_des = vbot.calc_leg_qd_body(leg, q_des, foot_vel_des_body)
    return q_des, qd_des, foot_des_world, foot_des_body


def compute_q_target(vbot, leg_controller, gait, q_stand, t, swing_limit):
    q_target = dict(q_stand)
    q_raw = {}
    debug_rows = []
    mask = np.asarray(gait.compute_current_mask(t), dtype=int).reshape(4)
    zero_force = np.zeros(3, dtype=float)

    for leg in LEG_ORDER:
        out = leg_controller.compute_leg_torque(leg, vbot, gait, zero_force, t)
        if int(mask[LEG_INDEX[leg]]) != 0:
            continue

        q_seed = [q_stand[joint] for joint in JOINTS[leg]]
        q_leg, qd_leg, foot_des_world, foot_des_body = swing_target_from_leg_output(vbot, leg, out, q_seed)
        for joint, q_value, qd_value in zip(JOINTS[leg], q_leg, qd_leg):
            raw = float(q_value)
            clipped = raw
            if swing_limit > 0.0:
                clipped = float(np.clip(raw, q_stand[joint] - swing_limit, q_stand[joint] + swing_limit))
            q_raw[joint] = raw
            q_target[joint] = clipped
            debug_rows.append(
                {
                    "time_s": t,
                    "mask": "".join(str(int(v)) for v in mask),
                    "leg": leg,
                    "joint": joint,
                    "q_stand": q_stand[joint],
                    "q_ik_raw": raw,
                    "q_target": clipped,
                    "qd_ik": float(qd_value),
                    "foot_des_world_x": float(foot_des_world[0]),
                    "foot_des_world_y": float(foot_des_world[1]),
                    "foot_des_world_z": float(foot_des_world[2]),
                    "foot_des_body_x": float(foot_des_body[0]),
                    "foot_des_body_y": float(foot_des_body[1]),
                    "foot_des_body_z": float(foot_des_body[2]),
                }
            )
    return q_target, q_raw, mask, debug_rows


def max_pose_delta(q_target: dict[str, float], q_stand: dict[str, float]):
    joint = max(JOINT_ORDER, key=lambda name: abs(q_target[name] - q_stand[name]))
    return joint, float(q_target[joint] - q_stand[joint])


def build_frames(args):
    q_stand = load_pose(Path(args.pose_file), args.pose)
    vbot = PinVBotModel()
    mujoco_vbot = MuJoCo_VBot_Model(xml_path=args.xml)

    q_stand_pin = pose_dict_to_pin_q(q_stand, args.base_height)
    dq_zero = np.zeros(18, dtype=float)
    vbot.update_model(q_stand_pin, dq_zero)
    mujoco_vbot.update_with_q_pin(q_stand_pin)

    gait = StepTrotGait(Gait(args.gait_frequency_hz, args.gait_duty, swing_height=args.swing_height))
    if args.swing_preserve_takeoff_z:
        gait = PreserveTakeoffZGait(gait)
    gait = WarmupGait(gait, args.gait_warmup_s)
    leg_controller = LegController()

    frames = []
    debug_rows = []
    last_mask = None
    sample_times = np.arange(0.0, args.duration_s + 0.5 * args.dt_s, args.dt_s)

    for t in sample_times:
        vbot.update_model(q_stand_pin, dq_zero)
        q_target, _q_raw, mask, rows = compute_q_target(
            vbot,
            leg_controller,
            gait,
            q_stand,
            float(t),
            args.swing_q_target_delta_limit,
        )
        mask_text = "".join(str(int(v)) for v in mask)
        debug_rows.extend(rows)

        include = args.keep_all_frames or mask_text != last_mask or rows
        if include:
            target_pin = pose_dict_to_pin_q(q_target, args.base_height)
            joint, delta = max_pose_delta(q_target, q_stand)
            frames.append(Frame(float(t), mask_text, "target", target_pin, joint, delta))
            if args.alternate_stand:
                frames.append(Frame(float(t), mask_text, "stand", q_stand_pin.copy(), joint, 0.0))
        last_mask = mask_text

    return mujoco_vbot, frames, debug_rows


def write_debug_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(frames, rows):
    print("Generated EX34-style swing IK target frames:")
    counts = {}
    for frame in frames:
        counts[(frame.mask, frame.pose_kind)] = counts.get((frame.mask, frame.pose_kind), 0) + 1
    if counts:
        print("Frame counts:")
        for (mask, pose_kind), count in sorted(counts.items()):
            print(f"  mask={mask} pose={pose_kind:6s} count={count}")
    for frame in frames:
        print(
            f"  t={frame.time_s:6.3f}s mask={frame.mask} pose={frame.pose_kind:6s} "
            f"max_delta={frame.max_delta_joint}:{frame.max_delta:+.4f}"
        )
    if rows:
        print("\nSwing IK rows:")
        for row in rows:
            print(
                f"  t={row['time_s']:6.3f}s mask={row['mask']} {row['joint']:14s} "
                f"raw={row['q_ik_raw']:+.4f} target={row['q_target']:+.4f} "
                f"stand={row['q_stand']:+.4f} "
                f"foot_body=({row['foot_des_body_x']:+.3f},"
                f"{row['foot_des_body_y']:+.3f},{row['foot_des_body_z']:+.3f})"
            )


def replay_frames(mujoco_vbot: MuJoCo_VBot_Model, frames, fps: float):
    if not frames:
        print("No frames to replay.")
        return

    model = mujoco_vbot.model
    data = mj.MjData(model)
    frame_dt = 1.0 / max(float(fps), 1.0)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco_vbot.base_bid
        viewer.cam.distance = 2.4
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 135
        viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        k = 0
        while viewer.is_running():
            frame = frames[k % len(frames)]
            temp = MuJoCo_VBot_Model(xml_path=mujoco_vbot.xml_path)
            temp.data = data
            temp.update_with_q_pin(frame.q_pin)
            data.qvel[:] = 0.0
            data.ctrl[:] = 0.0
            mj.mj_forward(model, data)
            print(
                f"\rview t={frame.time_s:6.3f}s mask={frame.mask} "
                f"pose={frame.pose_kind:6s} max_delta={frame.max_delta_joint}:{frame.max_delta:+.4f}",
                end="",
                flush=True,
            )
            viewer.sync()
            time.sleep(frame_dt)
            k += 1
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--pose-file", default=str(DEFAULT_POSE_FILE))
    parser.add_argument("--pose", default="stand")
    parser.add_argument("--base-height", type=float, default=0.462)
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--dt-s", type=float, default=0.05)
    parser.add_argument("--gait-frequency-hz", type=float, default=1.0)
    parser.add_argument("--gait-duty", type=float, default=0.60)
    parser.add_argument("--gait-warmup-s", type=float, default=2.0)
    parser.add_argument("--swing-height", type=float, default=0.025)
    parser.add_argument("--swing-preserve-takeoff-z", action="store_true", default=True)
    parser.add_argument("--swing-touchdown-ground-z", action="store_false", dest="swing_preserve_takeoff_z")
    parser.add_argument("--swing-q-target-delta-limit", type=float, default=0.18)
    parser.add_argument("--keep-all-frames", action="store_true")
    parser.add_argument("--alternate-stand", action="store_true", help="alternate target frames with the stand pose")
    parser.add_argument("--csv", default="")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--fps", type=float, default=4.0)
    args = parser.parse_args()

    mujoco_vbot, frames, debug_rows = build_frames(args)
    print_summary(frames, debug_rows)

    if args.csv:
        write_debug_csv(Path(args.csv), debug_rows)
        print(f"\ncsv_saved: {args.csv}")

    if args.viewer:
        print("\nOpening MuJoCo viewer. Close the window to exit.")
        replay_frames(mujoco_vbot, frames, args.fps)
    else:
        print("\nRun again with --viewer to inspect these target poses in MuJoCo.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
