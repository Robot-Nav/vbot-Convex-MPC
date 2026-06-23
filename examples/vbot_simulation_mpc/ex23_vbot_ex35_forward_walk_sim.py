#!/usr/bin/env python3
"""EX23: MuJoCo simulation mirror of EX35 real forward-walk logic.

This script keeps the same control shape used by EX35/EX34 on hardware, and
also provides a direct-torque mode for checking the migrated real-control
policy against the original MuJoCo MPC controller:

    StepTrotGait + WarmupGait
    ComTraj + CentroidalMPC
    stance MPC force -> tau_ff scale
    swing foot trajectory -> body-frame IK -> q/dq targets
    joint PD + tau_ff mixed command

The mixed mode is useful for sim-to-real bring-up. The direct-torque mode keeps
the same gait and MPC solve, but sends the full, unscaled LegController torque
to MuJoCo; this is closer to ex16/ex17 and is the quickest way to tell whether
poor walking comes from MPC itself or from the mixed low-level policy.

It does not use DDS. MuJoCo receives the mixed joint torque directly.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import mujoco as mj
import numpy as np


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from convex_mpc import centroidal_mpc as centroidal_mpc_module  # noqa: E402
from convex_mpc.centroidal_mpc import CentroidalMPC  # noqa: E402
from convex_mpc.com_trajectory import ComTraj  # noqa: E402
from convex_mpc.gait import Gait  # noqa: E402
from convex_mpc.leg_controller import LegController  # noqa: E402
from convex_mpc.mujoco_vbot_model import MuJoCo_VBot_Model  # noqa: E402
from convex_mpc.vbot_robot_data import JOINTS, MPC_LEG_ORDER, PIN_LEG_ORDER, PinVBotModel  # noqa: E402


DEFAULT_CONFIG = REPO / "configs" / "ex34_forward_walk_slow_imu.yaml"
DEFAULT_POSE_FILE = REPO / "configs" / "vbot_model_poses.yaml"
DEFAULT_XML = REPO / "models" / "MJCF" / "vbot" / "vbot_mpc_scene.xml"
SIM_NOMINAL_BODY_HEIGHT = 0.2774
SIM_TUNED_DEFAULTS = {
    # The real EX35 YAML keeps stance q_target locked to stand and uses tiny
    # tau_ff for early hardware safety. In MuJoCo this causes a crouched,
    # rear-biased gait, so EX23 defaults to a moderate stance correction and
    # stronger MPC injection. These are still below the aggressive sim-only
    # settings and are intended as a bridge toward cautious real-robot tests.
    "x_vel": 0.06,
    "mpc_hz": 20.0,
    "forward_touchdown": True,
    "forward_touchdown_scale": 3.0,
    "stance_mpc_q_delta_limit": 0.10,
    "stance_mpc_q_delta_scale": 0.60,
    "tau_ff_scale": 0.18,
    "swing_tau_ff_scale": 0.25,
    "mpc_force_xy_scale": 0.25,
    "swing_qd_target_limit": -1.0,
}

SIM_HZ = 1000.0
CTRL_HZ = 200.0
SIM_DT = 1.0 / SIM_HZ
CTRL_DECIM = int(SIM_HZ // CTRL_HZ)
VIEWER_HZ = 60.0
LEG_INDEX = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}
JOINT_ORDER = tuple(joint for leg in MPC_LEG_ORDER for joint in JOINTS[leg])
TAU_LIMIT = np.array([17.0, 17.0, 34.0] * 4, dtype=float)
JOINT_SOFT_LIMITS = {
    "hip": (-0.73304, 0.73304),
    "F_thigh": (-1.559, 3.1298),
    "R_thigh": (-0.51181, 4.177),
    "calf": (-2.6387, -0.7854),
}


class WarmupGait:
    def __init__(self, gait, warmup_s: float):
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


class ForwardTouchdownGait:
    """Override swing touchdown x placement for slow forward-walk debugging."""

    def __init__(self, gait, step_x_scale: float):
        self.gait = gait
        self.step_x_scale = float(step_x_scale)

    def __getattr__(self, name):
        return getattr(self.gait, name)

    def compute_current_mask(self, time_s):
        return self.gait.compute_current_mask(time_s)

    def compute_contact_table(self, t0, dt, n):
        return self.gait.compute_contact_table(t0, dt, n)

    def compute_touchdown_world_for_traj_purpose_only(self, vbot, leg: str, terrain_height_fn=None):
        td = np.asarray(
            self.gait.compute_touchdown_world_for_traj_purpose_only(vbot, leg, terrain_height_fn),
            dtype=float,
        ).reshape(3)
        return td

    def compute_swing_traj_and_touchdown(self, vbot, leg: str):
        foot_pos, _foot_vel = vbot.get_single_foot_state_in_world(leg)
        foot_pos = np.asarray(foot_pos, dtype=float).reshape(3)
        base_pos = np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)
        hip_offset = np.asarray(vbot.get_hip_offset(leg), dtype=float).reshape(3)
        hip_world = np.array([base_pos[0], base_pos[1], 0.0]) + vbot.R_z @ hip_offset

        step_x = float(vbot.x_vel_des_world) * float(self.swing_time) * self.step_x_scale
        step_y = float(vbot.y_vel_des_world) * float(self.swing_time) * self.step_x_scale
        touchdown = np.array([hip_world[0] + step_x, hip_world[1] + step_y, 0.02], dtype=float)
        touchdown[2] = 0.02
        return self.gait.make_swing_trajectory(
            foot_pos,
            touchdown,
            self.swing_time,
            self.swing_height,
        ), touchdown


@dataclass
class SimCommand:
    x_vel: float
    y_vel: float
    yaw_rate: float
    z_pos: float
    pitch: float


@dataclass
class MixedJointCommand:
    q_target: dict[str, float]
    qd_target: dict[str, float]
    kp: np.ndarray
    kd: np.ndarray
    tau_bias: np.ndarray
    tau_mpc: np.ndarray
    contact_mask: np.ndarray

    @property
    def tau_ff(self) -> np.ndarray:
        return self.tau_bias + self.tau_mpc


def load_yaml_defaults(path: Path) -> dict:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required for --config") from exc
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    data = raw.get("args", raw)
    return {str(k).replace("-", "_"): v for k, v in data.items()}


def load_pose(path: Path, pose_name: str) -> dict[str, float]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load poses") from exc
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        poses = yaml.safe_load(f) or {}
    pose = poses.get(pose_name)
    if not isinstance(pose, dict):
        raise RuntimeError(f"pose {pose_name!r} not found in {path}")
    return {joint: float(pose[joint]) for joint in JOINT_ORDER}


def pose_dict_to_pin_q(pose: dict[str, float], base_height: float):
    joint_q = []
    for leg in PIN_LEG_ORDER:
        joint_q.extend(pose[joint] for joint in JOINTS[leg])
    return np.concatenate(
        [
            np.array([0.0, 0.0, float(base_height)], dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            np.asarray(joint_q, dtype=float),
        ]
    )


def get_mujoco_joint_vectors(mujoco_vbot: MuJoCo_VBot_Model):
    q = {}
    dq = {}
    for leg in MPC_LEG_ORDER:
        for joint in JOINTS[leg]:
            jid = mujoco_vbot.joint_ids[joint]
            q[joint] = float(mujoco_vbot.data.qpos[int(mujoco_vbot.model.jnt_qposadr[jid])])
            dq[joint] = float(mujoco_vbot.data.qvel[int(mujoco_vbot.model.jnt_dofadr[jid])])
    return q, dq


def dict_to_vec(values: dict[str, float]) -> np.ndarray:
    return np.array([float(values[joint]) for joint in JOINT_ORDER], dtype=float)


def zero_joint_dict() -> dict[str, float]:
    return {joint: 0.0 for joint in JOINT_ORDER}


def compute_joint_bias_vector(vbot: PinVBotModel, contact_mask=None) -> np.ndarray:
    tau_bias = np.zeros(12, dtype=float)
    g, C, _ = vbot.compute_dynamcis_terms()
    joint_bias = np.asarray(C @ vbot.current_config.get_dq() + g, dtype=float).reshape(-1)
    if contact_mask is None:
        contact_mask = np.ones(4, dtype=int)
    contact_mask = np.asarray(contact_mask, dtype=int).reshape(4)
    for leg, leg_slice in LEG_SLICE.items():
        if int(contact_mask[LEG_INDEX[leg]]) == 1:
            tau_bias[leg_slice] = joint_bias[vbot.get_leg_joint_vcols(leg)]
    return tau_bias


def gain_vectors_for_contact(args, contact_mask) -> tuple[np.ndarray, np.ndarray]:
    contact_mask = np.asarray(contact_mask, dtype=int).reshape(4)
    kp = np.zeros(12, dtype=float)
    kd = np.zeros(12, dtype=float)
    kp_scale_by_type = {
        "hip": float(args.stance_hip_kp_scale),
        "thigh": float(args.stance_thigh_kp_scale),
        "calf": float(args.stance_calf_kp_scale),
    }
    kd_scale_by_type = {
        "hip": float(args.stance_hip_kd_scale),
        "thigh": float(args.stance_thigh_kd_scale),
        "calf": float(args.stance_calf_kd_scale),
    }
    for i, joint in enumerate(JOINT_ORDER):
        leg = joint.split("_", 1)[0]
        joint_type = joint.split("_", 2)[1]
        in_swing = int(contact_mask[LEG_INDEX[leg]]) == 0
        if in_swing:
            kp[i] = float(args.swing_kp)
            kd[i] = float(args.swing_kd)
        else:
            kp[i] = float(args.kp) * kp_scale_by_type[joint_type]
            kd[i] = float(args.kd) * kd_scale_by_type[joint_type]
    return kp, kd


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


def joint_kind(leg: str, joint: str) -> str:
    if "_hip_" in joint:
        return "hip"
    if "_calf_" in joint:
        return "calf"
    if "_thigh_" in joint:
        return "F_thigh" if leg in ("FL", "FR") else "R_thigh"
    raise ValueError(f"unknown joint kind: {joint}")


def joint_step_limit(joint: str, args) -> float:
    if "_hip_" in joint:
        return float(args.swing_hip_q_step_limit)
    if "_thigh_" in joint:
        return float(args.swing_thigh_q_step_limit)
    if "_calf_" in joint:
        return float(args.swing_calf_q_step_limit)
    return float(args.swing_q_step_limit)


def apply_joint_soft_limits(q_target: dict[str, float], args) -> dict[str, float]:
    margin = max(0.0, float(args.joint_soft_limit_margin))
    out = dict(q_target)
    for leg in MPC_LEG_ORDER:
        for joint in JOINTS[leg]:
            lo, hi = JOINT_SOFT_LIMITS[joint_kind(leg, joint)]
            out[joint] = float(np.clip(out[joint], lo + margin, hi - margin))
    return out


def rate_limit_q_target(q_raw: dict[str, float], q_prev: dict[str, float], args) -> dict[str, float]:
    out = {}
    for joint in JOINT_ORDER:
        step = joint_step_limit(joint, args)
        if step < 0.0:
            out[joint] = float(q_raw[joint])
            continue
        delta = float(q_raw[joint]) - float(q_prev[joint])
        out[joint] = float(q_prev[joint]) + float(np.clip(delta, -step, step))
    return out


def clamp_swing_foot_body(vbot: PinVBotModel, leg: str, foot_des_body, args):
    foot = np.asarray(foot_des_body, dtype=float).reshape(3).copy()
    hip = np.asarray(vbot.get_hip_offset(leg), dtype=float).reshape(3)
    rel = foot - hip
    rel[0] = np.clip(rel[0], float(args.swing_foot_x_min), float(args.swing_foot_x_max))
    rel[1] = np.clip(rel[1], float(args.swing_foot_y_min), float(args.swing_foot_y_max))
    rel[2] = np.clip(rel[2], float(args.swing_foot_z_min), float(args.swing_foot_z_max))
    return hip + rel


def swing_target_from_leg_output(vbot: PinVBotModel, leg: str, out, q_model, args):
    foot_des_world = np.asarray(out.pos_des, dtype=float).reshape(3)
    foot_vel_des_world = np.asarray(out.vel_des, dtype=float).reshape(3)
    foot_des_body = vbot.R_world_to_body @ (foot_des_world - vbot.current_config.base_pos)
    foot_des_body = clamp_swing_foot_body(vbot, leg, foot_des_body, args)
    foot_vel_des_body = vbot.R_world_to_body @ (foot_vel_des_world - vbot.current_config.base_vel)
    q_seed = [q_model[joint] for joint in JOINTS[leg]]
    q_des = solve_leg_ik_body_near(vbot, leg, foot_des_body, q_seed)
    qd_des = vbot.calc_leg_qd_body(leg, q_des, foot_vel_des_body)
    return q_des, qd_des


def filter_mpc_force_now(args, mpc_force_now):
    force = np.asarray(mpc_force_now, dtype=float).copy()
    xy_scale = float(args.mpc_force_xy_scale)
    for leg_slice in LEG_SLICE.values():
        force[leg_slice.start : leg_slice.start + 2] *= xy_scale
    return force


def apply_stance_mpc_q_delta(args, q_target, q_stand, contact_mask, tau_model, kp_stance):
    if float(args.stance_mpc_q_delta_limit) <= 0.0 or float(kp_stance) <= 1.0e-9:
        return q_target

    contact_mask = np.asarray(contact_mask, dtype=int).reshape(4)
    for leg, leg_slice in LEG_SLICE.items():
        if int(contact_mask[LEG_INDEX[leg]]) != 1:
            continue
        tau_leg = np.asarray(tau_model[leg_slice], dtype=float).reshape(3)
        q_delta = float(args.stance_mpc_q_delta_scale) * tau_leg / float(kp_stance)
        q_delta = np.clip(
            q_delta,
            -float(args.stance_mpc_q_delta_limit),
            float(args.stance_mpc_q_delta_limit),
        )
        for joint, delta in zip(JOINTS[leg], q_delta):
            q_target[joint] = float(q_stand[joint] + delta)
    return q_target


def compute_tau_and_targets(vbot, leg_controller, gait, args, q_stand, q_model, mpc_force_now, time_s):
    contact_mask = np.asarray(gait.compute_current_mask(time_s), dtype=int).reshape(4)
    tau_model = np.zeros(12, dtype=float)
    leg_outputs = {}
    q_target = dict(q_stand)
    qd_target = zero_joint_dict()

    # During WarmupGait all legs are stance. For zero-MPC diagnostics this must
    # collapse exactly to the stand-only command path: stand q, zero qd, no MPC tau.
    if bool(np.all(contact_mask == 1)) and (args.no_mpc_tau or float(args.tau_ff_scale) == 0.0):
        return tau_model, np.zeros(12, dtype=float), q_target, qd_target, contact_mask, leg_outputs

    for leg, leg_slice in LEG_SLICE.items():
        force_leg = np.asarray(mpc_force_now[leg_slice], dtype=float).reshape(3)
        out = leg_controller.compute_leg_torque(leg, vbot, gait, force_leg, time_s)
        leg_outputs[leg] = out
        tau_model[leg_slice] = np.asarray(out.tau, dtype=float).reshape(3)

    q_target = apply_stance_mpc_q_delta(
        args,
        q_target,
        q_stand,
        contact_mask,
        tau_model,
        args.kp,
    )

    for leg, out in leg_outputs.items():
        if int(contact_mask[LEG_INDEX[leg]]) != 0:
            continue
        q_leg, qd_leg = swing_target_from_leg_output(vbot, leg, out, q_model, args)
        for joint, q_value, qd_value in zip(JOINTS[leg], q_leg, qd_leg):
            if args.swing_q_target_delta_limit > 0.0:
                q_value = np.clip(
                    float(q_value),
                    q_stand[joint] - args.swing_q_target_delta_limit,
                    q_stand[joint] + args.swing_q_target_delta_limit,
                )
            if args.swing_qd_target_limit >= 0.0:
                qd_value = np.clip(
                    float(qd_value),
                    -float(args.swing_qd_target_limit),
                    float(args.swing_qd_target_limit),
                )
            q_target[joint] = float(q_value)
            qd_target[joint] = float(qd_value)

    q_target = apply_joint_soft_limits(q_target, args)

    tau_ff_scale = np.zeros(12, dtype=float)
    for i, joint in enumerate(JOINT_ORDER):
        leg = joint.split("_", 1)[0]
        in_swing = int(contact_mask[LEG_INDEX[leg]]) == 0
        tau_ff_scale[i] = float(args.swing_tau_ff_scale) if in_swing else float(args.tau_ff_scale)

    return tau_model, tau_ff_scale, q_target, qd_target, contact_mask, leg_outputs


def compute_direct_leg_torque(vbot, leg_controller, gait, mpc_force_now, time_s):
    contact_mask = np.asarray(gait.compute_current_mask(time_s), dtype=int).reshape(4)
    tau_model = np.zeros(12, dtype=float)
    leg_outputs = {}
    for leg, leg_slice in LEG_SLICE.items():
        force_leg = np.asarray(mpc_force_now[leg_slice], dtype=float).reshape(3)
        out = leg_controller.compute_leg_torque(leg, vbot, gait, force_leg, time_s)
        leg_outputs[leg] = out
        tau_model[leg_slice] = np.asarray(out.tau, dtype=float).reshape(3)
    return tau_model, contact_mask, leg_outputs


def build_mixed_joint_command(args, vbot, q_target, qd_target, contact_mask, tau_model, tau_scale):
    contact_mask = np.asarray(contact_mask, dtype=int).reshape(4)
    kp, kd = gain_vectors_for_contact(args, contact_mask)
    tau_bias = np.zeros(12, dtype=float)
    if args.stance_bias_comp:
        tau_bias = compute_joint_bias_vector(vbot, contact_mask)
    tau_mpc = np.zeros(12, dtype=float) if args.no_mpc_tau else np.asarray(tau_scale, dtype=float) * np.asarray(tau_model, dtype=float)
    return MixedJointCommand(
        q_target=dict(q_target),
        qd_target=dict(qd_target),
        kp=kp,
        kd=kd,
        tau_bias=tau_bias,
        tau_mpc=tau_mpc,
        contact_mask=contact_mask,
    )


def build_stand_mixed_joint_command(args, vbot, q_stand):
    contact_mask = np.ones(4, dtype=int)
    return build_mixed_joint_command(
        args,
        vbot,
        q_stand,
        zero_joint_dict(),
        contact_mask,
        np.zeros(12, dtype=float),
        np.zeros(12, dtype=float),
    )


def compute_equivalent_motor_torque(cmd: MixedJointCommand, q_model, dq_model):
    q_vec = dict_to_vec(q_model)
    dq_vec = dict_to_vec(dq_model)
    q_target_vec = dict_to_vec(cmd.q_target)
    qd_target_vec = dict_to_vec(cmd.qd_target)
    tau_pd = cmd.kp * (q_target_vec - q_vec) + cmd.kd * (qd_target_vec - dq_vec)
    tau_total = np.clip(tau_pd + cmd.tau_ff, -TAU_LIMIT, TAU_LIMIT)
    return tau_total, tau_pd


def build_direct_debug_command(q_model, dq_model, tau_cmd, contact_mask):
    return MixedJointCommand(
        q_target=dict(q_model),
        qd_target=dict(dq_model),
        kp=np.zeros(12, dtype=float),
        kd=np.zeros(12, dtype=float),
        tau_bias=np.zeros(12, dtype=float),
        tau_mpc=np.asarray(tau_cmd, dtype=float).reshape(12),
        contact_mask=np.asarray(contact_mask, dtype=int).reshape(4),
    )


def build_gait(args):
    gait = StepTrotGait(Gait(args.gait_frequency_hz, args.gait_duty, swing_height=args.swing_height))
    if args.forward_touchdown:
        gait = ForwardTouchdownGait(gait, args.forward_touchdown_scale)
    if args.gait_warmup_s > 0.0:
        gait = WarmupGait(gait, args.gait_warmup_s)
    return gait


def build_arg_parser():
    defaults = {
        "duration": 8.0,
        "x_vel": 0.03,
        "y_vel": 0.0,
        "yaw_rate": 0.0,
        "base_pitch": 0.0,
        "base_height": SIM_NOMINAL_BODY_HEIGHT,
        "gait_warmup_s": 2.0,
        "gait_frequency_hz": 1.5,
        "gait_duty": 0.60,
        "swing_height": 0.08,
        "swing_q_target_delta_limit": 0.20,
        "swing_qd_target_limit": -1.0,
        "swing_q_step_limit": 0.02,
        "swing_hip_q_step_limit": 0.012,
        "swing_thigh_q_step_limit": 0.018,
        "swing_calf_q_step_limit": 0.025,
        "joint_soft_limit_margin": 0.03,
        "swing_foot_x_min": -0.35,
        "swing_foot_x_max": 0.35,
        "swing_foot_y_min": -0.20,
        "swing_foot_y_max": 0.20,
        "swing_foot_z_min": -0.45,
        "swing_foot_z_max": -0.03,
        "swing_kp": 10.0,
        "swing_kd": 3.0,
        "kp": 50.0,
        "kd": 3.0,
        "stance_hip_kp_scale": 1.0,
        "stance_thigh_kp_scale": 1.0,
        "stance_calf_kp_scale": 2.0,
        "stance_hip_kd_scale": 1.0,
        "stance_thigh_kd_scale": 1.0,
        "stance_calf_kd_scale": 2.0,
        "stance_mpc_q_delta_limit": 0.0,
        "stance_mpc_q_delta_scale": 1.0,
        "tau_ff_scale": 0.05,
        "swing_tau_ff_scale": 0.0,
        "mpc_force_xy_scale": 0.15,
        "mpc_r": 3.0e-3,
        "mpc_hz": 20.0,
        "horizon_segments": 16,
        "stance_bias_comp": True,
        "control_mode": "mixed",
        "forward_touchdown": False,
        "forward_touchdown_scale": 1.0,
    }
    parser0 = argparse.ArgumentParser(add_help=False)
    parser0.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser0.add_argument("--real-mirror-defaults", action="store_true")
    known, _ = parser0.parse_known_args()
    if known.config:
        defaults.update(load_yaml_defaults(Path(known.config)))
    if not known.real_mirror_defaults:
        defaults.update(SIM_TUNED_DEFAULTS)
    # EX35 real YAML uses the real-state estimator's nominal 0.462 m height.
    # The MuJoCo VBot scene is calibrated around ~0.277 m for the same stand
    # joint angles, matching ex18/ex21 simulation examples.
    if "base_height" not in load_yaml_defaults(Path(known.config)) if known.config else True:
        defaults["base_height"] = SIM_NOMINAL_BODY_HEIGHT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(known.config))
    parser.add_argument(
        "--real-mirror-defaults",
        action="store_true",
        default=bool(known.real_mirror_defaults),
        help="use EX35 YAML defaults without EX23 simulation tuning",
    )
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--pose-file", default=str(DEFAULT_POSE_FILE))
    parser.add_argument("--pose", default="stand")
    parser.add_argument(
        "--control-mode",
        choices=("mixed", "direct-torque"),
        default=str(defaults.get("control_mode", "mixed")).replace("_", "-"),
        help="mixed mirrors EX34/EX35; direct-torque sends full LegController torque like ex16/ex17",
    )
    parser.add_argument("--duration", type=float, default=float(defaults["duration"]))
    parser.add_argument("--x-vel", type=float, default=float(defaults["x_vel"]))
    parser.add_argument("--y-vel", type=float, default=float(defaults["y_vel"]))
    parser.add_argument("--yaw-rate", type=float, default=float(defaults["yaw_rate"]))
    parser.add_argument("--base-pitch", type=float, default=float(defaults.get("base_pitch", 0.0)))
    parser.add_argument("--base-height", type=float, default=float(defaults.get("base_height", SIM_NOMINAL_BODY_HEIGHT)))
    parser.add_argument("--gait-warmup-s", type=float, default=float(defaults["gait_warmup_s"]))
    parser.add_argument("--gait-frequency-hz", type=float, default=float(defaults["gait_frequency_hz"]))
    parser.add_argument("--gait-duty", type=float, default=float(defaults["gait_duty"]))
    parser.add_argument("--swing-height", type=float, default=float(defaults["swing_height"]))
    parser.add_argument("--swing-q-target-delta-limit", type=float, default=float(defaults["swing_q_target_delta_limit"]))
    parser.add_argument("--swing-qd-target-limit", type=float, default=float(defaults["swing_qd_target_limit"]), help="clip swing qd target; 0 disables qd target, negative leaves it unclipped")
    parser.add_argument("--swing-q-step-limit", type=float, default=float(defaults["swing_q_step_limit"]), help="fallback per-control-tick q target rate limit; negative disables")
    parser.add_argument("--swing-hip-q-step-limit", type=float, default=float(defaults["swing_hip_q_step_limit"]), help="hip q target max change per control tick; negative disables")
    parser.add_argument("--swing-thigh-q-step-limit", type=float, default=float(defaults["swing_thigh_q_step_limit"]), help="thigh q target max change per control tick; negative disables")
    parser.add_argument("--swing-calf-q-step-limit", type=float, default=float(defaults["swing_calf_q_step_limit"]), help="calf q target max change per control tick; negative disables")
    parser.add_argument("--joint-soft-limit-margin", type=float, default=float(defaults["joint_soft_limit_margin"]))
    parser.add_argument("--swing-foot-x-min", type=float, default=float(defaults["swing_foot_x_min"]))
    parser.add_argument("--swing-foot-x-max", type=float, default=float(defaults["swing_foot_x_max"]))
    parser.add_argument("--swing-foot-y-min", type=float, default=float(defaults["swing_foot_y_min"]))
    parser.add_argument("--swing-foot-y-max", type=float, default=float(defaults["swing_foot_y_max"]))
    parser.add_argument("--swing-foot-z-min", type=float, default=float(defaults["swing_foot_z_min"]))
    parser.add_argument("--swing-foot-z-max", type=float, default=float(defaults["swing_foot_z_max"]))
    parser.add_argument("--swing-kp", type=float, default=float(defaults["swing_kp"]))
    parser.add_argument("--swing-kd", type=float, default=float(defaults["swing_kd"]))
    parser.add_argument("--kp", type=float, default=float(defaults["kp"]))
    parser.add_argument("--kd", type=float, default=float(defaults["kd"]))
    parser.add_argument("--stance-hip-kp-scale", type=float, default=float(defaults["stance_hip_kp_scale"]))
    parser.add_argument("--stance-thigh-kp-scale", type=float, default=float(defaults["stance_thigh_kp_scale"]))
    parser.add_argument("--stance-calf-kp-scale", type=float, default=float(defaults["stance_calf_kp_scale"]))
    parser.add_argument("--stance-hip-kd-scale", type=float, default=float(defaults["stance_hip_kd_scale"]))
    parser.add_argument("--stance-thigh-kd-scale", type=float, default=float(defaults["stance_thigh_kd_scale"]))
    parser.add_argument("--stance-calf-kd-scale", type=float, default=float(defaults["stance_calf_kd_scale"]))
    parser.add_argument("--stance-mpc-q-delta-limit", type=float, default=float(defaults["stance_mpc_q_delta_limit"]))
    parser.add_argument("--stance-mpc-q-delta-scale", type=float, default=float(defaults["stance_mpc_q_delta_scale"]))
    parser.add_argument("--tau-ff-scale", type=float, default=float(defaults["tau_ff_scale"]))
    parser.add_argument("--swing-tau-ff-scale", type=float, default=float(defaults.get("swing_tau_ff_scale", 0.0)))
    parser.add_argument("--mpc-force-xy-scale", type=float, default=float(defaults["mpc_force_xy_scale"]))
    parser.add_argument("--mpc-r", type=float, default=float(defaults["mpc_r"]))
    parser.add_argument("--mpc-hz", type=float, default=float(defaults["mpc_hz"]))
    parser.add_argument("--horizon-segments", type=int, default=int(defaults["horizon_segments"]))
    parser.add_argument("--stance-bias-comp", action="store_true", default=bool(defaults["stance_bias_comp"]))
    parser.add_argument("--no-stance-bias-comp", action="store_false", dest="stance_bias_comp")
    parser.add_argument(
        "--forward-touchdown",
        action="store_true",
        default=bool(defaults.get("forward_touchdown", False)),
        help="use simple velocity-based swing touchdown instead of Raibert corrections",
    )
    parser.add_argument("--no-forward-touchdown", action="store_false", dest="forward_touchdown")
    parser.add_argument("--forward-touchdown-scale", type=float, default=float(defaults.get("forward_touchdown_scale", 1.0)))
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true", help="sleep to keep simulation near realtime")
    parser.add_argument("--stand-only", action="store_true", help="hold stand pose with PD+bias, no gait or MPC")
    parser.add_argument("--no-mpc-tau", action="store_true", help="disable stance MPC tau_ff; keep PD+bias only")
    parser.add_argument("--csv", default="")
    parser.add_argument("--print-hz", type=float, default=5.0)
    parser.add_argument("--print-joints", action="store_true", help="print current/target joint angles at print-hz")
    return parser


def run(args):
    centroidal_mpc_module.COST_MATRIX_R = np.diag([float(args.mpc_r)] * 12)

    vbot = PinVBotModel()
    mujoco_vbot = MuJoCo_VBot_Model(xml_path=args.xml)
    mujoco_vbot.model.opt.timestep = SIM_DT

    q_stand = load_pose(Path(args.pose_file), args.pose)
    q_pin = pose_dict_to_pin_q(q_stand, args.base_height)
    mujoco_vbot.update_with_q_pin(q_pin)
    mujoco_vbot.data.qvel[:] = 0.0
    mujoco_vbot.data.ctrl[:] = 0.0
    mj.mj_forward(mujoco_vbot.model, mujoco_vbot.data)
    mujoco_vbot.update_pin_with_mujoco(vbot)

    leg_controller = LegController()
    gait = build_gait(args)
    traj = ComTraj(vbot)
    mpc_dt = gait.gait_period / int(args.horizon_segments)
    steps_per_mpc = max(1, int(round(CTRL_HZ / float(args.mpc_hz))))
    cmd = SimCommand(args.x_vel, args.y_vel, args.yaw_rate, args.base_height, args.base_pitch)

    traj.generate_traj(
        vbot,
        gait,
        0.0,
        cmd.x_vel,
        cmd.y_vel,
        cmd.z_pos,
        cmd.yaw_rate,
        mpc_dt,
        pitch_des_body=cmd.pitch,
    )
    mpc = CentroidalMPC(vbot, traj)
    u_opt = np.zeros((12, traj.N), dtype=float)
    mpc_force_now = np.zeros(12, dtype=float)
    tau_hold = np.zeros(12, dtype=float)
    tau_pd = np.zeros(12, dtype=float)
    mixed_cmd = build_stand_mixed_joint_command(args, vbot, q_stand)
    q_target_prev_model = dict(mixed_cmd.q_target)
    contact_mask = mixed_cmd.contact_mask
    prev_contact_mask = np.asarray(contact_mask, dtype=int).reshape(4).copy()
    leg_outputs = {}
    rows = []
    ctrl_i = 0
    sim_steps = int(float(args.duration) * SIM_HZ)
    next_print = 0.0
    next_viewer = 0.0
    start_wall = time.perf_counter()

    viewer_ctx = None
    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer_ctx = mujoco.viewer.launch_passive(mujoco_vbot.model, mujoco_vbot.data)
        viewer = viewer_ctx.__enter__()
        viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco_vbot.base_bid
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90
        viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    try:
        print(
            "EX23 sim: "
            f"mode={args.control_mode} duration={args.duration:.2f}s x_vel={args.x_vel:+.3f} "
            f"pitch={args.base_pitch:+.3f} "
            f"warmup={args.gait_warmup_s:.2f}s mpc_hz={args.mpc_hz:.1f} "
            f"tau_ff_scale={args.tau_ff_scale:.3f} "
            f"swing_tau_ff_scale={args.swing_tau_ff_scale:.3f} "
            f"force_xy_scale={args.mpc_force_xy_scale:.3f}"
        )
        print(
            "defaults: "
            f"{'real EX35 mirror' if args.real_mirror_defaults else 'EX23 sim tuned'} "
            f"forward_touchdown={bool(args.forward_touchdown)} "
            f"touchdown_scale={args.forward_touchdown_scale:.2f}"
        )
        print(
            "stance gain scale: "
            f"kp hip/thigh/calf="
            f"{args.stance_hip_kp_scale:.2f}/{args.stance_thigh_kp_scale:.2f}/{args.stance_calf_kp_scale:.2f} "
            f"kd hip/thigh/calf="
            f"{args.stance_hip_kd_scale:.2f}/{args.stance_thigh_kd_scale:.2f}/{args.stance_calf_kd_scale:.2f}"
        )
        print(
            "stance mpc q delta: "
            f"limit={args.stance_mpc_q_delta_limit:.3f} "
            f"scale={args.stance_mpc_q_delta_scale:.2f}"
        )
        for sim_i in range(sim_steps):
            sim_t = float(mujoco_vbot.data.time)
            if viewer is not None and not viewer.is_running():
                break

            if (sim_i % CTRL_DECIM) == 0:
                mujoco_vbot.update_pin_with_mujoco(vbot)
                q_model, dq_model = get_mujoco_joint_vectors(mujoco_vbot)
                if args.stand_only:
                    mixed_cmd = build_stand_mixed_joint_command(args, vbot, q_stand)
                    mixed_cmd.q_target = rate_limit_q_target(
                        apply_joint_soft_limits(mixed_cmd.q_target, args),
                        q_target_prev_model,
                        args,
                    )
                    q_target_prev_model = dict(mixed_cmd.q_target)
                    tau_hold, tau_pd = compute_equivalent_motor_torque(mixed_cmd, q_model, dq_model)
                    contact_mask = mixed_cmd.contact_mask
                elif (ctrl_i % steps_per_mpc) == 0:
                    traj.generate_traj(
                        vbot,
                        gait,
                        sim_t,
                        cmd.x_vel,
                        cmd.y_vel,
                        cmd.z_pos,
                        cmd.yaw_rate,
                        mpc_dt,
                        pitch_des_body=cmd.pitch,
                    )
                    sol = mpc.solve_QP(vbot, traj, False)
                    w_opt = sol["x"].full().flatten()
                    u_opt = w_opt[12 * traj.N :].reshape((12, traj.N), order="F")
                    if args.control_mode == "direct-torque":
                        mpc_force_now = np.asarray(u_opt[:, 0], dtype=float).copy()
                    else:
                        mpc_force_now = filter_mpc_force_now(args, u_opt[:, 0])

                if not args.stand_only and args.control_mode == "direct-torque":
                    tau_model, contact_mask, leg_outputs = compute_direct_leg_torque(
                        vbot,
                        leg_controller,
                        gait,
                        mpc_force_now,
                        sim_t,
                    )
                    tau_hold = np.clip(tau_model, -TAU_LIMIT, TAU_LIMIT)
                    tau_pd = np.zeros(12, dtype=float)
                    mixed_cmd = build_direct_debug_command(q_model, dq_model, tau_hold, contact_mask)
                elif not args.stand_only:
                    q_model, dq_model = get_mujoco_joint_vectors(mujoco_vbot)
                    tau_model, tau_scale, q_target, qd_target, contact_mask, leg_outputs = compute_tau_and_targets(
                        vbot,
                        leg_controller,
                        gait,
                        args,
                        q_stand,
                        q_model,
                        mpc_force_now,
                        sim_t,
                    )
                    for leg in MPC_LEG_ORDER:
                        leg_idx = LEG_INDEX[leg]
                        if int(prev_contact_mask[leg_idx]) == 1 and int(contact_mask[leg_idx]) == 0:
                            for joint in JOINTS[leg]:
                                q_target_prev_model[joint] = float(q_model[joint])
                    q_target = rate_limit_q_target(q_target, q_target_prev_model, args)
                    q_target_prev_model = dict(q_target)
                    mixed_cmd = build_mixed_joint_command(
                        args,
                        vbot,
                        q_target,
                        qd_target,
                        contact_mask,
                        tau_model,
                        tau_scale,
                    )
                    tau_hold, tau_pd = compute_equivalent_motor_torque(mixed_cmd, q_model, dq_model)
                    contact_mask = mixed_cmd.contact_mask
                    prev_contact_mask = np.asarray(contact_mask, dtype=int).reshape(4).copy()

                if args.csv:
                    x_vec = vbot.compute_com_x_vec().reshape(-1)
                    row = {
                        "time_s": sim_t,
                        "base_x": float(x_vec[0]),
                        "base_y": float(x_vec[1]),
                        "base_z": float(x_vec[2]),
                        "roll": float(x_vec[3]),
                        "pitch": float(x_vec[4]),
                        "yaw": float(x_vec[5]),
                        "base_vx": float(x_vec[6]),
                        "contact_mask": "".join(str(int(v)) for v in contact_mask),
                        "max_abs_q_err": float(
                            max(abs(q_model[joint] - mixed_cmd.q_target[joint]) for joint in JOINT_ORDER)
                        ),
                        "tau_pd_max": float(np.max(np.abs(tau_pd))),
                        "tau_bias_max": float(np.max(np.abs(mixed_cmd.tau_bias))),
                        "tau_mpc_max": float(np.max(np.abs(mixed_cmd.tau_mpc))),
                        "tau_max": float(np.max(np.abs(tau_hold))),
                        "mpc_solve_ms": float(getattr(mpc, "solve_time", 0.0)),
                    }
                    for leg in MPC_LEG_ORDER:
                        out = leg_outputs.get(leg)
                        if out is None:
                            row[f"{leg}_foot_des_x"] = float("nan")
                            row[f"{leg}_foot_now_x"] = float("nan")
                            row[f"{leg}_foot_dx"] = float("nan")
                        else:
                            row[f"{leg}_foot_des_x"] = float(np.asarray(out.pos_des).reshape(3)[0])
                            row[f"{leg}_foot_now_x"] = float(np.asarray(out.pos_now).reshape(3)[0])
                            row[f"{leg}_foot_dx"] = row[f"{leg}_foot_des_x"] - row[f"{leg}_foot_now_x"]
                    rows.append(row)

                if sim_t >= next_print:
                    x_vec = vbot.compute_com_x_vec().reshape(-1)
                    print(
                        f"t={sim_t:5.2f}s x={x_vec[0]:+.3f} y={x_vec[1]:+.3f} "
                        f"vx={x_vec[6]:+.3f} mask={''.join(str(int(v)) for v in contact_mask)} "
                        f"tau_pd={np.max(np.abs(tau_pd)):.2f} "
                        f"bias={np.max(np.abs(mixed_cmd.tau_bias)):.2f} "
                        f"mpc_tau={np.max(np.abs(mixed_cmd.tau_mpc)):.2f} "
                        f"tau_total={np.max(np.abs(tau_hold)):.2f} "
                        f"mpc={getattr(mpc, 'solve_time', 0.0):.2f}ms"
                    )
                    if args.print_joints:
                        q_now, dq_now = get_mujoco_joint_vectors(mujoco_vbot)
                        q_print_target = mixed_cmd.q_target
                        qd_print_target = mixed_cmd.qd_target
                        for leg in MPC_LEG_ORDER:
                            role = "stance" if int(contact_mask[LEG_INDEX[leg]]) == 1 else "swing "
                            print(f"  {leg} {role}")
                            out = leg_outputs.get(leg)
                            if out is not None:
                                foot_des = np.asarray(out.pos_des, dtype=float).reshape(3)
                                foot_now = np.asarray(out.pos_now, dtype=float).reshape(3)
                                print(
                                    f"    foot_x now={foot_now[0]:+.4f} "
                                    f"des={foot_des[0]:+.4f} dx={foot_des[0] - foot_now[0]:+.4f}"
                                )
                            for joint in JOINTS[leg]:
                                jidx = JOINT_ORDER.index(joint)
                                q = float(q_now[joint])
                                dq = float(dq_now[joint])
                                qt = float(q_print_target[joint])
                                dqt = float(qd_print_target[joint])
                                print(
                                    f"    {joint:14s} "
                                    f"q={q:+.4f} q_t={qt:+.4f} err={q - qt:+.4f} "
                                    f"dq={dq:+.4f} dq_t={dqt:+.4f} "
                                    f"tau_pd={tau_pd[jidx]:+.2f} "
                                    f"bias={mixed_cmd.tau_bias[jidx]:+.2f} "
                                    f"mpc={mixed_cmd.tau_mpc[jidx]:+.2f} "
                                    f"total={tau_hold[jidx]:+.2f}"
                                )
                    next_print += 1.0 / max(float(args.print_hz), 1.0e-9)
                ctrl_i += 1

            mj.mj_step1(mujoco_vbot.model, mujoco_vbot.data)
            mujoco_vbot.set_joint_torque(tau_hold)
            mj.mj_step2(mujoco_vbot.model, mujoco_vbot.data)

            if viewer is not None and sim_t >= next_viewer:
                viewer.sync()
                next_viewer += 1.0 / VIEWER_HZ
            if args.realtime:
                target_wall = start_wall + sim_t
                sleep_s = target_wall - time.perf_counter()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)

    mujoco_vbot.update_pin_with_mujoco(vbot)
    final_x = float(vbot.compute_com_x_vec().reshape(-1)[0])
    print(f"final_base_x={final_x:+.4f} m")

    if args.csv and rows:
        out = Path(args.csv).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv_saved: {out}")


def main() -> int:
    args = build_arg_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
