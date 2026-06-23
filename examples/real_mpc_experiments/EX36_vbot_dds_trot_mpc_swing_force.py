"""EX36: real VBot trot with stance MPC torque and swing Cartesian torque.

This standalone bring-up script borrows the proven DDS/state-estimation
plumbing from EX34, but changes the walking control policy to better match the
classic Unitree MPC guide:

* stance legs use centroidal-MPC ground-reaction force mapped to joint torque;
* swing legs keep q/qd targets and also send a clipped Cartesian swing-PD
  torque from ``LegController`` instead of forcing swing tau to zero.

It does not import EX34 or EX35. Defaults are intentionally slow and supported:
0.02 m/s forward trot, stance-foot velocity estimation, small MPC feed-forward,
and conservative swing torque scaling.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import inspect
import math
import signal
import sys
import threading
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
CONVEX_SRC = SRC / "convex_mpc"
DEFAULT_AFFINE = REPO / "configs" / "vbot_real_joint_affine.yaml"
DEFAULT_MODEL_POSES = REPO / "configs" / "vbot_model_poses.yaml"

for path in (str(CONVEX_SRC), str(SRC), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from vbot_real_affine import PIN_JOINT_ORDER, MPC_JOINT_ORDER, VBotRealJointAffine, load_model_pose  # noqa: E402

LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}

LEG_INDEX = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
LEG_JOINTS = {
    "FL": ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
    "FR": ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
    "RL": ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
    "RR": ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
}
FOOT_TOUCHDOWN_CLEARANCE = 0.02
LIFT_CONTACT_OFF_ALPHA = 0.08
MOTOR_TORQUE_LIMIT_BY_JOINT_TYPE = {
    "hip": 17.0,
    "thigh": 17.0,
    "calf": 17.0,
}

_GENERATE_TRAJ_SUPPORTS = None


class AllStanceGait:
    def __init__(self, horizon_s: float):
        self.gait_period = float(horizon_s)
        self.gait_duty = 1.0
        self.gait_hz = 1.0 / float(horizon_s)
        self.stance_time = float(horizon_s)
        self.swing_time = 0.0
        self.swing_height = 0.0
        self.contact_mask = None

    def compute_current_mask(self, _time):
        import numpy as np

        if self.contact_mask is None:
            return np.ones(4, dtype=np.int32)
        return np.asarray(self.contact_mask, dtype=np.int32)

    def compute_contact_table(self, _t0, _dt, n):
        import numpy as np

        mask = self.compute_current_mask(_t0)
        return np.repeat(mask.reshape(4, 1), int(n), axis=1)

    def _terrain_height(self, go2, x, y, terrain_height_fn=None):
        height_fn = terrain_height_fn
        if height_fn is None:
            height_fn = getattr(go2, "terrain_height_fn", None)
        if height_fn is None:
            return 0.0
        return float(height_fn(float(x), float(y)))

    def compute_touchdown_world_for_traj_purpose_only(self, go2, leg: str, terrain_height_fn=None):
        import numpy as np

        try:
            foot_pos_world, _foot_vel_world = go2.get_single_foot_state_in_world(leg)
            pos_touchdown_world = np.asarray(foot_pos_world, dtype=float).reshape(3).copy()
        except Exception:
            base_pos = np.asarray(go2.current_config.base_pos, dtype=float).reshape(3)
            hip_offset = np.asarray(go2.get_hip_offset(leg), dtype=float).reshape(3)
            r_z = np.asarray(getattr(go2, "R_z", np.eye(3)), dtype=float).reshape(3, 3)
            body_xy_pos = np.array([base_pos[0], base_pos[1], 0.0])
            pos_touchdown_world = body_xy_pos + r_z @ hip_offset

        pos_touchdown_world[2] = (
            self._terrain_height(
                go2,
                pos_touchdown_world[0],
                pos_touchdown_world[1],
                terrain_height_fn,
            )
            + FOOT_TOUCHDOWN_CLEARANCE
        )
        return pos_touchdown_world

    def make_swing_trajectory(self, p0, pf, t_swing, h_sw=0.0):
        import numpy as np

        p0 = np.asarray(p0, dtype=float).reshape(3)
        pf = np.asarray(pf, dtype=float).reshape(3)
        duration = max(float(t_swing), 1.0e-3)
        height = max(float(h_sw), 0.0)
        dp = pf - p0

        def eval_at(t):
            s = np.clip(float(t) / duration, 0.0, 1.0)
            mj = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
            dmj = 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4
            d2mj = 60.0 * s - 180.0 * s**2 + 120.0 * s**3
            pos = p0 + dp * mj
            vel = dp * dmj / duration
            acc = dp * d2mj / (duration**2)
            if height > 0.0:
                bump = 4.0 * s * (1.0 - s)
                dbump = 4.0 * (1.0 - 2.0 * s)
                d2bump = -8.0
                pos[2] += height * bump
                vel[2] += height * dbump / duration
                acc[2] += height * d2bump / (duration**2)
            return pos, vel, acc

        return eval_at

    def compute_swing_traj_and_touchdown(self, go2, leg: str):
        import numpy as np

        try:
            foot_pos, _foot_vel = go2.get_single_foot_state_in_world(leg)
            foot_pos = np.asarray(foot_pos, dtype=float).reshape(3)
        except Exception:
            foot_pos = self.compute_touchdown_world_for_traj_purpose_only(go2, leg)
        touchdown = self.compute_touchdown_world_for_traj_purpose_only(go2, leg)
        return self.make_swing_trajectory(foot_pos, touchdown, self.swing_time, self.swing_height), touchdown


class WarmupGait:
    def __init__(self, gait, warmup_s: float):
        self.gait = gait
        self.warmup_s = max(0.0, float(warmup_s))

    def __getattr__(self, name):
        return getattr(self.gait, name)

    def _active_time(self, time_s: float) -> float:
        return max(0.0, float(time_s) - self.warmup_s)

    def compute_current_mask(self, time_s):
        import numpy as np

        if float(time_s) < self.warmup_s:
            return np.ones(4, dtype=np.int32)
        return self.gait.compute_current_mask(self._active_time(time_s))

    def compute_contact_table(self, t0, dt, n):
        import numpy as np

        table = np.ones((4, int(n)), dtype=np.int32)
        for k in range(int(n)):
            t = float(t0) + float(dt) * (k + 0.5)
            if t >= self.warmup_s:
                table[:, k] = self.gait.compute_current_mask(self._active_time(t)).reshape(4)
        return table

    def compute_touchdown_world_for_traj_purpose_only(self, go2, leg: str, terrain_height_fn=None):
        fn = self.gait.compute_touchdown_world_for_traj_purpose_only
        params = inspect.signature(fn).parameters
        has_varargs = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params.values())
        if terrain_height_fn is not None and (has_varargs or "terrain_height_fn" in params):
            return fn(go2, leg, terrain_height_fn)
        return fn(go2, leg)

    def make_swing_trajectory(self, p0, pf, t_swing, h_sw=0.0):
        fn = self.gait.make_swing_trajectory
        params = inspect.signature(fn).parameters
        has_varargs = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params.values())
        if has_varargs or "h_sw" in params:
            return fn(p0, pf, t_swing, h_sw)
        return fn(p0, pf, t_swing)

    def compute_swing_traj_and_touchdown(self, go2, leg: str):
        return self.gait.compute_swing_traj_and_touchdown(go2, leg)

    @property
    def gait_period(self):
        return self.gait.gait_period

    @property
    def gait_duty(self):
        return self.gait.gait_duty

    @property
    def gait_hz(self):
        return self.gait.gait_hz

    @property
    def stance_time(self):
        return self.gait.stance_time

    @property
    def swing_time(self):
        return self.gait.swing_time

    @property
    def swing_height(self):
        return self.gait.swing_height

    @swing_height.setter
    def swing_height(self, value):
        self.gait.swing_height = float(value)


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
        import numpy as np

        t = float(time_s) % max(self.gait_period, 1.0e-9)
        if t < self.swing_time:
            return np.array([0, 1, 1, 0], dtype=np.int32)
        if t < self.step_period:
            return np.ones(4, dtype=np.int32)
        if t < self.step_period + self.swing_time:
            return np.array([1, 0, 0, 1], dtype=np.int32)
        return np.ones(4, dtype=np.int32)

    def compute_contact_table(self, t0, dt, n):
        import numpy as np

        table = np.ones((4, int(n)), dtype=np.int32)
        for k in range(int(n)):
            t = float(t0) + float(dt) * (k + 0.5)
            table[:, k] = self.compute_current_mask(t).reshape(4)
        return table

    def compute_touchdown_world_for_traj_purpose_only(self, go2, leg: str, terrain_height_fn=None):
        fn = self.gait.compute_touchdown_world_for_traj_purpose_only
        params = inspect.signature(fn).parameters
        has_varargs = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params.values())
        if terrain_height_fn is not None and (has_varargs or "terrain_height_fn" in params):
            return fn(go2, leg, terrain_height_fn)
        return fn(go2, leg)

    def make_swing_trajectory(self, p0, pf, t_swing, h_sw=0.0):
        fn = self.gait.make_swing_trajectory
        params = inspect.signature(fn).parameters
        has_varargs = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params.values())
        if has_varargs or "h_sw" in params:
            return fn(p0, pf, t_swing, h_sw)
        return fn(p0, pf, t_swing)

    def compute_swing_traj_and_touchdown(self, go2, leg: str):
        return self.gait.compute_swing_traj_and_touchdown(go2, leg)


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, alpha: float) -> float:
    return (1.0 - alpha) * float(a) + alpha * float(b)


def dict_interpolate(a, b, alpha: float):
    return {joint: (1.0 - alpha) * a[joint] + alpha * b[joint] for joint in a}


def field(obj, name: str):
    value = getattr(obj, name)
    return value() if callable(value) else value


def install_stop_handlers():
    stop = {"requested": False}

    def on_signal(_signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    return stop


def advance_periodic_deadline(next_t: float, period: float, now: float) -> float:
    if math.isinf(period):
        return math.inf
    next_t += period
    if next_t <= now:
        missed = math.floor((now - next_t) / period) + 1
        next_t += missed * period
    return next_t


def apply_deadband(value: float, deadband: float) -> float:
    return 0.0 if abs(value) < deadband else value


def rpy_to_xyzw(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quat_wxyz_to_rpy(w: float, x: float, y: float, z: float):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def read_imu_rpy_from_lowstate(imu):
    quat = field(imu, "quaternion")
    q = [float(quat[i]) for i in range(4)]
    norm = math.sqrt(sum(v * v for v in q))
    if norm > 1.0e-9:
        w, x, y, z = [v / norm for v in q]
        return quat_wxyz_to_rpy(w, x, y, z), "imu_quat"

    try:
        rpy = field(imu, "rpy")
        return (float(rpy[0]), float(rpy[1]), float(rpy[2])), "imu_rpy"
    except Exception:
        return (0.0, 0.0, 0.0), "imu_missing"


def read_lowstate_motor_feedback(msg, order):
    motor_state = field(msg, "motor_state")
    cache = {}
    for i, joint in enumerate(order):
        state = motor_state[i]
        cache[joint] = {
            "q": float(field(state, "q")),
            "dq": float(field(state, "dq")),
            "tau_est": float(field(state, "tau_est")),
            "lost": int(field(state, "lost")),
        }
    return cache


def read_base_state(msg, args, imu_zero_rp=(0.0, 0.0)):
    if not args.use_imu_base_state:
        return {
            "source": "fixed",
            "roll": float(args.base_roll),
            "pitch": float(args.base_pitch),
            "yaw": float(args.base_yaw),
            "gyro": (0.0, 0.0, 0.0),
        }

    imu = field(msg, "imu_state")
    gyro = field(imu, "gyroscope")
    (raw_roll, raw_pitch, raw_yaw), source = read_imu_rpy_from_lowstate(imu)
    roll = args.imu_roll_sign * (raw_roll - float(imu_zero_rp[0]))
    pitch = args.imu_pitch_sign * (raw_pitch - float(imu_zero_rp[1]))

    if args.fix_gateway_gyro_order:
        raw_gx, raw_gy, raw_gz = float(gyro[2]), float(gyro[1]), float(gyro[0])
    else:
        raw_gx, raw_gy, raw_gz = float(gyro[0]), float(gyro[1]), float(gyro[2])
    gx = args.imu_roll_sign * apply_deadband(raw_gx, args.imu_gyro_deadband)
    gy = args.imu_pitch_sign * apply_deadband(raw_gy, args.imu_gyro_deadband)
    gz = 0.0
    return {
        "source": source,
        "roll": roll,
        "pitch": pitch,
        "yaw": 0.0,
        "gyro": (gx, gy, gz),
    }


def estimate_imu_zero_rp(latest, args):
    if not args.use_imu_base_state or not args.imu_rp_zero_on_start:
        return (0.0, 0.0)
    deadline = time.monotonic() + max(0.0, args.imu_rp_zero_seconds)
    samples = []
    while time.monotonic() < deadline:
        msg = latest["msg"]
        if msg is not None:
            try:
                imu = field(msg, "imu_state")
                (roll, pitch, _yaw), _source = read_imu_rpy_from_lowstate(imu)
                samples.append((roll, pitch))
            except Exception:
                pass
        time.sleep(0.01)
    if not samples:
        return (0.0, 0.0)
    roll = sum(v[0] for v in samples) / len(samples)
    pitch = sum(v[1] for v in samples) / len(samples)
    print(f"IMU roll/pitch zero: roll={roll:+.5f} pitch={pitch:+.5f} samples={len(samples)}")
    return (roll, pitch)


def cache_to_motor_dict(cache, order, key: str) -> dict[str, float]:
    return {joint: float(cache[joint][key]) for joint in order}


def zero_joint_dict(order) -> dict[str, float]:
    return {joint: 0.0 for joint in order}


def feedback_to_model_dict(cache, bridge, args, dds_model_offset):
    q_dds = cache_to_motor_dict(cache, bridge.order, "q")
    if args.dds_model_space:
        return {joint: q_dds[joint] - dds_model_offset[joint] for joint in bridge.order}
    return bridge.motor_feedback_to_model_dict(q_dds)


def feedback_to_model_velocity_dict(cache, bridge, args):
    dq_dds = cache_to_motor_dict(cache, bridge.order, "dq")
    if args.dds_model_space:
        return dq_dds
    return bridge.motor_feedback_to_model_velocity_dict(dq_dds)


def model_command_to_dds_dict(q_model_by_joint, bridge, args, dds_model_offset):
    if args.dds_model_space:
        return {
            joint: float(q_model_by_joint[joint]) + dds_model_offset[joint]
            for joint in bridge.order
        }
    return bridge.motor_position_commands_from_model_dict(q_model_by_joint)


def model_velocity_command_to_dds_dict(dq_model_by_joint, bridge, args):
    if args.dds_model_space:
        return {joint: float(dq_model_by_joint[joint]) for joint in bridge.order}
    return {
        joint: float(bridge.model_velocity_to_motor(joint, dq_model_by_joint[joint]))
        for joint in bridge.order
    }


def lost_joints(cache, order) -> list[str]:
    return [joint for joint in order if int(cache[joint]["lost"]) != 0]


def validate_joint_order_compatibility(bridge):
    if len(set(MPC_JOINT_ORDER)) != 12:
        raise RuntimeError("MPC_JOINT_ORDER contains duplicate joints")
    if len(set(bridge.order)) != 12:
        raise RuntimeError("DDS bridge.order contains duplicate joints")
    if set(MPC_JOINT_ORDER) != set(bridge.order):
        missing_from_dds = sorted(set(MPC_JOINT_ORDER) - set(bridge.order))
        missing_from_mpc = sorted(set(bridge.order) - set(MPC_JOINT_ORDER))
        raise RuntimeError(
            "MPC/DDS joint sets differ: "
            f"missing_from_dds={missing_from_dds}, missing_from_mpc={missing_from_mpc}"
        )


def overlay_schedule(args, elapsed_s):
    final_kp = args.kp if args.final_kp is None else args.final_kp
    final_kd = args.kd if args.final_kd is None else args.final_kd
    alpha = 1.0 if args.handover_seconds <= 1.0e-9 else smoothstep(elapsed_s / args.handover_seconds)
    return (
        alpha,
        lerp(args.kp, final_kp, alpha),
        lerp(args.kd, final_kd, alpha),
    )


def motor_torque_limit_vector(order):
    import numpy as np

    limits = []
    for joint in order:
        if "_hip_" in joint:
            limits.append(MOTOR_TORQUE_LIMIT_BY_JOINT_TYPE["hip"])
        elif "_thigh_" in joint:
            limits.append(MOTOR_TORQUE_LIMIT_BY_JOINT_TYPE["thigh"])
        elif "_calf_" in joint:
            limits.append(MOTOR_TORQUE_LIMIT_BY_JOINT_TYPE["calf"])
        else:
            raise RuntimeError(f"unknown joint type for torque limit: {joint}")
    return np.asarray(limits, dtype=float)


def limit_tau_to_motor_range(tau_motor_raw, tau_limits):
    import numpy as np

    raw = np.asarray(tau_motor_raw, dtype=float)
    limits = np.asarray(tau_limits, dtype=float)
    if raw.shape != limits.shape:
        raise RuntimeError(f"tau limit shape mismatch: raw={raw.shape}, limits={limits.shape}")
    clipped = np.clip(raw, -limits, limits)
    return clipped, clipped, 1.0, int(np.count_nonzero(np.abs(raw) > limits + 1e-12))


def configure_lowcmd(
    cmd,
    mode: int,
    order=(),
    q_motor_by_joint=None,
    tau_motor=None,
    kp: float = 0.0,
    kd: float = 0.0,
    qd_motor_by_joint=None,
):
    if tau_motor is not None and len(tau_motor) < 12:
        raise RuntimeError(f"expected at least 12 tau commands, got {len(tau_motor)}")

    def gain_value(gain, joint):
        if isinstance(gain, dict):
            return float(gain[joint])
        return float(gain)

    for i in range(20):
        motor = cmd.motor_cmd[i]
        motor.mode = int(mode) if i < 12 else 0
        if q_motor_by_joint is not None and i < len(order):
            joint = order[i]
            motor.q = float(q_motor_by_joint[joint])
            motor.kp = gain_value(kp, joint)
            motor.kd = gain_value(kd, joint)
        else:
            motor.q = 0.0
            motor.kp = 0.0
            motor.kd = 0.0
        motor.dq = float(qd_motor_by_joint[order[i]]) if qd_motor_by_joint is not None and i < len(order) else 0.0
        motor.tau = float(tau_motor[i]) if tau_motor is not None and i < 12 else 0.0
    return cmd


def make_lowcmd(
    mode: int,
    order=(),
    q_motor_by_joint=None,
    tau_motor=None,
    kp: float = 0.0,
    kd: float = 0.0,
    qd_motor_by_joint=None,
):
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    return configure_lowcmd(
        cmd,
        mode,
        order,
        q_motor_by_joint,
        tau_motor,
        kp,
        kd,
        qd_motor_by_joint=qd_motor_by_joint,
    )


def make_crc():
    try:
        from unitree_sdk2py.utils.crc import CRC

        return CRC()
    except Exception:
        return None


def publish_lowcmd(pub, cmd, crc):
    if crc is not None:
        cmd.crc = crc.Crc(cmd)
    pub.Write(cmd)


def publish_for(
    pub,
    cmd,
    crc,
    hz: float,
    seconds: float,
    stop=None,
    latest=None,
    feedback_timeout_s=None,
):
    period = 1.0 / hz
    deadline = time.monotonic() + max(0.0, seconds)
    next_t = time.monotonic()
    while time.monotonic() < deadline and not (stop and stop["requested"]):
        now = time.monotonic()
        if latest is not None and feedback_timeout_s is not None:
            if now - latest["t"] > feedback_timeout_s:
                raise RuntimeError(
                    f"rt/lowstate timeout during publish_for > {feedback_timeout_s:.3f}s"
                )
        if now >= next_t:
            publish_lowcmd(pub, cmd, crc)
            next_t = advance_periodic_deadline(next_t, period, now)
        time.sleep(0.001)


class PeriodicLowcmdPublisher:
    def __init__(self, pub, crc, order, hz: float, stop, latest=None, feedback_timeout_s=None):
        self.pub = pub
        self.crc = crc
        self.order = tuple(order)
        self.period = 1.0 / float(hz)
        self.stop = stop
        self.latest = latest
        self.feedback_timeout_s = feedback_timeout_s
        self.lock = threading.Lock()
        self.cmd_msg = make_lowcmd(1, self.order)
        self.mode = 1
        self.q_motor_by_joint = None
        self.qd_motor_by_joint = None
        self.tau_motor = None
        self.kp = 0.0
        self.kd = 0.0
        self.times = []
        self.publish_time_sum = 0.0
        self.publish_time_max = 0.0
        self.error = None
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ex36-lowcmd-publisher", daemon=True)

    def update(
        self,
        mode: int,
        q_motor_by_joint=None,
        tau_motor=None,
        kp: float = 0.0,
        kd: float = 0.0,
        qd_motor_by_joint=None,
    ):
        tau_copy = None if tau_motor is None else [float(v) for v in tau_motor]
        q_copy = None if q_motor_by_joint is None else dict(q_motor_by_joint)
        qd_copy = None if qd_motor_by_joint is None else dict(qd_motor_by_joint)
        with self.lock:
            self.mode = int(mode)
            self.q_motor_by_joint = q_copy
            self.qd_motor_by_joint = qd_copy
            self.tau_motor = tau_copy
            self.kp = dict(kp) if isinstance(kp, dict) else float(kp)
            self.kd = dict(kd) if isinstance(kd, dict) else float(kd)

    def start(self):
        self._thread.start()

    def stop_and_join(self, timeout=1.0):
        self._done.set()
        self._thread.join(timeout=timeout)

    def snapshot_stats(self):
        with self.lock:
            times = list(self.times)
            publish_time_sum = float(self.publish_time_sum)
            publish_time_max = float(self.publish_time_max)
            error = self.error
        return {
            "times": times,
            "count": len(times),
            "publish_time_sum": publish_time_sum,
            "publish_time_max": publish_time_max,
            "error": error,
        }

    def get_error(self):
        with self.lock:
            return self.error

    def _run(self):
        next_t = time.monotonic()
        while not self._done.is_set() and not self.stop["requested"]:
            now = time.monotonic()
            if self.latest is not None and self.feedback_timeout_s is not None:
                if now - self.latest["t"] > self.feedback_timeout_s:
                    with self.lock:
                        self.error = f"rt_lowstate_timeout>{self.feedback_timeout_s:.3f}s"
                    return
            if now >= next_t:
                with self.lock:
                    configure_lowcmd(
                        self.cmd_msg,
                        self.mode,
                        self.order,
                        self.q_motor_by_joint,
                        self.tau_motor,
                        self.kp,
                        self.kd,
                        qd_motor_by_joint=self.qd_motor_by_joint,
                    )
                publish_t0 = time.monotonic()
                try:
                    publish_lowcmd(self.pub, self.cmd_msg, self.crc)
                except Exception as exc:
                    with self.lock:
                        self.error = f"{type(exc).__name__}: {exc}"
                    return
                publish_dt = time.monotonic() - publish_t0
                with self.lock:
                    self.times.append(publish_t0)
                    self.publish_time_sum += publish_dt
                    self.publish_time_max = max(self.publish_time_max, publish_dt)
                next_t = advance_periodic_deadline(next_t, self.period, publish_t0)
            time.sleep(min(0.001, max(0.0, next_t - time.monotonic())))


def compute_dds_model_offset_if_requested(bridge, cache, args):
    offset = zero_joint_dict(bridge.order)
    if not args.dds_model_space:
        return offset, False

    if args.anchor_current_to_target_on_start:
        pose_name = args.target_pose
    elif args.prone_calibrate_on_start:
        pose_name = args.prone_pose
    else:
        return offset, False

    q_dds = cache_to_motor_dict(cache, bridge.order, "q")
    pose = load_model_pose(args.model_poses, pose_name, bridge.order)
    offset = {joint: q_dds[joint] - pose[joint] for joint in bridge.order}
    print(
        f"DDS model-space anchor: current lowstate.q -> "
        f"{Path(args.model_poses).name}:{pose_name}"
    )
    print("joint                 q_dds_now   offset  mapped_err")
    print("-" * 58)
    for joint in bridge.order:
        err = q_dds[joint] - offset[joint] - pose[joint]
        print(f"{joint:20s} {q_dds[joint]:+10.4f} {offset[joint]:+8.4f} {err:+11.3e}")
    return offset, True


def write_summary(
    log_path,
    exit_reason,
    overlay_elapsed_s,
    overlay_loop_count,
    latest,
    overlay_lowcmd_times,
    extra=None,
):
    if log_path is None:
        return
    intervals = [
        b - a for a, b in zip(overlay_lowcmd_times[:-1], overlay_lowcmd_times[1:])
    ]
    summary_path = Path(str(log_path).replace(".csv", "_summary.txt"))
    last_feedback_age = time.monotonic() - latest["t"] if latest.get("t", 0.0) else math.inf
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"exit_reason={exit_reason}\n")
        f.write(f"overlay_elapsed_s={overlay_elapsed_s:.6f}\n")
        f.write(f"overlay_loop_count={overlay_loop_count}\n")
        f.write(f"last_feedback_age_s={last_feedback_age:.6f}\n")
        f.write(f"overlay_lowcmd_count={len(overlay_lowcmd_times)}\n")
        if intervals:
            f.write(f"overlay_lowcmd_interval_mean_s={sum(intervals) / len(intervals):.6f}\n")
            f.write(f"overlay_lowcmd_interval_min_s={min(intervals):.6f}\n")
            f.write(f"overlay_lowcmd_interval_max_s={max(intervals):.6f}\n")
        else:
            f.write("overlay_lowcmd_interval_mean_s=nan\n")
            f.write("overlay_lowcmd_interval_min_s=nan\n")
            f.write("overlay_lowcmd_interval_max_s=nan\n")
        for key, value in (extra or {}).items():
            f.write(f"{key}={value}\n")
    print(f"summary_saved: {summary_path}")


def mask_text(contact_mask) -> str:
    return "".join(str(int(v)) for v in contact_mask)


def lift_schedule(args, elapsed_s):
    if args.lift_leg is None:
        return 0.0
    start = args.lift_start_s if args.lift_start_s is not None else 2.0
    ramp = args.lift_ramp_s if args.lift_ramp_s is not None else 2.0
    hold = args.lift_hold_s if args.lift_hold_s is not None else 2.0
    if elapsed_s < start:
        return 0.0
    if elapsed_s < start + ramp:
        return smoothstep((elapsed_s - start) / max(ramp, 1.0e-9))
    if elapsed_s < start + ramp + hold:
        return 1.0
    if elapsed_s < start + 2.0 * ramp + hold:
        return 1.0 - smoothstep((elapsed_s - start - ramp - hold) / max(ramp, 1.0e-9))
    return 0.0


def lift_target_and_contact(args, q_stand_model, elapsed_s):
    q_target = dict(q_stand_model)
    contact_mask = [1, 1, 1, 1]
    lift_alpha = lift_schedule(args, elapsed_s)
    if args.lift_leg is None or lift_alpha <= 0.0:
        return q_target, contact_mask, lift_alpha
    thigh = LEG_JOINTS[args.lift_leg][1]
    calf = LEG_JOINTS[args.lift_leg][2]
    q_target[thigh] += lift_alpha * args.lift_thigh_offset
    q_target[calf] += lift_alpha * args.lift_calf_offset
    if lift_alpha > LIFT_CONTACT_OFF_ALPHA:
        contact_mask[LEG_INDEX[args.lift_leg]] = 0
    return q_target, contact_mask, lift_alpha


def generate_traj_compat(
    traj,
    vbot,
    gait,
    time_s: float,
    x_vel: float,
    y_vel: float,
    z_pos: float,
    yaw_rate: float,
    time_step: float,
    pitch_des_body: float,
):
    global _GENERATE_TRAJ_SUPPORTS

    if _GENERATE_TRAJ_SUPPORTS is None:
        params = inspect.signature(traj.generate_traj).parameters
        has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
        _GENERATE_TRAJ_SUPPORTS = {
            "pitch_des_body": "pitch_des_body" in params or has_kwargs,
        }

    common_args = (vbot, gait, time_s, x_vel, y_vel, z_pos, yaw_rate)
    if _GENERATE_TRAJ_SUPPORTS["pitch_des_body"]:
        traj.generate_traj(
            *common_args,
            time_step=time_step,
            pitch_des_body=pitch_des_body,
        )
        return

    traj.generate_traj(*common_args, time_step=time_step)
    if hasattr(traj, "rpy_traj_world"):
        traj.rpy_traj_world[1, :] = float(pitch_des_body)
    if hasattr(traj, "omega_traj_world"):
        traj.omega_traj_world[1, :] = 0.0


class RealStateEstimator:
    def __init__(self, args):
        self.base_xy = [0.0, 0.0]
        self.last_t = None
        self.last_state = None
        self.args = args

    def update(self, vbot, bridge, cache, args, dds_model_offset, base, contact_mask, now_s):
        import numpy as np

        q_model = feedback_to_model_dict(cache, bridge, args, dds_model_offset)
        dq_model = feedback_to_model_velocity_dict(cache, bridge, args)
        contact_mask = np.asarray(contact_mask, dtype=int).reshape(4)
        support_legs = [leg for leg in ("FL", "FR", "RL", "RR") if contact_mask[LEG_INDEX[leg]] == 1]

        dt = 0.0 if self.last_t is None else max(0.0, float(now_s - self.last_t))
        self.last_t = float(now_s)

        q = vbot.current_config.get_q().copy()
        dq = vbot.current_config.get_dq().copy()
        q[0:3] = [self.base_xy[0], self.base_xy[1], args.base_height]
        q[3:7] = rpy_to_xyzw(base["roll"], base["pitch"], base["yaw"])
        q[7:19] = [q_model[joint] for joint in PIN_JOINT_ORDER]
        dq[0:3] = [0.0, 0.0, 0.0]
        dq[3:6] = list(base["gyro"])
        dq[6:18] = [dq_model[joint] for joint in PIN_JOINT_ORDER]

        height_source = "fixed"
        if args.base_height_mode == "stance-feet" and support_legs:
            q_zero_z = q.copy()
            q_zero_z[2] = 0.0
            vbot.update_model(q_zero_z, dq)
            foot_z = [float(vbot.get_single_foot_state_in_world(leg)[0][2]) for leg in support_legs]
            z_est = float(args.foot_ground_z - sum(foot_z) / len(foot_z))
            if args.base_height_clip > 0.0:
                z_est = float(
                    np.clip(
                        z_est,
                        args.base_height - args.base_height_clip,
                        args.base_height + args.base_height_clip,
                    )
                )
            q[2] = z_est
            height_source = "stance_feet"

        vbot.update_model(q, dq)

        vel_source = "zero"
        if args.base_vel_mode == "stance-feet" and support_legs:
            foot_vel = [vbot.get_single_foot_state_in_world(leg)[1] for leg in support_legs]
            base_vel = -np.mean(np.asarray(foot_vel, dtype=float), axis=0)
            base_vel[0:2] = np.clip(base_vel[0:2], -args.max_base_xy_vel, args.max_base_xy_vel)
            base_vel[2] = float(np.clip(base_vel[2], -args.max_base_z_vel, args.max_base_z_vel))
            dq[0:3] = base_vel
            vel_source = "stance_feet"
        else:
            base_vel = np.zeros(3, dtype=float)

        if args.integrate_base_xy and dt > 0.0:
            self.base_xy[0] += float(base_vel[0]) * dt
            self.base_xy[1] += float(base_vel[1]) * dt
            q[0:2] = self.base_xy

        vbot.update_model(q, dq)
        state = {
            "source": base["source"],
            "height_source": height_source,
            "vel_source": vel_source,
            "x": float(q[0]),
            "y": float(q[1]),
            "z": float(q[2]),
            "roll": float(base["roll"]),
            "pitch": float(base["pitch"]),
            "yaw": float(base["yaw"]),
            "vx": float(dq[0]),
            "vy": float(dq[1]),
            "vz": float(dq[2]),
            "gyro": tuple(float(v) for v in base["gyro"]),
            "contact_mask": contact_mask,
        }
        self.last_state = state
        return state


def build_arg_parser(profile_defaults=None):
    parser = argparse.ArgumentParser(description="EX36 real VBot trot: stance MPC tau + swing Cartesian tau")
    parser.add_argument("--test-profile", choices=("custom", "trot-final", "all-stance"), default="custom")
    parser.add_argument("--config", default=None, help="YAML file with EX36 argument defaults")
    parser.add_argument("--network", default="lo")
    parser.add_argument("--affine", default=str(DEFAULT_AFFINE))
    parser.add_argument("--model-poses", default=str(DEFAULT_MODEL_POSES))
    parser.add_argument("--target-pose", default="stand")
    parser.add_argument("--dds-model-space", action="store_true", default=True)
    parser.add_argument("--dds-raw-motor-space", action="store_false", dest="dds_model_space")
    parser.add_argument("--prone-calibrate-on-start", action="store_true")
    parser.add_argument("--prone-pose", default="down")
    parser.add_argument("--anchor-current-to-target-on-start", action="store_true")

    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--startup-ramp-seconds", type=float, default=12.0)
    parser.add_argument("--prehold-seconds", type=float, default=1.0)
    parser.add_argument("--cmd-hz", type=float, default=100.0)
    parser.add_argument("--state-hz", type=float, default=50.0)
    parser.add_argument("--leg-control-hz", type=float, default=None)
    parser.add_argument("--mpc-hz", type=float, default=5.0)
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--control-sleep-s", type=float, default=0.0002)
    parser.add_argument("--wait-lowstate-s", type=float, default=5.0)
    parser.add_argument("--feedback-timeout-s", type=float, default=0.25)

    parser.add_argument("--kp", type=float, default=12.0)
    parser.add_argument("--kd", type=float, default=0.8)
    parser.add_argument("--final-kp", type=float, default=None)
    parser.add_argument("--final-kd", type=float, default=None)
    parser.add_argument("--swing-kp", type=float, default=None)
    parser.add_argument("--swing-kd", type=float, default=None)
    parser.add_argument("--handover-seconds", type=float, default=3.0)
    parser.add_argument("--allow-large-gains", action="store_true")
    parser.add_argument("--abort-on-large-error", action="store_true")
    parser.add_argument("--abort-qerr", type=float, default=0.45)
    parser.add_argument("--abort-swing-qerr", type=float, default=None)
    parser.add_argument("--abort-tilt", type=float, default=0.08)
    parser.add_argument("--allow-large-tau-limit", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--base-height", type=float, default=0.462)
    parser.add_argument("--base-height-mode", choices=("fixed", "stance-feet"), default="stance-feet")
    parser.add_argument("--base-height-clip", type=float, default=0.06)
    parser.add_argument("--foot-ground-z", type=float, default=0.0)
    parser.add_argument("--base-vel-mode", choices=("zero", "stance-feet"), default="stance-feet")
    parser.add_argument("--max-base-xy-vel", type=float, default=0.30)
    parser.add_argument("--max-base-z-vel", type=float, default=0.20)
    parser.add_argument("--integrate-base-xy", action="store_true")
    parser.add_argument("--base-roll", type=float, default=0.0)
    parser.add_argument("--base-pitch", type=float, default=0.0)
    parser.add_argument("--base-yaw", type=float, default=0.0)
    parser.add_argument("--use-imu-base-state", action="store_true", default=True)
    parser.add_argument("--fixed-base-state", action="store_false", dest="use_imu_base_state")
    parser.add_argument("--imu-rp-zero-on-start", action="store_true")
    parser.add_argument("--imu-rp-zero-seconds", type=float, default=0.5)
    parser.add_argument("--imu-roll-sign", type=float, default=1.0)
    parser.add_argument("--imu-pitch-sign", type=float, default=1.0)
    parser.add_argument("--imu-gyro-deadband", type=float, default=0.005)
    parser.add_argument("--fix-gateway-gyro-order", action="store_true")

    parser.add_argument("--gait", choices=("all-stance", "trot"), default="all-stance")
    parser.add_argument("--allow-swing-control", action="store_true")
    parser.add_argument("--gait-frequency-hz", type=float, default=1.5)
    parser.add_argument("--gait-duty", type=float, default=0.75)
    parser.add_argument("--gait-warmup-s", type=float, default=0.0)
    parser.add_argument("--swing-height", type=float, default=0.04)
    parser.add_argument("--swing-q-target-delta-limit", type=float, default=0.0)
    parser.add_argument("--stance-mpc-q-delta-limit", type=float, default=0.0)
    parser.add_argument("--stance-mpc-q-delta-scale", type=float, default=1.0)
    parser.add_argument("--tau-ff-scale", type=float, default=1.0)
    parser.add_argument("--swing-tau-ff-scale", type=float, default=0.20)
    parser.add_argument("--swing-tau-limit", type=float, default=2.0)
    parser.add_argument("--horizon-seconds", type=float, default=0.40)
    parser.add_argument("--horizon-segments", type=int, default=16)
    parser.add_argument("--x-vel", type=float, default=0.0)
    parser.add_argument("--y-vel", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--mpc-r", type=float, default=3.0e-3)
    parser.add_argument("--mpc-force-xy-scale", type=float, default=1.0)
    parser.add_argument("--yaw-weight", type=float, default=0.0)
    parser.add_argument("--yaw-rate-weight", type=float, default=0.0)
    parser.add_argument("--stance-bias-comp", action="store_true", default=True)
    parser.add_argument("--verbose-mpc", action="store_true")
    parser.add_argument("--lift-leg", choices=("FL", "FR", "RL", "RR"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lift-start-s", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lift-ramp-s", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lift-hold-s", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lift-thigh-offset", type=float, default=0.08, help=argparse.SUPPRESS)
    parser.add_argument("--lift-calf-offset", type=float, default=-0.16, help=argparse.SUPPRESS)
    parser.add_argument("--allow-single-leg-lift", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--zero-on-exit-seconds", type=float, default=0.8)
    parser.add_argument("--return-pose-on-exit", default="down")
    parser.add_argument("--return-ramp-seconds", type=float, default=5.0)
    parser.add_argument("--disable-on-exit", action="store_true")
    parser.add_argument("--log-dir", default=str(REPO / "logs" / "real_mpc"))
    parser.add_argument("--log-prefix", default="ex36_trot_mpc_swing_force")
    parser.add_argument("--no-log", action="store_true")

    parser.add_argument("--robot-standing-supported", action="store_true")
    parser.add_argument("--i-accept-risk", action="store_true")
    parser.add_argument("--allow-long-duration", action="store_true")
    if profile_defaults:
        parser.set_defaults(**profile_defaults)
    return parser


PROFILE_DEFAULTS = {
    "all-stance": {
        "gait": "all-stance",
        "allow_swing_control": False,
        "duration": 6.0,
        "startup_ramp_seconds": 12.0,
        "prehold_seconds": 2.0,
        "cmd_hz": 100.0,
        "state_hz": 50.0,
        "leg_control_hz": 100.0,
        "mpc_hz": 5.0,
        "kp": 50.0,
        "kd": 3.0,
        "final_kp": 50.0,
        "final_kd": 3.0,
        "handover_seconds": 0.0,
        "base_height_mode": "stance-feet",
        "base_vel_mode": "stance-feet",
        "use_imu_base_state": True,
        "imu_rp_zero_on_start": True,
        "fix_gateway_gyro_order": True,
        "abort_on_large_error": True,
        "abort_qerr": 0.60,
        "abort_tilt": 0.15,
        "return_pose_on_exit": "down",
        "return_ramp_seconds": 5.0,
        "disable_on_exit": True,
        "allow_large_gains": True,
    },
    "trot-final": {
        "target_pose": "stand",
        "prone_calibrate_on_start": False,
        "prone_pose": "down",
        "duration": 8.0,
        "startup_ramp_seconds": 12.0,
        "prehold_seconds": 1.0,
        "cmd_hz": 100.0,
        "state_hz": 100.0,
        "leg_control_hz": 100.0,
        "mpc_hz": 10.0,
        "kp": 50.0,
        "kd": 3.0,
        "final_kp": 50.0,
        "final_kd": 3.0,
        "swing_kp": 10.0,
        "swing_kd": 3.0,
        "handover_seconds": 1.5,
        "base_height": 0.462,
        "base_height_mode": "stance-feet",
        "base_height_clip": 0.04,
        "base_vel_mode": "stance-feet",
        "gait": "trot",
        "allow_swing_control": True,
        "gait_warmup_s": 2.0,
        "gait_frequency_hz": 1.0,
        "gait_duty": 0.70,
        "swing_height": 0.05,
        "swing_q_target_delta_limit": 0.22,
        "stance_mpc_q_delta_limit": 0.0,
        "stance_mpc_q_delta_scale": 1.0,
        "tau_ff_scale": 0.05,
        "swing_tau_ff_scale": 0.20,
        "swing_tau_limit": 2.0,
        "x_vel": 0.02,
        "y_vel": 0.0,
        "yaw_rate": 0.0,
        "mpc_r": 3.0e-3,
        "mpc_force_xy_scale": 0.3,
        "use_imu_base_state": True,
        "imu_rp_zero_on_start": True,
        "imu_rp_zero_seconds": 0.5,
        "imu_gyro_deadband": 0.005,
        "fix_gateway_gyro_order": True,
        "abort_on_large_error": True,
        "abort_qerr": 0.80,
        "abort_swing_qerr": 1.20,
        "abort_tilt": 0.35,
        "return_pose_on_exit": "down",
        "return_ramp_seconds": 5.0,
        "disable_on_exit": True,
        "allow_large_gains": True,
    },
}


def load_ex36_config_defaults(path):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required for --config") from exc

    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"--config must contain a YAML mapping: {config_path}")

    config = raw.get("args", raw)
    if not isinstance(config, dict):
        raise RuntimeError(f"--config args must be a YAML mapping: {config_path}")

    defaults = {}
    for key, value in config.items():
        defaults[str(key).strip().replace("-", "_")] = value

    valid_keys = {
        action.dest
        for action in build_arg_parser()._actions
        if action.dest and action.dest != argparse.SUPPRESS
    }
    unknown = sorted(set(defaults) - valid_keys)
    if unknown:
        raise RuntimeError(
            f"unknown --config keys in {config_path}: {', '.join(unknown)}"
        )
    return defaults


def defaults_from_argv(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test-profile", choices=("custom", "trot-final", "all-stance"), default=None)
    parser.add_argument("--config", default=None)
    args, _unknown = parser.parse_known_args(argv)
    config_defaults = load_ex36_config_defaults(args.config) if args.config else {}
    profile_name = args.test_profile or config_defaults.get("test_profile", "custom")
    defaults = dict(PROFILE_DEFAULTS.get(profile_name, {}))
    defaults.update(config_defaults)
    defaults["test_profile"] = profile_name
    if args.config:
        defaults["config"] = args.config
    return defaults


def validate_args(args):
    if not args.robot_standing_supported:
        raise RuntimeError("EX36 requires --robot-standing-supported")
    if not args.i_accept_risk:
        raise RuntimeError("EX36 sends real motor commands; pass --i-accept-risk")
    if args.duration <= 0.0 or (args.duration > 12.0 and not args.allow_long_duration):
        raise RuntimeError("--duration must be in (0, 12] unless --allow-long-duration")
    leg_control_hz = args.cmd_hz if args.leg_control_hz is None else args.leg_control_hz
    if args.cmd_hz <= 0.0 or args.state_hz <= 0.0 or leg_control_hz <= 0.0 or args.mpc_hz <= 0.0 or args.print_hz < 0.0:
        raise RuntimeError("--cmd-hz/--state-hz/--leg-control-hz/--mpc-hz must be positive and --print-hz non-negative")
    if args.control_sleep_s < 0.0:
        raise RuntimeError("--control-sleep-s must be non-negative")
    if args.wait_lowstate_s <= 0.0 or args.feedback_timeout_s <= 0.0:
        raise RuntimeError("--wait-lowstate-s and --feedback-timeout-s must be positive")
    if min(args.kp, args.kd) < 0.0:
        raise RuntimeError("gains must be non-negative")
    if args.abort_qerr <= 0.0 or (args.abort_swing_qerr is not None and args.abort_swing_qerr <= 0.0):
        raise RuntimeError("--abort-qerr and --abort-swing-qerr must be positive")
    final_kp = args.kp if args.final_kp is None else args.final_kp
    final_kd = args.kd if args.final_kd is None else args.final_kd
    swing_kp = args.kp if args.swing_kp is None else args.swing_kp
    swing_kd = args.kd if args.swing_kd is None else args.swing_kd
    if min(final_kp, final_kd, swing_kp, swing_kd) < 0.0:
        raise RuntimeError("--final-kp/--final-kd/--swing-kp/--swing-kd must be non-negative")
    if (
        max(args.kp, args.kd, final_kp, final_kd, swing_kp, swing_kd) > 20.0
        or max(args.kd, final_kd, swing_kd) > 1.0
    ) and not args.allow_large_gains:
        raise RuntimeError("large kp/kd requires --allow-large-gains")
    if args.horizon_seconds <= 0.0 or args.horizon_segments <= 0:
        raise RuntimeError("--horizon-seconds and --horizon-segments must be positive")
    if args.gait != "all-stance" and not args.allow_swing_control:
        raise RuntimeError("--gait trot requires --allow-swing-control")
    if args.gait_frequency_hz <= 0.0 or not (0.0 < args.gait_duty <= 1.0):
        raise RuntimeError("--gait-frequency-hz must be positive and --gait-duty in (0, 1]")
    if args.gait_warmup_s < 0.0:
        raise RuntimeError("--gait-warmup-s must be non-negative")
    if args.swing_height < 0.0:
        raise RuntimeError("--swing-height must be non-negative")
    if args.swing_q_target_delta_limit < 0.0 or args.stance_mpc_q_delta_limit < 0.0:
        raise RuntimeError("--swing-q-target-delta-limit and --stance-mpc-q-delta-limit must be non-negative")
    if args.tau_ff_scale < 0.0 or args.swing_tau_ff_scale < 0.0:
        raise RuntimeError("--tau-ff-scale and --swing-tau-ff-scale must be non-negative")
    if args.swing_tau_limit < 0.0:
        raise RuntimeError("--swing-tau-limit must be non-negative")
    if args.lift_leg is not None and not args.allow_single_leg_lift:
        raise RuntimeError("--lift-leg requires --allow-single-leg-lift")
    for name in ("lift_start_s", "lift_ramp_s", "lift_hold_s"):
        value = getattr(args, name)
        if value is not None and value < 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be non-negative")
    if args.base_height_clip < 0.0 or args.max_base_xy_vel < 0.0 or args.max_base_z_vel < 0.0:
        raise RuntimeError("state-estimator limits must be non-negative")
    if args.mpc_r <= 0.0:
        raise RuntimeError("--mpc-r must be positive")
    if abs(args.mpc_force_xy_scale) > 1.0:
        raise RuntimeError("--mpc-force-xy-scale must be in [-1, 1]")
    if args.return_pose_on_exit is not None and args.return_pose_on_exit not in ("stand", "down"):
        raise RuntimeError("--return-pose-on-exit currently supports stand/down")


def build_gait(args):
    if args.gait == "all-stance":
        return AllStanceGait(args.horizon_seconds)
    from convex_mpc.gait import Gait

    params = inspect.signature(Gait).parameters
    if "swing_height" in params:
        gait = Gait(args.gait_frequency_hz, args.gait_duty, swing_height=args.swing_height)
    else:
        gait = Gait(args.gait_frequency_hz, args.gait_duty)
        gait.swing_height = float(args.swing_height)
    gait = StepTrotGait(gait)
    if args.gait_warmup_s > 0.0:
        gait = WarmupGait(gait, args.gait_warmup_s)
    return gait


def open_log_writer(args):
    if args.no_log:
        return None, None, None
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{args.log_prefix}_{stamp}.csv"
    f = path.open("w", newline="", encoding="utf-8")
    fields = [
        "elapsed_s",
        "phase",
        "joint",
        "q_model",
        "q_target",
        "q_err",
        "tau_model",
        "tau_raw",
        "tau_clip",
        "tau_cmd",
        "tau_est",
        "alpha",
        "clipped_count",
        "mpc_solve_ms",
        "handover_alpha",
        "kp_cmd",
        "kd_cmd",
        "kp_lowcmd",
        "kd_lowcmd",
        "kp_joint",
        "kd_joint",
        "leg_role",
        "tau_limit_cmd",
        "base_source",
        "height_source",
        "vel_source",
        "base_x",
        "base_y",
        "base_z",
        "base_roll",
        "base_pitch",
        "base_yaw",
        "base_vx",
        "base_vy",
        "base_vz",
        "base_gx",
        "base_gy",
        "base_gz",
        "contact_mask",
        "lift_leg",
        "lift_alpha",
        "mpc_fx",
        "mpc_fy",
        "mpc_fz",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    print(f"log_csv: {path}")
    return path, f, writer


def write_log_rows(
    writer,
    bridge,
    cache,
    q_target_model,
    tau_model,
    tau_raw,
    tau_clip,
    tau_cmd,
    elapsed_s,
    phase,
    mpc,
    alpha,
    clipped_count,
    handover_alpha,
    kp_cmd,
    kd_cmd,
    kp_lowcmd,
    kd_lowcmd,
    tau_limit_cmd,
    state,
    contact_mask,
    lift_alpha,
    mpc_force_now,
    args,
    dds_model_offset,
):
    if writer is None:
        return
    q_model = feedback_to_model_dict(cache, bridge, args, dds_model_offset)
    tau_model_by_joint = {joint: float(value) for joint, value in zip(MPC_JOINT_ORDER, tau_model)}
    kp_by_joint, kd_by_joint = gain_dict_for_contact(args, bridge, contact_mask, kp_lowcmd, kd_lowcmd)
    for i, joint in enumerate(bridge.order):
        leg = joint.split("_", 1)[0]
        in_swing = args.gait != "all-stance" and int(contact_mask[LEG_INDEX[leg]]) == 0
        force = mpc_force_now[LEG_SLICE[leg]]
        try:
            tau_limit_value = float(tau_limit_cmd[i])
        except (TypeError, IndexError):
            tau_limit_value = float(tau_limit_cmd)
        writer.writerow(
            {
                "elapsed_s": f"{elapsed_s:.6f}",
                "phase": phase,
                "joint": joint,
                "q_model": f"{q_model[joint]:.9f}",
                "q_target": f"{q_target_model[joint]:.9f}",
                "q_err": f"{q_model[joint] - q_target_model[joint]:.9f}",
                "tau_model": f"{tau_model_by_joint[joint]:.9f}",
                "tau_raw": f"{float(tau_raw[i]):.9f}",
                "tau_clip": f"{float(tau_clip[i]):.9f}",
                "tau_cmd": f"{float(tau_cmd[i]):.9f}",
                "tau_est": f"{float(cache[joint]['tau_est']):.9f}",
                "alpha": f"{alpha:.6f}",
                "clipped_count": int(clipped_count),
                "mpc_solve_ms": f"{getattr(mpc, 'solve_time', 0.0):.6f}",
                "handover_alpha": f"{handover_alpha:.6f}",
                "kp_cmd": f"{kp_cmd:.6f}",
                "kd_cmd": f"{kd_cmd:.6f}",
                "kp_lowcmd": f"{kp_lowcmd:.6f}",
                "kd_lowcmd": f"{kd_lowcmd:.6f}",
                "kp_joint": f"{kp_by_joint[joint]:.6f}",
                "kd_joint": f"{kd_by_joint[joint]:.6f}",
                "leg_role": "swing" if in_swing else "stance",
                "tau_limit_cmd": f"{tau_limit_value:.6f}",
                "base_source": state["source"],
                "height_source": state["height_source"],
                "vel_source": state["vel_source"],
                "base_x": f"{state['x']:.9f}",
                "base_y": f"{state['y']:.9f}",
                "base_z": f"{state['z']:.9f}",
                "base_roll": f"{state['roll']:.9f}",
                "base_pitch": f"{state['pitch']:.9f}",
                "base_yaw": f"{state['yaw']:.9f}",
                "base_vx": f"{state['vx']:.9f}",
                "base_vy": f"{state['vy']:.9f}",
                "base_vz": f"{state['vz']:.9f}",
                "base_gx": f"{state['gyro'][0]:.9f}",
                "base_gy": f"{state['gyro'][1]:.9f}",
                "base_gz": f"{state['gyro'][2]:.9f}",
                "contact_mask": mask_text(contact_mask),
                "lift_leg": args.lift_leg or "",
                "lift_alpha": f"{lift_alpha:.6f}",
                "mpc_fx": f"{float(force[0]):.9f}",
                "mpc_fy": f"{float(force[1]):.9f}",
                "mpc_fz": f"{float(force[2]):.9f}",
            }
        )


def model_tau_to_dds_order(bridge, args, tau_model):
    if args.dds_model_space:
        # The C++ gateway maps q/dq from model-space to motor-space, but tau is
        # currently passed through 1:1. Convert model tau to the motor-side
        # sign/scale here so FL/RL thigh/calf and calf gear match the q affine.
        return bridge.motor_torque_commands_from_mpc(tau_model)

    # Raw motor-space mode bypasses the gateway model mapping, so keep the old
    # affine torque conversion for that explicit debug path.
    return bridge.motor_torque_commands_from_mpc(tau_model)


def compute_joint_bias(vbot, leg):
    import numpy as np

    g, c, _m = vbot.compute_dynamcis_terms()
    bias = np.asarray(c @ vbot.current_config.get_dq() + g, dtype=float).reshape(-1)
    return bias[vbot.get_leg_joint_vcols(leg)]


def solve_leg_ik_body_near(vbot, leg, foot_des_body, q_seed, max_iters=12):
    import numpy as np

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


def swing_target_from_leg_output(vbot, leg, out, q_model, args):
    import numpy as np

    foot_des_world = np.asarray(out.pos_des, dtype=float).reshape(3)
    foot_vel_des_world = np.asarray(out.vel_des, dtype=float).reshape(3)
    foot_des_body = vbot.R_world_to_body @ (foot_des_world - vbot.current_config.base_pos)
    foot_vel_des_body = vbot.R_world_to_body @ (foot_vel_des_world - vbot.current_config.base_vel)
    q_seed = [q_model[joint] for joint in LEG_JOINTS[leg]]
    q_des = solve_leg_ik_body_near(vbot, leg, foot_des_body, q_seed)
    qd_des = vbot.calc_leg_qd_body(leg, q_des, foot_vel_des_body)
    return q_des, qd_des


def apply_stance_mpc_q_delta(args, q_target, q_stand_model, contact_mask, tau_model, kp_stance):
    import numpy as np

    if args.stance_mpc_q_delta_limit <= 0.0 or kp_stance <= 1.0e-9:
        return q_target

    contact_mask = np.asarray(contact_mask, dtype=int).reshape(4)
    for leg, leg_slice in LEG_SLICE.items():
        if int(contact_mask[LEG_INDEX[leg]]) != 1:
            continue
        tau_leg = np.asarray(tau_model[leg_slice], dtype=float).reshape(3)
        dq_leg = args.stance_mpc_q_delta_scale * tau_leg / float(kp_stance)
        dq_leg = np.clip(
            dq_leg,
            -float(args.stance_mpc_q_delta_limit),
            float(args.stance_mpc_q_delta_limit),
        )
        for joint, delta in zip(LEG_JOINTS[leg], dq_leg):
            q_target[joint] = float(q_stand_model[joint] + delta)
    return q_target


def update_targets_from_swing_outputs(
    args,
    vbot,
    bridge,
    q_stand_model,
    q_model,
    contact_mask,
    leg_outputs,
    tau_model=None,
    kp_stance=0.0,
):
    import numpy as np

    q_target = dict(q_stand_model)
    qd_target = zero_joint_dict(bridge.order)
    if tau_model is not None:
        q_target = apply_stance_mpc_q_delta(args, q_target, q_stand_model, contact_mask, tau_model, kp_stance)
    if args.gait == "all-stance":
        return q_target, qd_target

    contact_mask = np.asarray(contact_mask, dtype=int).reshape(4)
    for leg, out in leg_outputs.items():
        if int(contact_mask[LEG_INDEX[leg]]) != 0:
            continue
        q_leg, qd_leg = swing_target_from_leg_output(vbot, leg, out, q_model, args)
        for joint, q_value, qd_value in zip(LEG_JOINTS[leg], q_leg, qd_leg):
            if args.swing_q_target_delta_limit > 0.0:
                q_value = np.clip(
                    float(q_value),
                    q_stand_model[joint] - args.swing_q_target_delta_limit,
                    q_stand_model[joint] + args.swing_q_target_delta_limit,
                )
            q_target[joint] = float(q_value)
            qd_target[joint] = float(qd_value)
    return q_target, qd_target


def gain_dict_for_contact(args, bridge, contact_mask, kp_stance, kd_stance):
    import numpy as np

    contact_mask = np.asarray(contact_mask, dtype=int).reshape(4)
    swing_kp = kp_stance if args.swing_kp is None else args.swing_kp
    swing_kd = kd_stance if args.swing_kd is None else args.swing_kd
    kp_by_joint = {}
    kd_by_joint = {}
    for joint in bridge.order:
        leg = joint.split("_", 1)[0]
        in_swing = int(contact_mask[LEG_INDEX[leg]]) == 0 and args.gait != "all-stance"
        kp_by_joint[joint] = float(swing_kp if in_swing else kp_stance)
        kd_by_joint[joint] = float(swing_kd if in_swing else kd_stance)
    return kp_by_joint, kd_by_joint


def tau_scale_vector_for_contact(args, bridge, contact_mask):
    import numpy as np

    contact_mask = np.asarray(contact_mask, dtype=int).reshape(4)
    scale = np.zeros(12, dtype=float)
    for i, joint in enumerate(bridge.order):
        leg = joint.split("_", 1)[0]
        in_swing = args.gait != "all-stance" and int(contact_mask[LEG_INDEX[leg]]) == 0
        scale[i] = float(args.swing_tau_ff_scale if in_swing else args.tau_ff_scale)
    return scale


def torque_limit_vector_for_contact(args, base_limit, bridge, contact_mask):
    import numpy as np

    limit = np.asarray(base_limit, dtype=float).reshape(12).copy()
    contact_mask = np.asarray(contact_mask, dtype=int).reshape(4)
    if args.gait == "all-stance" or args.swing_tau_limit <= 0.0:
        return limit
    for i, joint in enumerate(bridge.order):
        leg = joint.split("_", 1)[0]
        if int(contact_mask[LEG_INDEX[leg]]) == 0:
            limit[i] = min(float(limit[i]), float(args.swing_tau_limit))
    return limit


def filter_mpc_force_now(args, mpc_force_now):
    import numpy as np

    force = np.asarray(mpc_force_now, dtype=float).copy()
    xy_scale = float(args.mpc_force_xy_scale)
    for leg_slice in LEG_SLICE.values():
        force[leg_slice.start : leg_slice.start + 2] *= xy_scale
    return force


def compute_tau_from_force(vbot, gait, leg_controller, mpc_force_now, gait_time_s, args):
    import numpy as np

    tau_model = np.zeros(12, dtype=float)
    contact_mask = gait.compute_current_mask(gait_time_s).reshape(4)
    leg_outputs = {}
    for leg, leg_slice in LEG_SLICE.items():
        in_swing = int(contact_mask[LEG_INDEX[leg]]) == 0 and args.gait != "all-stance"
        if in_swing and not args.allow_swing_control:
            if args.stance_bias_comp:
                tau_model[leg_slice] = compute_joint_bias(vbot, leg)
            continue
        out = leg_controller.compute_leg_torque(
            leg,
            vbot,
            gait,
            mpc_force_now[leg_slice],
            gait_time_s,
        )
        tau_leg = np.asarray(out.tau, dtype=float).reshape(3)
        if args.stance_bias_comp and int(contact_mask[LEG_INDEX[leg]]) == 1:
            tau_leg = tau_leg + compute_joint_bias(vbot, leg)
        tau_model[leg_slice] = tau_leg
        # Stance legs use MPC contact force. Swing legs use the Cartesian
        # swing-PD torque from LegController, then get separately scaled and
        # clipped before lowcmd publication.
        leg_outputs[leg] = out
    return tau_model, contact_mask, leg_outputs


def print_status(
    bridge,
    cache,
    q_target_model,
    tau_model,
    tau_cmd,
    elapsed_s,
    mpc,
    alpha,
    clipped_count,
    handover_alpha,
    kp_cmd,
    kd_cmd,
    kp_lowcmd,
    kd_lowcmd,
    tau_limit_cmd,
    state,
    contact_mask,
    lift_alpha,
    args,
    dds_model_offset,
):
    import numpy as np

    q_model = feedback_to_model_dict(cache, bridge, args, dds_model_offset)
    max_err_joint = max(bridge.order, key=lambda joint: abs(q_model[joint] - q_target_model[joint]))
    max_cmd_idx = int(np.argmax(np.abs(tau_cmd)))
    try:
        tau_limit_text = f"{float(tau_limit_cmd[max_cmd_idx]):.1f}"
    except (TypeError, IndexError):
        tau_limit_text = f"{float(tau_limit_cmd):.1f}"
    print(
        f"\nt={elapsed_s:.2f}s mode={args.gait} alpha={alpha:.2f} handover={handover_alpha:.2f} "
        f"kp_cmd={kp_cmd:.1f}/{kp_lowcmd:.1f} kd_cmd={kd_cmd:.2f}/{kd_lowcmd:.2f} "
        f"tau_lim={tau_limit_text} clipped={clipped_count}/12 "
        f"mask={mask_text(contact_mask)} lift={args.lift_leg or '-'}:{lift_alpha:.2f} "
        f"mpc={getattr(mpc, 'solve_time', 0.0):.2f}ms "
        f"base={state['source']}/{state['height_source']}/{state['vel_source']} "
        f"z={state['z']:.3f} rpy=[{state['roll']:+.4f},{state['pitch']:+.4f},{state['yaw']:+.4f}] "
        f"v=[{state['vx']:+.3f},{state['vy']:+.3f},{state['vz']:+.3f}] "
        f"gyro=[{state['gyro'][0]:+.4f},{state['gyro'][1]:+.4f},{state['gyro'][2]:+.4f}] "
        f"max_q_err={max_err_joint}:{q_model[max_err_joint] - q_target_model[max_err_joint]:+.4f} "
        f"max_tau_cmd={bridge.order[max_cmd_idx]}:{float(tau_cmd[max_cmd_idx]):+.3f}"
    )


def main() -> int:
    args = build_arg_parser(defaults_from_argv()).parse_args()
    validate_args(args)

    import numpy as np

    from convex_mpc import centroidal_mpc as centroidal_mpc_module
    from convex_mpc.com_trajectory import ComTraj
    from convex_mpc.leg_controller import LegController
    from convex_mpc.vbot_robot_data import PinVBotModel

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    except Exception as exc:
        print("ERROR: unitree_sdk2py is required in this Python environment.")
        print(f"Import failure: {exc}")
        return 2

    np.set_printoptions(precision=4, suppress=True)
    mpc_q = centroidal_mpc_module.COST_MATRIX_Q.copy()
    mpc_q[5, 5] = args.yaw_weight
    mpc_q[11, 11] = args.yaw_rate_weight
    centroidal_mpc_module.COST_MATRIX_Q = mpc_q
    centroidal_mpc_module.COST_MATRIX_R = np.diag([args.mpc_r] * 12)
    CentroidalMPC = centroidal_mpc_module.CentroidalMPC

    bridge = VBotRealJointAffine.from_yaml(args.affine)
    validate_joint_order_compatibility(bridge)
    q_stand_model = load_model_pose(args.model_poses, args.target_pose, bridge.order)
    q_target_model = dict(q_stand_model)
    qd_target_model = zero_joint_dict(bridge.order)
    q_exit_model = q_target_model

    latest = {"msg": None, "t": 0.0}
    stop = install_stop_handlers()

    def on_lowstate(msg):
        latest["msg"] = msg
        latest["t"] = time.monotonic()

    print("EX36 DDS trot: stance MPC tau + swing Cartesian tau")
    leg_control_hz = args.cmd_hz if args.leg_control_hz is None else args.leg_control_hz
    print(
        f"network={args.network} gait={args.gait} cmd_hz={args.cmd_hz:.1f} "
        f"leg_hz={leg_control_hz:.1f} mpc_hz={args.mpc_hz:.1f}"
    )
    print(f"state: imu={args.use_imu_base_state} height={args.base_height_mode} vel={args.base_vel_mode}")
    print(
        "torque injection: "
        f"tau_ff_scale={args.tau_ff_scale:.2f} "
        f"swing_tau_ff_scale={args.swing_tau_ff_scale:.2f} "
        f"swing_tau_limit={args.swing_tau_limit:.2f} "
        f"force_xy_scale={args.mpc_force_xy_scale:.2f} "
        f"stance_q_delta_limit={args.stance_mpc_q_delta_limit:.3f} "
        f"swing_q_delta_limit={args.swing_q_target_delta_limit:.3f}"
    )
    print(
        "mixed lowcmd: q/dq/kp/kd joint impedance + MPC tau_ff, "
        f"stance_bias_comp={args.stance_bias_comp} "
        f"motor tau limits={MOTOR_TORQUE_LIMIT_BY_JOINT_TYPE} mpc_r={args.mpc_r:.3e}"
    )
    if args.dds_model_space:
        print("mapping: DDS q/dq are gateway model-space; EX36 converts model tau to motor-side sign/scale")
    else:
        print("mapping: raw motor-space debug path; EX36 applies affine q/tau conversion before LowCmd")
    if args.gait != "all-stance":
        print("WARNING: swing control is enabled; keep the robot supported.")
    if args.lift_leg:
        lift_start = args.lift_start_s if args.lift_start_s is not None else 2.0
        lift_ramp = args.lift_ramp_s if args.lift_ramp_s is not None else 2.0
        lift_hold = args.lift_hold_s if args.lift_hold_s is not None else 2.0
        print(
            f"Single-leg lift schedule: {args.lift_leg} "
            f"MPC 0-{lift_start:.1f}s stand, "
            f"{lift_start:.1f}-{lift_start + lift_ramp:.1f}s lift, "
            f"{lift_start + lift_ramp:.1f}-{lift_start + lift_ramp + lift_hold:.1f}s hold, "
            f"{lift_start + lift_ramp + lift_hold:.1f}-{lift_start + 2.0 * lift_ramp + lift_hold:.1f}s return"
        )

    ChannelFactoryInitialize(0, args.network)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_lowstate, 1)
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = make_crc()
    log_path, log_file, log_writer = open_log_writer(args)

    entered_control = False
    dds_model_offset = zero_joint_dict(bridge.order)
    imu_zero_rp = (0.0, 0.0)
    overlay_lowcmd_times = []
    overlay_loop_count = 0
    overlay_elapsed_s = 0.0
    max_observed_qerr = 0.0
    max_observed_qerr_joint = ""
    max_observed_tilt = 0.0
    max_observed_lift_alpha = 0.0
    last_contact_mask_text = "1111"
    publish_count = 0
    publish_time_sum = 0.0
    publish_time_max = 0.0
    state_update_count = 0
    state_time_sum = 0.0
    state_time_max = 0.0
    leg_update_count = 0
    leg_time_sum = 0.0
    leg_time_max = 0.0
    mpc_update_count = 0
    mpc_time_sum = 0.0
    mpc_time_max = 0.0
    print_update_count = 0
    print_time_sum = 0.0
    print_time_max = 0.0
    exit_reason = "not_started"
    q_target_motor = None
    qd_target_motor = None
    lowcmd_publisher = None
    cmd_period = 1.0 / args.cmd_hz

    try:
        wait_deadline = time.monotonic() + args.wait_lowstate_s
        while latest["msg"] is None and time.monotonic() < wait_deadline and not stop["requested"]:
            time.sleep(0.02)
        if latest["msg"] is None:
            raise RuntimeError("no rt/lowstate received; start dds_to_serial_gateway first")

        print("Waiting for non-lost feedback...")
        first_cache = None
        poll_cmd = make_lowcmd(1, bridge.order, None, None, 0.0, 0.0)
        next_cmd = time.monotonic()
        wait_deadline = time.monotonic() + args.wait_lowstate_s
        while time.monotonic() < wait_deadline and not stop["requested"]:
            now = time.monotonic()
            if now >= next_cmd:
                publish_lowcmd(pub, poll_cmd, crc)
                next_cmd = advance_periodic_deadline(next_cmd, cmd_period, now)
            cache = read_lowstate_motor_feedback(latest["msg"], bridge.order)
            if not lost_joints(cache, bridge.order):
                first_cache = cache
                break
            time.sleep(0.002)
        if first_cache is None:
            raise RuntimeError("motor feedback is still lost")

        imu_zero_rp = estimate_imu_zero_rp(latest, args)
        if args.dds_model_space:
            dds_model_offset, _anchored = compute_dds_model_offset_if_requested(bridge, first_cache, args)
        else:
            raise RuntimeError("EX36 currently expects --dds-model-space gateway mapping")

        q_target_motor = model_command_to_dds_dict(q_target_model, bridge, args, dds_model_offset)
        qd_target_motor = model_velocity_command_to_dds_dict(qd_target_model, bridge, args)
        q_start_model = feedback_to_model_dict(first_cache, bridge, args, dds_model_offset)

        if args.startup_ramp_seconds > 0.0:
            entered_control = True
            print(f"Ramping current mapped pose -> {args.target_pose} for {args.startup_ramp_seconds:.2f}s...")
            ramp_start = time.monotonic()
            ramp_deadline = ramp_start + args.startup_ramp_seconds
            next_cmd = ramp_start
            while time.monotonic() < ramp_deadline and not stop["requested"]:
                now = time.monotonic()
                if now - latest["t"] > args.feedback_timeout_s:
                    raise RuntimeError(f"rt/lowstate timeout > {args.feedback_timeout_s:.3f}s")
                alpha_ramp = smoothstep((now - ramp_start) / max(args.startup_ramp_seconds, 1.0e-9))
                q_cmd_model = dict_interpolate(q_start_model, q_target_model, alpha_ramp)
                q_cmd_motor = model_command_to_dds_dict(q_cmd_model, bridge, args, dds_model_offset)
                if now >= next_cmd:
                    cmd = make_lowcmd(1, bridge.order, q_cmd_motor, None, args.kp, args.kd)
                    publish_lowcmd(pub, cmd, crc)
                    next_cmd = advance_periodic_deadline(next_cmd, cmd_period, now)
                time.sleep(0.001)

        if args.prehold_seconds > 0.0:
            entered_control = True
            print(f"Publishing stand PD with zero MPC torque for {args.prehold_seconds:.2f}s...")
            hold_cmd = make_lowcmd(
                1,
                bridge.order,
                q_target_motor,
                None,
                args.kp,
                args.kd,
                qd_motor_by_joint=qd_target_motor,
            )
            publish_for(
                pub,
                hold_cmd,
                crc,
                args.cmd_hz,
                args.prehold_seconds,
                stop=stop,
                latest=latest,
                feedback_timeout_s=args.feedback_timeout_s,
            )

        vbot = PinVBotModel()
        estimator = RealStateEstimator(args)
        gait = build_gait(args)
        leg_controller = LegController()
        traj = ComTraj(vbot)
        mpc_dt = gait.gait_period / args.horizon_segments
        cache = read_lowstate_motor_feedback(latest["msg"], bridge.order)
        base = read_base_state(latest["msg"], args, imu_zero_rp)
        contact_mask = gait.compute_current_mask(0.0).reshape(4)
        state = estimator.update(vbot, bridge, cache, args, dds_model_offset, base, contact_mask, time.monotonic())
        generate_traj_compat(
            traj,
            vbot,
            gait,
            0.0,
            args.x_vel,
            args.y_vel,
            args.base_height,
            args.yaw_rate,
            mpc_dt,
            args.base_pitch,
        )
        mpc = CentroidalMPC(vbot, traj)
        u_opt = np.zeros((12, traj.N), dtype=float)
        mpc_force_now = np.zeros(12, dtype=float)
        tau_model = np.zeros(12, dtype=float)
        tau_raw = np.zeros(12, dtype=float)
        tau_clip = np.zeros(12, dtype=float)
        tau_cmd = np.zeros(12, dtype=float)
        clipped_count = 0
        alpha = 0.0
        lift_alpha = 0.0
        handover_alpha = 0.0
        kp_cmd = args.kp
        kd_cmd = args.kd
        kp_lowcmd = args.kp
        kd_lowcmd = args.kd
        tau_limit_cmd = motor_torque_limit_vector(bridge.order)
        leg_outputs = {}
        kp_gain_cmd, kd_gain_cmd = gain_dict_for_contact(args, bridge, contact_mask, kp_lowcmd, kd_lowcmd)
        max_qerr = 0.0
        max_qerr_joint = bridge.order[0]
        max_tilt = max(abs(state["roll"]), abs(state["pitch"]))
        qerr_by_joint = {joint: 0.0 for joint in bridge.order}
        publish_count = 0
        publish_time_sum = 0.0
        publish_time_max = 0.0
        state_update_count = 0
        state_time_sum = 0.0
        state_time_max = 0.0
        leg_update_count = 0
        leg_time_sum = 0.0
        leg_time_max = 0.0
        mpc_update_count = 0
        mpc_time_sum = 0.0
        mpc_time_max = 0.0
        print_update_count = 0
        print_time_sum = 0.0
        print_time_max = 0.0

        start_t = time.monotonic()
        deadline = start_t + args.duration
        next_state = start_t
        next_leg = start_t
        next_mpc = start_t
        next_print = start_t
        next_safety_timeout_check = start_t
        leg_period = 1.0 / leg_control_hz
        mpc_period = 1.0 / args.mpc_hz
        state_period = 1.0 / args.state_hz
        print_period = math.inf if args.print_hz <= 0.0 else 1.0 / args.print_hz
        lowcmd_publisher = PeriodicLowcmdPublisher(
            pub,
            crc,
            bridge.order,
            args.cmd_hz,
            stop,
            latest=latest,
            feedback_timeout_s=args.feedback_timeout_s,
        )
        lowcmd_publisher.update(
            1,
            q_target_motor,
            tau_cmd,
            kp_gain_cmd,
            kd_gain_cmd,
            qd_motor_by_joint=qd_target_motor,
        )
        lowcmd_publisher.start()

        while time.monotonic() < deadline and not stop["requested"]:
            entered_control = True
            now = time.monotonic()
            elapsed = now - start_t
            gait_time = elapsed
            overlay_elapsed_s = elapsed
            overlay_loop_count += 1
            max_observed_lift_alpha = max(max_observed_lift_alpha, float(lift_alpha))
            last_contact_mask_text = mask_text(contact_mask)
            if now - latest["t"] > args.feedback_timeout_s:
                exit_reason = f"rt_lowstate_timeout>{args.feedback_timeout_s:.3f}s"
                raise RuntimeError(f"rt/lowstate timeout > {args.feedback_timeout_s:.3f}s")
            publisher_error = lowcmd_publisher.get_error()
            if publisher_error:
                exit_reason = f"lowcmd_publisher:{publisher_error}"
                raise RuntimeError(f"lowcmd publisher failed: {publisher_error}")

            handover_alpha, kp_cmd, kd_cmd = overlay_schedule(args, elapsed)
            kp_lowcmd = kp_cmd
            kd_lowcmd = kd_cmd
            kp_gain_cmd, kd_gain_cmd = gain_dict_for_contact(args, bridge, contact_mask, kp_lowcmd, kd_lowcmd)
            lowcmd_publisher.update(
                1,
                q_target_motor,
                tau_cmd,
                kp_gain_cmd,
                kd_gain_cmd,
                qd_motor_by_joint=qd_target_motor,
            )

            now = time.monotonic()
            if now >= next_safety_timeout_check:
                if now - latest["t"] > args.feedback_timeout_s:
                    exit_reason = f"rt_lowstate_timeout>{args.feedback_timeout_s:.3f}s"
                    raise RuntimeError(f"rt/lowstate timeout > {args.feedback_timeout_s:.3f}s")
                next_safety_timeout_check = advance_periodic_deadline(
                    next_safety_timeout_check,
                    min(state_period, 0.02),
                    now,
                )

            need_state = now >= next_state or now >= next_leg or now >= next_mpc or now >= next_print
            if need_state:
                state_t0 = time.monotonic()
                cache = read_lowstate_motor_feedback(latest["msg"], bridge.order)
                lost = lost_joints(cache, bridge.order)
                if lost:
                    exit_reason = "motor_feedback_lost"
                    raise RuntimeError(f"motor feedback lost: {', '.join(lost)}")

                base = read_base_state(latest["msg"], args, imu_zero_rp)
                if args.lift_leg is not None:
                    q_target_model, contact_mask, lift_alpha = lift_target_and_contact(args, q_stand_model, elapsed)
                else:
                    contact_mask = gait.compute_current_mask(gait_time).reshape(4)
                    lift_alpha = 0.0
                    q_model_for_targets = feedback_to_model_dict(cache, bridge, args, dds_model_offset)
                    q_target_model, qd_target_model = update_targets_from_swing_outputs(
                        args,
                        vbot,
                        bridge,
                        q_stand_model,
                        q_model_for_targets,
                        contact_mask,
                        leg_outputs,
                        tau_model=tau_model,
                        kp_stance=kp_lowcmd,
                    )
                max_observed_lift_alpha = max(max_observed_lift_alpha, float(lift_alpha))
                last_contact_mask_text = mask_text(contact_mask)
                q_target_motor = model_command_to_dds_dict(q_target_model, bridge, args, dds_model_offset)
                qd_target_motor = model_velocity_command_to_dds_dict(qd_target_model, bridge, args)
                if hasattr(gait, "contact_mask"):
                    gait.contact_mask = contact_mask
                kp_gain_cmd, kd_gain_cmd = gain_dict_for_contact(args, bridge, contact_mask, kp_lowcmd, kd_lowcmd)
                lowcmd_publisher.update(
                    1,
                    q_target_motor,
                    tau_cmd,
                    kp_gain_cmd,
                    kd_gain_cmd,
                    qd_motor_by_joint=qd_target_motor,
                )
                q_model_for_safety = feedback_to_model_dict(cache, bridge, args, dds_model_offset)
                qerr_by_joint = {
                    joint: q_model_for_safety[joint] - q_target_model[joint]
                    for joint in bridge.order
                }
                max_qerr_joint = max(bridge.order, key=lambda joint: abs(qerr_by_joint[joint]))
                max_qerr = abs(qerr_by_joint[max_qerr_joint])
                swing_qerr_limit = args.abort_qerr if args.abort_swing_qerr is None else args.abort_swing_qerr

                def qerr_limit_for_joint(joint):
                    leg = joint.split("_", 1)[0]
                    in_swing = args.gait != "all-stance" and int(contact_mask[LEG_INDEX[leg]]) == 0
                    return float(swing_qerr_limit if in_swing else args.abort_qerr)

                safety_qerr_joint = max(
                    bridge.order,
                    key=lambda joint: abs(qerr_by_joint[joint]) / qerr_limit_for_joint(joint),
                )
                safety_qerr = abs(qerr_by_joint[safety_qerr_joint])
                safety_qerr_limit = qerr_limit_for_joint(safety_qerr_joint)
                max_tilt = max(abs(base["roll"]), abs(base["pitch"]))
                if max_qerr > max_observed_qerr:
                    max_observed_qerr = max_qerr
                    max_observed_qerr_joint = max_qerr_joint
                max_observed_tilt = max(max_observed_tilt, max_tilt)
                if args.abort_on_large_error and (safety_qerr > safety_qerr_limit or max_tilt > args.abort_tilt):
                    exit_reason = "abort_large_error"
                    raise RuntimeError(
                        "abort EX36 safety limit: "
                        f"max_qerr={max_qerr:.4f} joint={max_qerr_joint} "
                        f"safety_qerr={safety_qerr:.4f}/{safety_qerr_limit:.4f} "
                        f"joint={safety_qerr_joint} qerr={qerr_by_joint[safety_qerr_joint]:+.4f} "
                        f"max_tilt={max_tilt:.4f}/{args.abort_tilt:.4f}"
                    )
                state_dt = time.monotonic() - state_t0
                state_time_sum += state_dt
                state_time_max = max(state_time_max, state_dt)
                state_update_count += 1
                if now >= next_state:
                    next_state = advance_periodic_deadline(next_state, state_period, now)

            now = time.monotonic()
            if now >= next_mpc:
                mpc_t0 = time.monotonic()
                state = estimator.update(vbot, bridge, cache, args, dds_model_offset, base, contact_mask, now)
                generate_traj_compat(
                    traj,
                    vbot,
                    gait,
                    now - start_t,
                    args.x_vel,
                    args.y_vel,
                    args.base_height,
                    args.yaw_rate,
                    mpc_dt,
                    args.base_pitch,
                )
                sol = mpc.solve_QP(vbot, traj, args.verbose_mpc)
                w_opt = sol["x"].full().flatten()
                u_opt = w_opt[12 * traj.N :].reshape((12, traj.N), order="F")
                mpc_force_now = filter_mpc_force_now(args, u_opt[:, 0])
                mpc_dt_observed = time.monotonic() - mpc_t0
                mpc_time_sum += mpc_dt_observed
                mpc_time_max = max(mpc_time_max, mpc_dt_observed)
                mpc_update_count += 1
                next_mpc = advance_periodic_deadline(next_mpc, mpc_period, now)

            now = time.monotonic()
            if now >= next_leg:
                leg_t0 = time.monotonic()
                tau_model, contact_mask, leg_outputs = compute_tau_from_force(
                    vbot,
                    gait,
                    leg_controller,
                    mpc_force_now,
                    gait_time,
                    args,
                )
                tau_raw = model_tau_to_dds_order(bridge, args, tau_model)
                tau_limit_cmd = torque_limit_vector_for_contact(
                    args,
                    motor_torque_limit_vector(bridge.order),
                    bridge,
                    contact_mask,
                )
                tau_full, tau_clip, _alpha_unused, clipped_count = limit_tau_to_motor_range(
                    tau_raw,
                    tau_limit_cmd,
                )
                alpha = handover_alpha
                tau_cmd = alpha * tau_scale_vector_for_contact(args, bridge, contact_mask) * tau_full
                if args.lift_leg is None:
                    q_model_for_targets = feedback_to_model_dict(cache, bridge, args, dds_model_offset)
                    q_target_model, qd_target_model = update_targets_from_swing_outputs(
                        args,
                        vbot,
                        bridge,
                        q_stand_model,
                        q_model_for_targets,
                        contact_mask,
                        leg_outputs,
                        tau_model=tau_model,
                        kp_stance=kp_lowcmd,
                    )
                    q_target_motor = model_command_to_dds_dict(q_target_model, bridge, args, dds_model_offset)
                    qd_target_motor = model_velocity_command_to_dds_dict(qd_target_model, bridge, args)
                kp_gain_cmd, kd_gain_cmd = gain_dict_for_contact(args, bridge, contact_mask, kp_lowcmd, kd_lowcmd)
                lowcmd_publisher.update(
                    1,
                    q_target_motor,
                    tau_cmd,
                    kp_gain_cmd,
                    kd_gain_cmd,
                    qd_motor_by_joint=qd_target_motor,
                )
                leg_dt_observed = time.monotonic() - leg_t0
                leg_time_sum += leg_dt_observed
                leg_time_max = max(leg_time_max, leg_dt_observed)
                leg_update_count += 1
                next_leg = advance_periodic_deadline(next_leg, leg_period, now)

            now = time.monotonic()
            if now >= next_print:
                print_t0 = time.monotonic()
                print_status(
                    bridge,
                    cache,
                    q_target_model,
                    tau_model,
                    tau_cmd,
                    now - start_t,
                    mpc,
                    alpha,
                    clipped_count,
                    handover_alpha,
                    kp_cmd,
                    kd_cmd,
                    kp_lowcmd,
                    kd_lowcmd,
                    tau_limit_cmd,
                    state,
                    contact_mask,
                    lift_alpha,
                    args,
                    dds_model_offset,
                )
                write_log_rows(
                    log_writer,
                    bridge,
                    cache,
                    q_target_model,
                    tau_model,
                    tau_raw,
                    tau_clip,
                    tau_cmd,
                    now - start_t,
                    "real_mpc",
                    mpc,
                    alpha,
                    clipped_count,
                    handover_alpha,
                    kp_cmd,
                    kd_cmd,
                    kp_lowcmd,
                    kd_lowcmd,
                    tau_limit_cmd,
                    state,
                    contact_mask,
                    lift_alpha,
                    mpc_force_now,
                    args,
                    dds_model_offset,
                )
                if log_file is not None:
                    log_file.flush()
                print_dt = time.monotonic() - print_t0
                print_time_sum += print_dt
                print_time_max = max(print_time_max, print_dt)
                print_update_count += 1
                next_print = advance_periodic_deadline(next_print, print_period, now)

            sleep_s = min(
                args.control_sleep_s,
                max(0.0, min(next_state, next_leg, next_mpc, next_print, deadline) - time.monotonic()),
            )
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        exit_reason = "stop_requested_signal" if stop["requested"] else "completed_duration"
        return 130 if stop["requested"] else 0

    except Exception as exc:
        if exit_reason in ("not_started", "completed_duration"):
            exit_reason = f"exception:{type(exc).__name__}"
        raise
    finally:
        if lowcmd_publisher is not None:
            lowcmd_publisher.stop_and_join()
            publisher_stats = lowcmd_publisher.snapshot_stats()
            overlay_lowcmd_times = publisher_stats["times"]
            publish_count = publisher_stats["count"]
            publish_time_sum = publisher_stats["publish_time_sum"]
            publish_time_max = publisher_stats["publish_time_max"]
            if publisher_stats["error"] and exit_reason in ("not_started", "completed_duration"):
                exit_reason = f"lowcmd_publisher:{publisher_stats['error']}"

        if entered_control and q_target_motor is not None:
            if args.return_pose_on_exit is not None and args.return_ramp_seconds > 0.0:
                try:
                    print(f"\nReturning to {args.return_pose_on_exit} for {args.return_ramp_seconds:.2f}s with PD only...")
                    cache = read_lowstate_motor_feedback(latest["msg"], bridge.order)
                    q_return_start = feedback_to_model_dict(cache, bridge, args, dds_model_offset)
                    q_exit_model = load_model_pose(args.model_poses, args.return_pose_on_exit, bridge.order)
                    return_start_t = time.monotonic()
                    return_deadline = return_start_t + args.return_ramp_seconds
                    next_return_cmd = return_start_t
                    while time.monotonic() < return_deadline:
                        now = time.monotonic()
                        if now - latest["t"] > args.feedback_timeout_s:
                            raise RuntimeError(f"rt/lowstate timeout during return ramp > {args.feedback_timeout_s:.3f}s")
                        alpha_return = smoothstep((now - return_start_t) / max(args.return_ramp_seconds, 1.0e-9))
                        q_return = dict_interpolate(q_return_start, q_exit_model, alpha_return)
                        q_return_motor = model_command_to_dds_dict(q_return, bridge, args, dds_model_offset)
                        if now >= next_return_cmd:
                            cmd = make_lowcmd(1, bridge.order, q_return_motor, None, args.kp, args.kd)
                            publish_lowcmd(pub, cmd, crc)
                            next_return_cmd = advance_periodic_deadline(next_return_cmd, cmd_period, now)
                        time.sleep(0.001)
                    q_target_motor = model_command_to_dds_dict(q_exit_model, bridge, args, dds_model_offset)
                except Exception as exc:
                    print(f"Return pose failed: {exc}", file=sys.stderr)

            print(f"\nPublishing stand PD with zero MPC torque for {args.zero_on_exit_seconds:.2f}s...")
            try:
                hold_cmd = make_lowcmd(1, bridge.order, q_target_motor, None, args.kp, args.kd)
                publish_for(
                    pub,
                    hold_cmd,
                    crc,
                    args.cmd_hz,
                    args.zero_on_exit_seconds,
                    latest=latest,
                    feedback_timeout_s=args.feedback_timeout_s,
                )
            except Exception as exc:
                print(f"Exit hold failed: {exc}", file=sys.stderr)

            if args.disable_on_exit:
                print("Publishing mode=0 disable command on exit...")
                try:
                    publish_for(pub, make_lowcmd(0, bridge.order), crc, args.cmd_hz, args.zero_on_exit_seconds)
                except Exception as exc:
                    print(f"Exit disable failed: {exc}", file=sys.stderr)
        else:
            print("\nNo valid control phase entered; skip exit stand hold.")

        write_summary(
            log_path,
            exit_reason,
            overlay_elapsed_s,
            overlay_loop_count,
            latest,
            overlay_lowcmd_times,
            extra={
                "max_observed_qerr": f"{max_observed_qerr:.6f}",
                "max_observed_qerr_joint": max_observed_qerr_joint,
                "max_observed_tilt": f"{max_observed_tilt:.6f}",
                "max_observed_lift_alpha": f"{max_observed_lift_alpha:.6f}",
                "last_contact_mask": last_contact_mask_text,
                "state_update_count": state_update_count,
                "leg_update_count": leg_update_count,
                "mpc_update_count": mpc_update_count,
                "print_update_count": print_update_count,
                "publish_count": publish_count,
                "publish_ms_mean": f"{(1000.0 * publish_time_sum / publish_count) if publish_count else math.nan:.6f}",
                "publish_ms_max": f"{1000.0 * publish_time_max:.6f}",
                "state_ms_mean": f"{(1000.0 * state_time_sum / state_update_count) if state_update_count else math.nan:.6f}",
                "state_ms_max": f"{1000.0 * state_time_max:.6f}",
                "leg_ms_mean": f"{(1000.0 * leg_time_sum / leg_update_count) if leg_update_count else math.nan:.6f}",
                "leg_ms_max": f"{1000.0 * leg_time_max:.6f}",
                "mpc_wall_ms_mean": f"{(1000.0 * mpc_time_sum / mpc_update_count) if mpc_update_count else math.nan:.6f}",
                "mpc_wall_ms_max": f"{1000.0 * mpc_time_max:.6f}",
                "print_ms_mean": f"{(1000.0 * print_time_sum / print_update_count) if print_update_count else math.nan:.6f}",
                "print_ms_max": f"{1000.0 * print_time_max:.6f}",
            },
        )
        if log_file is not None:
            log_file.close()
            print(f"log_csv_saved: {log_path}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
