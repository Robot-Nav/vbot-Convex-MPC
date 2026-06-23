"""EX33B: DDS stand hold with small MPC torque overlay.

This script assumes the robot can already reach/hold the model ``stand`` pose
with joint-position PD. With the current C++ DDS gateway, rt/lowstate and
rt/lowcmd are model-space joint coordinates by default; the gateway owns the
motor sign, calf 2:1 gear, and prone bias mapping. This script keeps publishing
the stand position target and adds a clipped/ramped MPC feed-forward torque:

    q_model_feedback = lowstate.q - optional_startup_offset
    tau_model_mpc    = MPC(q_model_feedback, fixed base state, all stance)
    lowcmd           = q_model_stand + optional_startup_offset + kp/kd + tau_model_mpc

The serial/CAN side remains owned by the existing C++ DDS gateway.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import inspect
import math
import signal
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
CONVEX_SRC = SRC / "convex_mpc"
DEFAULT_AFFINE = REPO / "configs" / "vbot_real_joint_affine.yaml"
DEFAULT_MODEL_POSES = REPO / "configs" / "vbot_model_poses.yaml"

for path in (str(CONVEX_SRC), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from vbot_real_affine import PIN_JOINT_ORDER, MPC_JOINT_ORDER, VBotRealJointAffine, load_model_pose  # noqa: E402


LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}

LEG_JOINTS = {
    "FL": ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
    "FR": ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
    "RL": ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
    "RR": ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
}

LEG_INDEX = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
FOOT_TOUCHDOWN_CLEARANCE = 0.02

_GENERATE_TRAJ_SUPPORTS_PITCH_DES_BODY = None


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
        traj = self.make_swing_trajectory(foot_pos, touchdown, self.swing_time, self.swing_height)
        return traj, touchdown


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, alpha: float) -> float:
    return (1.0 - alpha) * float(a) + alpha * float(b)


def advance_periodic_deadline(next_t: float, period: float, now: float) -> float:
    if math.isinf(period):
        return math.inf
    next_t += period
    if next_t <= now:
        missed = math.floor((now - next_t) / period) + 1
        next_t += missed * period
    return next_t


def generate_stand_traj_compat(traj, go2, gait, time_now, z_pos_des_body, time_step, pitch_des_body):
    global _GENERATE_TRAJ_SUPPORTS_PITCH_DES_BODY

    if _GENERATE_TRAJ_SUPPORTS_PITCH_DES_BODY is None:
        params = inspect.signature(traj.generate_traj).parameters
        _GENERATE_TRAJ_SUPPORTS_PITCH_DES_BODY = (
            "pitch_des_body" in params
            or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
        )

    common_args = (
        go2,
        gait,
        time_now,
        0.0,
        0.0,
        z_pos_des_body,
        0.0,
    )
    if _GENERATE_TRAJ_SUPPORTS_PITCH_DES_BODY:
        traj.generate_traj(
            *common_args,
            time_step=time_step,
            pitch_des_body=pitch_des_body,
        )
        return

    traj.generate_traj(*common_args, time_step=time_step)
    if hasattr(traj, "rpy_traj_world"):
        try:
            traj.rpy_traj_world[1, :] = float(pitch_des_body)
        except TypeError:
            traj.rpy_traj_world[1] = [float(pitch_des_body)] * len(traj.rpy_traj_world[1])
    if hasattr(traj, "omega_traj_world"):
        try:
            traj.omega_traj_world[1, :] = 0.0
        except TypeError:
            traj.omega_traj_world[1] = [0.0] * len(traj.omega_traj_world[1])


def apply_deadband(value: float, deadband: float) -> float:
    return 0.0 if abs(value) < deadband else value


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
    yaw = 0.0

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
        "yaw": yaw,
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


def update_vbot_from_lowstate(vbot, bridge, cache, args, dds_model_offset, base):
    q_model = feedback_to_model_dict(cache, bridge, args, dds_model_offset)
    dq_model = feedback_to_model_velocity_dict(cache, bridge, args)

    q = vbot.current_config.get_q().copy()
    dq = vbot.current_config.get_dq().copy()
    q[0:3] = [0.0, 0.0, args.base_height]
    q[3:7] = rpy_to_xyzw(base["roll"], base["pitch"], base["yaw"])
    q[7:19] = [q_model[joint] for joint in PIN_JOINT_ORDER]
    dq[0:3] = [0.0, 0.0, 0.0]
    dq[3:6] = list(base["gyro"])
    dq[6:18] = [dq_model[joint] for joint in PIN_JOINT_ORDER]
    vbot.update_model(q, dq)


def compute_stand_mpc_tau(vbot, traj, gait, mpc, leg_controller, bridge, elapsed_s, args, contact_mask):
    import numpy as np

    gait.contact_mask = np.asarray(contact_mask, dtype=np.int32)
    dt = gait.gait_period / args.horizon_segments
    generate_stand_traj_compat(
        traj,
        vbot,
        gait,
        elapsed_s,
        args.base_height,
        dt,
        args.base_pitch,
    )
    sol = mpc.solve_QP(vbot, traj, args.verbose_mpc)
    w_opt = sol["x"].full().flatten()
    u_opt = w_opt[12 * traj.N :].reshape((12, traj.N), order="F")
    force_now = u_opt[:, 0]

    tau_model = np.zeros(12, dtype=float)
    for leg, leg_slice in LEG_SLICE.items():
        if int(contact_mask[LEG_INDEX[leg]]) == 0:
            # EX33B lift is driven by q_target PD; MPC feed-forward is support-leg only.
            continue
        out = leg_controller.compute_leg_torque(
            leg,
            vbot,
            gait,
            force_now[leg_slice],
            elapsed_s,
        )
        tau_model[leg_slice] = out.tau
    tau_model_by_joint = {
        joint: float(tau)
        for joint, tau in zip(MPC_JOINT_ORDER, tau_model)
    }
    if args.dds_model_space:
        tau_motor_raw = np.asarray(
            [tau_model_by_joint[joint] for joint in bridge.order],
            dtype=float,
        )
    else:
        tau_motor_raw = bridge.motor_torque_commands_from_mpc(tau_model)
    return tau_model, tau_motor_raw


def overlay_schedule(args, elapsed_s):
    final_kp = args.kp if args.final_kp is None else args.final_kp
    final_kd = args.kd if args.final_kd is None else args.final_kd
    final_tau_limit = args.tau_limit if args.final_tau_limit is None else args.final_tau_limit
    alpha = 1.0 if args.handover_seconds <= 1.0e-9 else smoothstep(elapsed_s / args.handover_seconds)
    return (
        alpha,
        lerp(args.kp, final_kp, alpha),
        lerp(args.kd, final_kd, alpha),
        lerp(args.tau_limit, final_tau_limit, alpha),
    )


def linear_safety_scale(value: float, soft: float, hard: float, min_scale: float) -> float:
    value = abs(float(value))
    if hard <= soft:
        return 1.0 if value <= soft else min_scale
    if value <= soft:
        return 1.0
    if value >= hard:
        return min_scale
    ratio = (value - soft) / (hard - soft)
    return 1.0 - ratio * (1.0 - min_scale)


def adaptive_tau_limit(args, scheduled_tau_limit, max_qerr, max_tilt):
    if not args.adaptive_tau_limit:
        return float(scheduled_tau_limit)
    qerr_scale = linear_safety_scale(
        max_qerr,
        args.adaptive_qerr_soft,
        args.adaptive_qerr_hard,
        args.adaptive_min_scale,
    )
    tilt_scale = linear_safety_scale(
        max_tilt,
        args.adaptive_tilt_soft,
        args.adaptive_tilt_hard,
        args.adaptive_min_scale,
    )
    return float(scheduled_tau_limit) * min(qerr_scale, tilt_scale)


def slew_limit_value(current: float | None, target: float, rate: float, dt: float) -> float:
    if current is None or rate <= 0.0:
        return float(target)
    step = abs(rate) * max(0.0, dt)
    return max(current - step, min(current + step, float(target)))


def limit_and_ramp_tau(tau_motor_raw, args, elapsed_s, tau_limit):
    import numpy as np

    tau_limit = float(tau_limit)
    if args.tau_limit_mode == "scale":
        max_abs = float(np.max(np.abs(tau_motor_raw))) if len(tau_motor_raw) else 0.0
        if max_abs > tau_limit:
            clipped = np.asarray(tau_motor_raw, dtype=float) * (tau_limit / max_abs)
        else:
            clipped = np.asarray(tau_motor_raw, dtype=float).copy()
    else:
        clipped = np.clip(tau_motor_raw, -tau_limit, tau_limit)
    alpha = 1.0 if args.ramp_seconds <= 1.0e-9 else max(0.0, min(1.0, elapsed_s / args.ramp_seconds))
    return alpha * clipped, clipped, alpha, int(np.count_nonzero(np.abs(tau_motor_raw) > tau_limit + 1e-12))


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
    contact_mask[LEG_INDEX[args.lift_leg]] = 0
    return q_target, contact_mask, lift_alpha


def mask_text(contact_mask) -> str:
    return "".join(str(int(v)) for v in contact_mask)


def make_lowcmd(mode: int, order=(), q_motor_by_joint=None, tau_motor=None, kp: float = 0.0, kd: float = 0.0):
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

    if tau_motor is not None and len(tau_motor) < 12:
        raise RuntimeError(f"expected at least 12 tau commands, got {len(tau_motor)}")

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        motor = cmd.motor_cmd[i]
        motor.mode = int(mode) if i < 12 else 0
        if q_motor_by_joint is not None and i < len(order):
            motor.q = float(q_motor_by_joint[order[i]])
            motor.kp = float(kp)
            motor.kd = float(kd)
        else:
            motor.q = 0.0
            motor.kp = 0.0
            motor.kd = 0.0
        motor.dq = 0.0
        motor.tau = float(tau_motor[i]) if tau_motor is not None and i < 12 else 0.0
    return cmd


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
    while time.monotonic() < deadline and not (stop and stop["requested"]):
        now = time.monotonic()
        if latest is not None and feedback_timeout_s is not None:
            if now - latest["t"] > feedback_timeout_s:
                raise RuntimeError(
                    f"rt/lowstate timeout during publish_for > {feedback_timeout_s:.3f}s"
                )
        publish_lowcmd(pub, cmd, crc)
        sleep_s = period - (time.monotonic() - now)
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def open_log_writer(args):
    if args.no_log:
        return None, None, None
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{args.log_prefix}_{stamp}.csv"
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        f,
        fieldnames=[
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
            "tau_limit_cmd",
            "base_source",
            "base_roll",
            "base_pitch",
            "base_yaw",
            "base_gx",
            "base_gy",
            "base_gz",
            "contact_mask",
            "lift_leg",
            "lift_alpha",
        ],
    )
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
    tau_limit_cmd,
    base,
    contact_mask,
    lift_alpha,
    args,
    dds_model_offset,
):
    if writer is None:
        return
    q_model = feedback_to_model_dict(cache, bridge, args, dds_model_offset)
    tau_model_by_joint = {joint: float(value) for joint, value in zip(MPC_JOINT_ORDER, tau_model)}
    for i, joint in enumerate(bridge.order):
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
                "tau_limit_cmd": f"{tau_limit_cmd:.6f}",
                "base_source": base["source"],
                "base_roll": f"{base['roll']:.9f}",
                "base_pitch": f"{base['pitch']:.9f}",
                "base_yaw": f"{base['yaw']:.9f}",
                "base_gx": f"{base['gyro'][0]:.9f}",
                "base_gy": f"{base['gyro'][1]:.9f}",
                "base_gz": f"{base['gyro'][2]:.9f}",
                "contact_mask": mask_text(contact_mask),
                "lift_leg": args.lift_leg or "",
                "lift_alpha": f"{lift_alpha:.6f}",
            }
        )


def calibrate_prone_anchor_if_requested(bridge, cache, args):
    if args.anchor_current_to_target_on_start:
        q_motor = cache_to_motor_dict(cache, bridge.order, "q")
        pose = load_model_pose(args.model_poses, args.target_pose, bridge.order)
        calibrated = bridge.with_model_anchor(q_motor, pose)
        print(f"Target anchor calibration: current feedback -> {Path(args.model_poses).name}:{args.target_pose}")
        print("joint                    scale       bias    mapped_err")
        print("-" * 64)
        for joint in calibrated.order:
            affine = calibrated.joints[joint]
            err = calibrated.motor_to_model(joint, q_motor[joint]) - pose[joint]
            print(f"{joint:20s} {affine.scale:+10.4f} {affine.bias:+10.4f} {err:+12.3e}")
        return calibrated

    if not args.prone_calibrate_on_start:
        return bridge

    q_motor = cache_to_motor_dict(cache, bridge.order, "q")
    pose = load_model_pose(args.model_poses, args.prone_pose, bridge.order)
    calibrated = bridge.with_model_anchor(q_motor, pose)
    print(f"Prone anchor calibration: current feedback -> {Path(args.model_poses).name}:{args.prone_pose}")
    print("joint                    scale       bias    mapped_err")
    print("-" * 64)
    for joint in calibrated.order:
        affine = calibrated.joints[joint]
        err = calibrated.motor_to_model(joint, q_motor[joint]) - pose[joint]
        print(f"{joint:20s} {affine.scale:+10.4f} {affine.bias:+10.4f} {err:+12.3e}")
    return calibrated


def write_summary(
    log_path,
    exit_reason,
    overlay_elapsed_s,
    overlay_loop_count,
    latest,
    overlay_lowcmd_times,
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
    print(f"summary_saved: {summary_path}")


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


def print_table(
    bridge,
    cache,
    q_target_model,
    tau_model,
    tau_raw,
    tau_clip,
    tau_cmd,
    elapsed_s,
    mpc,
    alpha,
    clipped_count,
    handover_alpha,
    kp_cmd,
    kd_cmd,
    tau_limit_cmd,
    base,
    contact_mask,
    lift_alpha,
    args,
    dds_model_offset,
):
    import numpy as np

    q_model = feedback_to_model_dict(cache, bridge, args, dds_model_offset)
    tau_model_by_joint = {joint: float(value) for joint, value in zip(MPC_JOINT_ORDER, tau_model)}
    max_err_joint = max(bridge.order, key=lambda joint: abs(q_model[joint] - q_target_model[joint]))
    max_cmd_idx = int(np.argmax(np.abs(tau_cmd)))
    print(
        f"\nt={elapsed_s:.2f}s alpha={alpha:.2f} handover={handover_alpha:.2f} "
        f"kp={kp_cmd:.1f} kd={kd_cmd:.2f} tau_lim={tau_limit_cmd:.2f} "
        f"clipped={clipped_count}/12 mask={mask_text(contact_mask)} lift={args.lift_leg or '-'}:{lift_alpha:.2f} "
        f"mpc_solve={getattr(mpc, 'solve_time', 0.0):.2f}ms "
        f"base={base['source']} roll={base['roll']:+.4f} pitch={base['pitch']:+.4f} "
        f"gyro=[{base['gyro'][0]:+.4f},{base['gyro'][1]:+.4f},{base['gyro'][2]:+.4f}] "
        f"max_q_err={max_err_joint}:{q_model[max_err_joint] - q_target_model[max_err_joint]:+.4f} "
        f"max_tau_cmd={bridge.order[max_cmd_idx]}:{float(tau_cmd[max_cmd_idx]):+.3f}"
    )
    print("joint                 q_model q_stand  q_err  tau_model tau_raw tau_clip tau_cmd tau_est")
    print("-" * 96)
    for i, joint in enumerate(bridge.order):
        print(
            f"{joint:20s} {q_model[joint]:+7.3f} {q_target_model[joint]:+7.3f} "
            f"{q_model[joint] - q_target_model[joint]:+7.3f} "
            f"{tau_model_by_joint[joint]:+9.3f} {float(tau_raw[i]):+7.3f} "
            f"{float(tau_clip[i]):+8.3f} {float(tau_cmd[i]):+7.3f} {cache[joint]['tau_est']:+7.3f}"
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="DDS stand PD hold with clipped MPC torque overlay")
    parser.add_argument("--network", default="lo")
    parser.add_argument("--affine", default=str(DEFAULT_AFFINE))
    parser.add_argument("--model-poses", default=str(DEFAULT_MODEL_POSES))
    parser.add_argument("--target-pose", default="stand")
    parser.add_argument(
        "--dds-model-space",
        action="store_true",
        default=True,
        help="DDS lowstate.q/lowcmd.q are already model-space through the C++ gateway",
    )
    parser.add_argument(
        "--dds-raw-motor-space",
        action="store_false",
        dest="dds_model_space",
        help="DDS lowstate.q/lowcmd.q are raw motor-space; Python applies affine directly",
    )
    parser.add_argument("--prone-calibrate-on-start", action="store_true")
    parser.add_argument("--prone-pose", default="down")
    parser.add_argument(
        "--anchor-current-to-target-on-start",
        action="store_true",
        help="use when the robot is already at --target-pose before starting this overlay",
    )

    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--startup-ramp-seconds", type=float, default=8.0)
    parser.add_argument("--cmd-hz", type=float, default=100.0)
    parser.add_argument("--mpc-hz", type=float, default=10.0)
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--wait-lowstate-s", type=float, default=5.0)
    parser.add_argument("--feedback-timeout-s", type=float, default=0.25)

    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--kd", type=float, default=0.35)
    parser.add_argument("--final-kp", type=float, default=None)
    parser.add_argument("--final-kd", type=float, default=None)
    parser.add_argument("--handover-seconds", type=float, default=0.0)
    parser.add_argument("--allow-large-gains", action="store_true")
    parser.add_argument("--tau-limit", type=float, default=0.10)
    parser.add_argument("--final-tau-limit", type=float, default=None)
    parser.add_argument("--adaptive-tau-limit", action="store_true")
    parser.add_argument("--tau-limit-rate", type=float, default=0.0)
    parser.add_argument("--adaptive-qerr-soft", type=float, default=0.25)
    parser.add_argument("--adaptive-qerr-hard", type=float, default=0.50)
    parser.add_argument("--adaptive-tilt-soft", type=float, default=0.05)
    parser.add_argument("--adaptive-tilt-hard", type=float, default=0.10)
    parser.add_argument("--adaptive-min-scale", type=float, default=0.20)
    parser.add_argument("--tau-limit-mode", choices=("scale", "clip"), default="scale")
    parser.add_argument("--abort-on-large-error", action="store_true")
    parser.add_argument("--abort-qerr", type=float, default=0.60)
    parser.add_argument("--abort-tilt", type=float, default=0.20)
    parser.add_argument("--allow-large-tau-limit", action="store_true")
    parser.add_argument("--ramp-seconds", type=float, default=2.0)
    parser.add_argument("--prehold-seconds", type=float, default=1.0)
    parser.add_argument("--zero-on-exit-seconds", type=float, default=0.8)
    parser.add_argument("--return-pose-on-exit", default=None)
    parser.add_argument("--return-ramp-seconds", type=float, default=0.0)
    parser.add_argument("--disable-on-exit", action="store_true")

    parser.add_argument("--base-height", type=float, default=0.462)
    parser.add_argument("--base-roll", type=float, default=0.0)
    parser.add_argument("--base-pitch", type=float, default=0.0)
    parser.add_argument("--base-yaw", type=float, default=0.0)
    parser.add_argument("--use-imu-base-state", action="store_true")
    parser.add_argument("--imu-rp-zero-on-start", action="store_true")
    parser.add_argument("--imu-rp-zero-seconds", type=float, default=0.5)
    parser.add_argument("--imu-roll-sign", type=float, default=1.0)
    parser.add_argument("--imu-pitch-sign", type=float, default=1.0)
    parser.add_argument("--imu-gyro-deadband", type=float, default=0.0)
    parser.add_argument("--fix-gateway-gyro-order", action="store_true")
    parser.add_argument("--horizon-seconds", type=float, default=0.40)
    parser.add_argument("--horizon-segments", type=int, default=16)
    parser.add_argument("--yaw-weight", type=float, default=0.0)
    parser.add_argument("--yaw-rate-weight", type=float, default=0.0)
    parser.add_argument("--mpc-r", type=float, default=1.0e-5)

    parser.add_argument("--lift-leg", choices=("FL", "FR", "RL", "RR"), default=None)
    parser.add_argument("--lift-start-s", type=float, default=None)
    parser.add_argument("--lift-ramp-s", type=float, default=None)
    parser.add_argument("--lift-hold-s", type=float, default=None)
    parser.add_argument("--lift-thigh-offset", type=float, default=0.08)
    parser.add_argument("--lift-calf-offset", type=float, default=-0.16)
    parser.add_argument("--allow-single-leg-lift", action="store_true")

    parser.add_argument("--verbose-mpc", action="store_true")
    parser.add_argument("--log-dir", default=str(REPO / "logs" / "real_mpc"))
    parser.add_argument("--log-prefix", default="ex33b_stand_mpc_overlay")
    parser.add_argument("--no-log", action="store_true")

    parser.add_argument("--robot-standing-supported", action="store_true")
    parser.add_argument("--i-accept-risk", action="store_true")
    parser.add_argument("--allow-long-duration", action="store_true")
    return parser


def validate_args(args):
    if not args.robot_standing_supported:
        raise RuntimeError("EX33B requires --robot-standing-supported")
    if not args.i_accept_risk:
        raise RuntimeError("EX33B can move/load the robot; pass --i-accept-risk")
    if args.prone_calibrate_on_start and args.anchor_current_to_target_on_start:
        raise RuntimeError("choose only one startup anchor mode")
    if args.duration <= 0.0 or args.duration > 15.0 and not args.allow_long_duration:
        raise RuntimeError("--duration must be in (0, 15] unless --allow-long-duration")
    if args.startup_ramp_seconds < 0.0:
        raise RuntimeError("--startup-ramp-seconds must be non-negative")
    if args.cmd_hz <= 0.0 or args.mpc_hz <= 0.0 or args.print_hz < 0.0:
        raise RuntimeError("--cmd-hz/--mpc-hz must be positive and --print-hz non-negative")
    if args.wait_lowstate_s <= 0.0 or args.feedback_timeout_s <= 0.0:
        raise RuntimeError("--wait-lowstate-s and --feedback-timeout-s must be positive")
    if args.kp < 0.0 or args.kd < 0.0:
        raise RuntimeError("--kp/--kd must be non-negative")
    if args.final_kp is not None and args.final_kp < 0.0:
        raise RuntimeError("--final-kp must be non-negative")
    if args.final_kd is not None and args.final_kd < 0.0:
        raise RuntimeError("--final-kd must be non-negative")
    gain_values = [args.kp, args.kd]
    if args.final_kp is not None:
        gain_values.append(args.final_kp)
    if args.final_kd is not None:
        gain_values.append(args.final_kd)
    if (max(gain_values) > 20.0 or args.kd > 1.0 or (args.final_kd or 0.0) > 1.0) and not args.allow_large_gains:
        raise RuntimeError("--kp > 20 or --kd > 1 requires --allow-large-gains")
    if args.tau_limit <= 0.0:
        raise RuntimeError("--tau-limit must be positive")
    if args.final_tau_limit is not None and args.final_tau_limit <= 0.0:
        raise RuntimeError("--final-tau-limit must be positive")
    max_tau_limit = max(args.tau_limit, args.final_tau_limit or args.tau_limit)
    if max_tau_limit > 1.0 and not args.allow_large_tau_limit:
        raise RuntimeError("--tau-limit > 1.0 Nm requires --allow-large-tau-limit")
    if args.horizon_seconds <= 0.0 or args.horizon_segments <= 0:
        raise RuntimeError("--horizon-seconds and --horizon-segments must be positive")
    if (
        args.ramp_seconds < 0.0
        or args.prehold_seconds < 0.0
        or args.zero_on_exit_seconds < 0.0
        or args.handover_seconds < 0.0
        or args.return_ramp_seconds < 0.0
    ):
        raise RuntimeError("timing values must be non-negative")
    if args.tau_limit_rate < 0.0:
        raise RuntimeError("--tau-limit-rate must be non-negative")
    if not (0.0 <= args.adaptive_min_scale <= 1.0):
        raise RuntimeError("--adaptive-min-scale must be in [0, 1]")
    if args.abort_qerr <= 0.0 or args.abort_tilt <= 0.0:
        raise RuntimeError("--abort-qerr and --abort-tilt must be positive")
    if args.imu_rp_zero_seconds < 0.0 or args.imu_gyro_deadband < 0.0:
        raise RuntimeError("IMU timing/deadband values must be non-negative")
    if args.mpc_r <= 0.0:
        raise RuntimeError("--mpc-r must be positive")
    if args.lift_leg is not None and not args.allow_single_leg_lift:
        raise RuntimeError("--lift-leg requires --allow-single-leg-lift")
    for name in ("lift_start_s", "lift_ramp_s", "lift_hold_s"):
        value = getattr(args, name)
        if value is not None and value < 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be non-negative")
    if args.return_pose_on_exit is not None and args.return_pose_on_exit not in ("stand", "down"):
        raise RuntimeError("--return-pose-on-exit currently supports stand/down")


def main() -> int:
    args = build_arg_parser().parse_args()
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
    latest = {"msg": None, "t": 0.0}
    stop = install_stop_handlers()

    def on_lowstate(msg):
        latest["msg"] = msg
        latest["t"] = time.monotonic()

    print("EX33B DDS stand PD hold + MPC torque overlay")
    print(f"network: {args.network}")
    print(f"target_pose={args.target_pose} kp={args.kp:.3f} kd={args.kd:.3f}")
    print(f"tau_limit={args.tau_limit:.3f} Nm duration={args.duration:.2f}s")
    print(f"DDS joint space: {'model-space gateway' if args.dds_model_space else 'raw motor-space'}")
    print(f"MPC input cost R diagonal: {args.mpc_r:.3e}")
    print(f"Torque limit mode: {args.tau_limit_mode}")
    if args.use_imu_base_state:
        print("Base state: IMU roll/pitch + gyro gx/gy, yaw=0 and gz=0")
        if args.fix_gateway_gyro_order:
            print("IMU gyro order workaround: DDS [gz,gy,gx] -> controller [gx,gy,gz]")
    else:
        print("Base state: fixed command-line state")
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
    entered_stand_control = False
    q_stand_motor = None
    dds_model_offset = zero_joint_dict(bridge.order)
    imu_zero_rp = (0.0, 0.0)
    exit_reason = "not_started"
    overlay_elapsed_s = 0.0
    overlay_loop_count = 0
    overlay_lowcmd_times = []

    try:
        wait_deadline = time.monotonic() + args.wait_lowstate_s
        while latest["msg"] is None and time.monotonic() < wait_deadline and not stop["requested"]:
            time.sleep(0.02)
        if latest["msg"] is None:
            raise RuntimeError("no rt/lowstate received; start dds_to_serial_gateway first")

        print("Waiting for non-lost feedback...")
        first_cache = None
        poll_cmd = make_lowcmd(1, bridge.order, None, None, 0.0, 0.0)
        cmd_period = 1.0 / args.cmd_hz
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
            raise RuntimeError("motor feedback is still lost; run EX29B first and check gateway")

        imu_zero_rp = estimate_imu_zero_rp(latest, args)

        if args.dds_model_space:
            dds_model_offset, _anchored = compute_dds_model_offset_if_requested(bridge, first_cache, args)
        else:
            bridge = calibrate_prone_anchor_if_requested(bridge, first_cache, args)

        q_stand_motor = model_command_to_dds_dict(q_stand_model, bridge, args, dds_model_offset)
        stand_cmd_zero_tau = make_lowcmd(1, bridge.order, q_stand_motor, None, args.kp, args.kd)

        q_start_model = feedback_to_model_dict(first_cache, bridge, args, dds_model_offset)

        if args.startup_ramp_seconds > 0.0:
            entered_stand_control = True
            print(
                f"Ramping current mapped pose -> {args.target_pose} for "
                f"{args.startup_ramp_seconds:.2f}s with PD only..."
            )
            ramp_start = time.monotonic()
            ramp_deadline = ramp_start + args.startup_ramp_seconds
            print_period = math.inf if args.print_hz <= 0.0 else 1.0 / args.print_hz
            next_cmd = ramp_start
            next_print = ramp_start
            while time.monotonic() < ramp_deadline and not stop["requested"]:
                now = time.monotonic()
                elapsed = now - ramp_start
                if now - latest["t"] > args.feedback_timeout_s:
                    raise RuntimeError(f"rt/lowstate timeout > {args.feedback_timeout_s:.3f}s")
                cache = read_lowstate_motor_feedback(latest["msg"], bridge.order)
                lost = lost_joints(cache, bridge.order)
                if lost:
                    raise RuntimeError(f"motor feedback lost during stand ramp: {', '.join(lost)}")

                alpha_ramp = smoothstep(elapsed / args.startup_ramp_seconds)
                q_cmd_model = dict_interpolate(q_start_model, q_stand_model, alpha_ramp)
                q_cmd_motor = model_command_to_dds_dict(q_cmd_model, bridge, args, dds_model_offset)

                if now >= next_cmd:
                    cmd = make_lowcmd(1, bridge.order, q_cmd_motor, None, args.kp, args.kd)
                    publish_lowcmd(pub, cmd, crc)
                    next_cmd = advance_periodic_deadline(next_cmd, cmd_period, now)

                if now >= next_print:
                    q_now = feedback_to_model_dict(cache, bridge, args, dds_model_offset)
                    max_err_joint = max(bridge.order, key=lambda joint: abs(q_now[joint] - q_cmd_model[joint]))
                    print(
                        f"\nramp t={elapsed:.2f}s alpha={alpha_ramp:.2f} "
                        f"max_q_err={max_err_joint}:{q_now[max_err_joint] - q_cmd_model[max_err_joint]:+.4f}"
                    )
                    next_print = advance_periodic_deadline(next_print, print_period, now)

                time.sleep(0.001)

        if args.prehold_seconds > 0.0:
            entered_stand_control = True
            print(f"Publishing stand PD with zero MPC torque for {args.prehold_seconds:.2f}s...")
            publish_for(
                pub,
                stand_cmd_zero_tau,
                crc,
                args.cmd_hz,
                args.prehold_seconds,
                stop=stop,
                latest=latest,
                feedback_timeout_s=args.feedback_timeout_s,
            )

        vbot = PinVBotModel()
        cache = read_lowstate_motor_feedback(latest["msg"], bridge.order)
        base = read_base_state(latest["msg"], args, imu_zero_rp)
        update_vbot_from_lowstate(vbot, bridge, cache, args, dds_model_offset, base)
        gait = AllStanceGait(args.horizon_seconds)
        traj = ComTraj(vbot)
        generate_stand_traj_compat(
            traj,
            vbot,
            gait,
            0.0,
            args.base_height,
            gait.gait_period / args.horizon_segments,
            args.base_pitch,
        )
        mpc = CentroidalMPC(vbot, traj)
        leg_controller = LegController()

        tau_model = np.zeros(12, dtype=float)
        tau_raw = np.zeros(12, dtype=float)
        tau_clip = np.zeros(12, dtype=float)
        tau_cmd = np.zeros(12, dtype=float)
        alpha = 0.0
        clipped_count = 0
        handover_alpha = 0.0
        kp_cmd = args.kp
        kd_cmd = args.kd
        tau_limit_cmd = args.tau_limit
        q_target_model = dict(q_stand_model)
        q_target_motor = q_stand_motor
        contact_mask = [1, 1, 1, 1]
        lift_alpha = 0.0
        last_tau_limit = None
        last_tau_limit_t = None

        start_t = time.monotonic()
        deadline = start_t + args.duration
        mpc_period = 1.0 / args.mpc_hz
        print_period = math.inf if args.print_hz <= 0.0 else 1.0 / args.print_hz
        next_cmd = start_t
        next_mpc = start_t
        next_print = start_t

        while time.monotonic() < deadline and not stop["requested"]:
            entered_stand_control = True
            now = time.monotonic()
            elapsed = now - start_t
            overlay_elapsed_s = elapsed
            overlay_loop_count += 1
            if now - latest["t"] > args.feedback_timeout_s:
                exit_reason = f"rt_lowstate_timeout>{args.feedback_timeout_s:.3f}s"
                raise RuntimeError(f"rt/lowstate timeout > {args.feedback_timeout_s:.3f}s")
            cache = read_lowstate_motor_feedback(latest["msg"], bridge.order)
            lost = lost_joints(cache, bridge.order)
            if lost:
                exit_reason = "motor_feedback_lost"
                raise RuntimeError(f"motor feedback lost: {', '.join(lost)}")

            base = read_base_state(latest["msg"], args, imu_zero_rp)
            q_target_model, contact_mask, lift_alpha = lift_target_and_contact(args, q_stand_model, elapsed)
            q_target_motor = model_command_to_dds_dict(q_target_model, bridge, args, dds_model_offset)
            q_model_for_safety = feedback_to_model_dict(cache, bridge, args, dds_model_offset)
            max_qerr = max(abs(q_model_for_safety[joint] - q_target_model[joint]) for joint in bridge.order)
            max_tilt = max(abs(base["roll"]), abs(base["pitch"]))
            if args.abort_on_large_error and (max_qerr > args.abort_qerr or max_tilt > args.abort_tilt):
                exit_reason = "abort_large_error"
                raise RuntimeError(
                    "abort overlay safety limit: "
                    f"max_qerr={max_qerr:.4f}/{args.abort_qerr:.4f} "
                    f"max_tilt={max_tilt:.4f}/{args.abort_tilt:.4f}"
                )

            handover_alpha, kp_cmd, kd_cmd, tau_limit_scheduled = overlay_schedule(args, elapsed)
            tau_limit_target = adaptive_tau_limit(args, tau_limit_scheduled, max_qerr, max_tilt)
            tau_limit_cmd = slew_limit_value(
                last_tau_limit,
                tau_limit_target,
                args.tau_limit_rate,
                0.0 if last_tau_limit_t is None else now - last_tau_limit_t,
            )
            last_tau_limit = tau_limit_cmd
            last_tau_limit_t = now

            if now >= next_mpc:
                update_vbot_from_lowstate(vbot, bridge, cache, args, dds_model_offset, base)
                tau_model, tau_raw = compute_stand_mpc_tau(
                    vbot, traj, gait, mpc, leg_controller, bridge, elapsed, args, contact_mask
                )
                tau_cmd, tau_clip, alpha, clipped_count = limit_and_ramp_tau(
                    tau_raw, args, elapsed, tau_limit_cmd
                )
                next_mpc = advance_periodic_deadline(next_mpc, mpc_period, now)

            if now >= next_cmd:
                cmd = make_lowcmd(1, bridge.order, q_target_motor, tau_cmd, kp_cmd, kd_cmd)
                publish_lowcmd(pub, cmd, crc)
                overlay_lowcmd_times.append(now)
                next_cmd = advance_periodic_deadline(next_cmd, cmd_period, now)

            if now >= next_print:
                print_table(
                    bridge, cache, q_target_model, tau_model, tau_raw, tau_clip, tau_cmd,
                    elapsed, mpc, alpha, clipped_count, handover_alpha, kp_cmd, kd_cmd,
                    tau_limit_cmd, base, contact_mask, lift_alpha, args, dds_model_offset
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
                    elapsed,
                    "mpc_overlay",
                    mpc,
                    alpha,
                    clipped_count,
                    handover_alpha,
                    kp_cmd,
                    kd_cmd,
                    tau_limit_cmd,
                    base,
                    contact_mask,
                    lift_alpha,
                    args,
                    dds_model_offset,
                )
                if log_file is not None:
                    log_file.flush()
                next_print = advance_periodic_deadline(next_print, print_period, now)

            time.sleep(0.001)

        exit_reason = "stop_requested_signal" if stop["requested"] else "completed_duration"
        return 130 if stop["requested"] else 0
    except Exception as exc:
        if exit_reason in ("not_started", "completed_duration"):
            exit_reason = f"exception:{type(exc).__name__}"
        raise
    finally:
        if entered_stand_control and q_stand_motor is not None:
            if args.return_pose_on_exit is not None and args.return_ramp_seconds > 0.0:
                try:
                    print(
                        f"\nReturning to {args.return_pose_on_exit} for "
                        f"{args.return_ramp_seconds:.2f}s with PD only..."
                    )
                    cache = read_lowstate_motor_feedback(latest["msg"], bridge.order)
                    q_return_start = feedback_to_model_dict(cache, bridge, args, dds_model_offset)
                    q_return_end = load_model_pose(args.model_poses, args.return_pose_on_exit, bridge.order)
                    return_start_t = time.monotonic()
                    return_deadline = return_start_t + args.return_ramp_seconds
                    next_return_cmd = return_start_t
                    while time.monotonic() < return_deadline:
                        now = time.monotonic()
                        if now - latest["t"] > args.feedback_timeout_s:
                            raise RuntimeError(
                                f"rt/lowstate timeout during return ramp > {args.feedback_timeout_s:.3f}s"
                            )
                        elapsed = now - return_start_t
                        alpha_return = smoothstep(elapsed / max(args.return_ramp_seconds, 1.0e-9))
                        q_return = dict_interpolate(q_return_start, q_return_end, alpha_return)
                        q_return_motor = model_command_to_dds_dict(q_return, bridge, args, dds_model_offset)
                        if now >= next_return_cmd:
                            cmd = make_lowcmd(1, bridge.order, q_return_motor, None, args.kp, args.kd)
                            publish_lowcmd(pub, cmd, crc)
                            next_return_cmd = advance_periodic_deadline(next_return_cmd, cmd_period, now)
                        time.sleep(0.001)
                    q_stand_motor = model_command_to_dds_dict(q_return_end, bridge, args, dds_model_offset)
                except Exception as exc:
                    print(f"Return pose failed: {exc}", file=sys.stderr)

            print(f"\nPublishing stand PD with zero MPC torque for {args.zero_on_exit_seconds:.2f}s...")
            try:
                hold_cmd = make_lowcmd(1, bridge.order, q_stand_motor, None, args.kp, args.kd)
                publish_for(
                    pub,
                    hold_cmd,
                    crc,
                    args.cmd_hz,
                    args.zero_on_exit_seconds,
                    stop=None,
                    latest=latest,
                    feedback_timeout_s=args.feedback_timeout_s,
                )
            except Exception as exc:
                print(f"Exit hold failed: {exc}", file=sys.stderr)

            if args.disable_on_exit:
                print("Publishing mode=0 disable command on exit...")
                try:
                    publish_for(
                        pub,
                        make_lowcmd(0, bridge.order),
                        crc,
                        args.cmd_hz,
                        args.zero_on_exit_seconds,
                        stop=None,
                    )
                except Exception as exc:
                    print(f"Exit disable failed: {exc}", file=sys.stderr)
        else:
            print("\nNo valid motor feedback/control phase entered; skip exit stand hold.")
        write_summary(
            log_path,
            exit_reason,
            overlay_elapsed_s,
            overlay_loop_count,
            latest,
            overlay_lowcmd_times,
        )
        if log_file is not None:
            log_file.close()
            print(f"log_csv_saved: {log_path}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
