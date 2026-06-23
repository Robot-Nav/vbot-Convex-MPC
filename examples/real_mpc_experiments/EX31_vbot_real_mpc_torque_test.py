"""EX31: suspended low-torque real VBot MPC test.

This is the first script in this folder that can send nonzero motor torque.
Use it only with the robot suspended or firmly supported.

Safety posture:
- fixed/yaw-free base state: yaw = 0, wz = 0
- zero body velocity command: x_vel = y_vel = yaw_rate = 0
- kp = kd = q = dq = 0 in the motor command frame
- tau_motor is clipped by --tau-limit and ramped in over --ramp-seconds
- feedback timeout, Ctrl+C, or any exception sends zero torque before exit

This test does not prove standing control.  It only checks that
MPC tau_model -> affine tau_motor -> serial command -> motor response is
bounded and directionally plausible.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
SRC = REPO / "src"
CONVEX_SRC = SRC / "convex_mpc"
AFFINE_TOOLS = REPO / "examples" / "vbot_real_affine"
DEFAULT_AFFINE = REPO / "configs" / "vbot_real_joint_affine.yaml"
DEFAULT_MODEL_POSES = REPO / "configs" / "vbot_model_poses.yaml"
DEFAULT_FATUDOG_SERIAL = WORKSPACE / "fatuDog" / "serial_dds_gateway"

MPC_LEG_ORDER = ("FL", "FR", "RL", "RR")
LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}


def default_fatudog_serial() -> str:
    env_path = os.environ.get("FATUDOG_SERIAL")
    if env_path:
        return env_path
    candidates = (
        DEFAULT_FATUDOG_SERIAL,
        WORKSPACE / "fatuDog0609" / "fatuDog" / "serial_dds_gateway",
    )
    for path in candidates:
        if path.exists():
            return str(path)
    return str(DEFAULT_FATUDOG_SERIAL)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Suspended yaw-free low-torque VBot MPC hardware test"
    )
    parser.add_argument("--port-a", default="/dev/myttyCAN0")
    parser.add_argument("--port-b", default="/dev/myttyCAN1")
    parser.add_argument("--baudrate", type=int, default=2_000_000)
    parser.add_argument("--channel", type=lambda x: int(x, 0), default=0x00)
    parser.add_argument("--master-id", type=lambda x: int(x, 0), default=0x00FD)
    parser.add_argument("--fatudog-serial", default=default_fatudog_serial())
    parser.add_argument("--affine", default=str(DEFAULT_AFFINE))
    parser.add_argument("--model-poses", default=str(DEFAULT_MODEL_POSES))
    parser.add_argument(
        "--prone-calibrate-on-start",
        action="store_true",
        help="after feedback is available, map the current motor pose to --prone-pose",
    )
    parser.add_argument("--prone-pose", default="down")

    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--cmd-hz", type=float, default=100.0)
    parser.add_argument("--mpc-hz", type=float, default=10.0)
    parser.add_argument("--print-hz", type=float, default=2.0, help="0 disables tables")
    parser.add_argument("--wait-feedback-s", type=float, default=3.0)
    parser.add_argument("--feedback-timeout-s", type=float, default=0.25)

    parser.add_argument("--base-height", type=float, default=0.462)
    parser.add_argument("--base-roll", type=float, default=0.0)
    parser.add_argument("--base-pitch", type=float, default=0.0)
    parser.add_argument("--gait-hz", type=float, default=3.0)
    parser.add_argument("--gait-duty", type=float, default=0.6)
    parser.add_argument("--horizon-segments", type=int, default=16)
    parser.add_argument("--yaw-weight", type=float, default=0.0)
    parser.add_argument("--yaw-rate-weight", type=float, default=0.0)
    parser.add_argument("--verbose-mpc", action="store_true")

    parser.add_argument("--tau-limit", type=float, default=0.5, help="absolute motor torque limit in Nm")
    parser.add_argument("--allow-large-tau-limit", action="store_true")
    parser.add_argument("--ramp-seconds", type=float, default=1.0)
    parser.add_argument("--prezero-seconds", type=float, default=0.25)
    parser.add_argument("--zero-on-exit-seconds", type=float, default=0.5)

    parser.add_argument("--send-enable", action="store_true")
    parser.add_argument(
        "--assume-enabled",
        action="store_true",
        help="send torque without enable bursts because another process/operator already enabled motors",
    )
    parser.add_argument("--disable-on-exit", action="store_true", default=True)
    parser.add_argument("--no-disable-on-exit", action="store_false", dest="disable_on_exit")
    parser.add_argument("--enable-bursts", type=int, default=3)
    parser.add_argument("--disable-bursts", type=int, default=3)
    parser.add_argument("--motor-mode-interval", type=float, default=0.05)
    parser.add_argument("--enable-settle", type=float, default=0.1)

    parser.add_argument("--robot-is-suspended", action="store_true")
    parser.add_argument("--i-accept-risk", action="store_true")
    parser.add_argument("--allow-long-duration", action="store_true")
    return parser


def validate_args(args):
    if not args.i_accept_risk:
        raise RuntimeError("EX31 sends nonzero torque; pass --i-accept-risk explicitly")
    if not args.robot_is_suspended:
        raise RuntimeError("EX31 requires --robot-is-suspended")
    if not (args.send_enable or args.assume_enabled):
        raise RuntimeError("pass --send-enable or --assume-enabled explicitly")
    if args.send_enable and args.assume_enabled:
        raise RuntimeError("choose only one of --send-enable or --assume-enabled")
    if args.duration <= 0.0:
        raise RuntimeError("--duration must be positive")
    if args.duration > 10.0 and not args.allow_long_duration:
        raise RuntimeError("--duration > 10s requires --allow-long-duration")
    if args.cmd_hz <= 0.0 or args.mpc_hz <= 0.0 or args.print_hz < 0.0:
        raise RuntimeError("--cmd-hz/--mpc-hz must be positive and --print-hz non-negative")
    if args.wait_feedback_s <= 0.0 or args.feedback_timeout_s <= 0.0:
        raise RuntimeError("--wait-feedback-s and --feedback-timeout-s must be positive")
    if args.gait_hz <= 0.0 or not 0.0 < args.gait_duty < 1.0:
        raise RuntimeError("--gait-hz must be positive and --gait-duty must be in (0, 1)")
    if args.horizon_segments <= 0:
        raise RuntimeError("--horizon-segments must be positive")
    if args.yaw_weight < 0.0 or args.yaw_rate_weight < 0.0:
        raise RuntimeError("--yaw-weight and --yaw-rate-weight must be non-negative")
    if args.tau_limit <= 0.0:
        raise RuntimeError("--tau-limit must be positive")
    if args.tau_limit > 2.0 and not args.allow_large_tau_limit:
        raise RuntimeError("--tau-limit > 2.0 Nm requires --allow-large-tau-limit")
    if args.ramp_seconds < 0.0 or args.prezero_seconds < 0.0 or args.zero_on_exit_seconds < 0.0:
        raise RuntimeError("--ramp-seconds/--prezero-seconds/--zero-on-exit-seconds must be non-negative")
    if args.enable_bursts <= 0 or args.disable_bursts <= 0:
        raise RuntimeError("--enable-bursts and --disable-bursts must be positive")
    if args.motor_mode_interval < 0.0 or args.enable_settle < 0.0:
        raise RuntimeError("--motor-mode-interval and --enable-settle must be non-negative")


def add_import_paths(args):
    for path in (str(CONVEX_SRC), str(SRC), str(AFFINE_TOOLS), str(Path(args.fatudog_serial))):
        if path not in sys.path:
            sys.path.insert(0, path)


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
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


def cache_to_motor_dict(cache, order, field_name: str) -> dict[str, float]:
    return {joint: float(getattr(cache[joint], field_name)) for joint in order}


def update_vbot_yaw_free_from_feedback(vbot, bridge, cache, args):
    q_motor = cache_to_motor_dict(cache, bridge.order, "q")
    dq_motor = cache_to_motor_dict(cache, bridge.order, "dq")

    q = vbot.current_config.get_q().copy()
    dq = vbot.current_config.get_dq().copy()

    q[0:3] = [0.0, 0.0, args.base_height]
    q[3:7] = rpy_to_xyzw(args.base_roll, args.base_pitch, 0.0)
    q[7:19] = bridge.pin_joint_positions_from_motor(q_motor)

    dq[0:3] = [0.0, 0.0, 0.0]
    dq[3:6] = [0.0, 0.0, 0.0]
    dq[6:18] = bridge.pin_joint_velocities_from_motor(dq_motor)

    vbot.update_model(q, dq)
    return q_motor, dq_motor


def read_feedback_timed(bus, ranges, cache, last_seen):
    from lingzu_motor_protocol import decode_type2_serial_frame
    from motor_map import CAN_ID_TO_JOINT
    from vbot_real_serial_utils import Feedback

    now = time.monotonic()
    count = 0
    for framer in (bus.a, bus.b):
        for frame in framer.read_available_frames():
            try:
                fb = decode_type2_serial_frame(frame, ranges)
            except Exception:
                continue
            joint = CAN_ID_TO_JOINT.get(fb.motor_id)
            if joint is None:
                continue
            cache[joint] = Feedback(
                q=fb.q,
                dq=fb.dq,
                tau=fb.tau,
                temp_c=fb.temp_c,
                seen=True,
            )
            last_seen[joint] = now
            count += 1
    return count


def send_motor_mode_all(bus, args, mode_code: int):
    from lingzu_motor_protocol import (
        LINGZU_MOTOR_ENABLE_CODE,
        build_motor_mode_frame,
    )
    from motor_map import JOINT_ORDER, JOINT_TO_CAN_ID

    mode_name = "enable" if mode_code == LINGZU_MOTOR_ENABLE_CODE else "disable"
    bursts = args.enable_bursts if mode_name == "enable" else args.disable_bursts
    print(f"{mode_name} all joints: bursts={bursts}")
    for burst in range(bursts):
        for joint in JOINT_ORDER:
            motor_id = JOINT_TO_CAN_ID[joint]
            frame = build_motor_mode_frame(
                channel=args.channel,
                master_id=args.master_id,
                motor_id=motor_id,
                mode_code=mode_code,
            )
            bus.write_motor_frame(motor_id, frame)
        if burst + 1 < bursts:
            time.sleep(args.motor_mode_interval)
    if mode_name == "enable" and args.enable_settle > 0.0:
        time.sleep(args.enable_settle)


def send_torque_all(bus, ranges, args, bridge, tau_motor_cmd):
    from lingzu_motor_protocol import encode_type1_standard_serial_frame
    from motor_map import JOINT_TO_CAN_ID
    from protocol_codec import Type1Command

    for joint, tau in zip(bridge.order, tau_motor_cmd):
        motor_id = JOINT_TO_CAN_ID[joint]
        cmd = Type1Command(
            motor_id=motor_id,
            q=0.0,
            dq=0.0,
            kp=0.0,
            kd=0.0,
            tau=float(tau),
        )
        frame = encode_type1_standard_serial_frame(args.channel, cmd, ranges)
        bus.write_motor_frame(motor_id, frame)


def send_zero_torque_for(bus, ranges, args, bridge, seconds: float, stop=None):
    import numpy as np

    zero = np.zeros(len(bridge.order), dtype=float)
    if seconds <= 0.0:
        send_torque_all(bus, ranges, args, bridge, zero)
        return
    period = 1.0 / args.cmd_hz
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not (stop and stop["requested"]):
        send_torque_all(bus, ranges, args, bridge, zero)
        time.sleep(period)


def wait_for_feedback(bus, ranges, cache, last_seen, args, stop, bridge):
    deadline = time.monotonic() + args.wait_feedback_s
    next_poll = 0.0
    poll_period = 1.0 / args.cmd_hz
    zero = [0.0] * len(bridge.order)
    while time.monotonic() < deadline and not stop["requested"]:
        now = time.monotonic()
        if now >= next_poll:
            send_torque_all(bus, ranges, args, bridge, zero)
            next_poll = now + poll_period
        read_feedback_timed(bus, ranges, cache, last_seen)
        if all(cache[joint].seen for joint in bridge.order):
            return True
        time.sleep(0.002)
    return all(cache[joint].seen for joint in bridge.order)


def stale_feedback_joints(last_seen, order, timeout_s: float):
    now = time.monotonic()
    stale = []
    for joint in order:
        seen_t = last_seen.get(joint)
        if seen_t is None or now - seen_t > timeout_s:
            stale.append(joint)
    return stale


def compute_yaw_free_tau(vbot, traj, gait, mpc, leg_controller, bridge, elapsed_s, args):
    import numpy as np

    traj.generate_traj(
        vbot,
        gait,
        elapsed_s,
        0.0,
        0.0,
        args.base_height,
        0.0,
        time_step=gait.gait_period / args.horizon_segments,
    )
    sol = mpc.solve_QP(vbot, traj, args.verbose_mpc)
    w_opt = sol["x"].full().flatten()
    u_opt = w_opt[12 * traj.N :].reshape((12, traj.N), order="F")
    mpc_force_now = u_opt[:, 0]

    tau_model = np.zeros(12, dtype=float)
    for leg, leg_slice in LEG_SLICE.items():
        out = leg_controller.compute_leg_torque(
            leg,
            vbot,
            gait,
            mpc_force_now[leg_slice],
            elapsed_s,
        )
        tau_model[leg_slice] = out.tau

    tau_motor_raw = bridge.motor_torque_commands_from_mpc(tau_model)
    return tau_model, tau_motor_raw, mpc_force_now


def limit_and_ramp_tau(tau_motor_raw, args, elapsed_s):
    import numpy as np

    clipped = np.clip(tau_motor_raw, -args.tau_limit, args.tau_limit)
    if args.ramp_seconds <= 1.0e-9:
        alpha = 1.0
    else:
        alpha = max(0.0, min(1.0, elapsed_s / args.ramp_seconds))
    cmd = alpha * clipped
    clipped_count = int(np.count_nonzero(np.abs(tau_motor_raw) > args.tau_limit + 1.0e-12))
    return cmd, clipped, alpha, clipped_count


def print_torque_table(
    bridge,
    cache,
    last_seen,
    tau_model,
    tau_motor_raw,
    tau_motor_clipped,
    tau_motor_cmd,
    elapsed_s,
    mpc,
    mpc_joint_order,
    alpha,
    clipped_count,
):
    import numpy as np

    tau_model_by_joint = {
        joint: float(value) for joint, value in zip(mpc_joint_order, tau_model)
    }
    max_raw_idx = int(np.argmax(np.abs(tau_motor_raw)))
    max_cmd_idx = int(np.argmax(np.abs(tau_motor_cmd)))
    max_raw_joint = bridge.order[max_raw_idx]
    max_cmd_joint = bridge.order[max_cmd_idx]

    print(
        f"\nt={elapsed_s:.3f}s  base_source=yaw_free_fixed(yaw=0,wz=0)  "
        f"ramp={alpha:.2f}  clipped={clipped_count}/{len(bridge.order)}  "
        f"mpc_update={getattr(mpc, 'update_time', 0.0):.2f}ms  "
        f"mpc_solve={getattr(mpc, 'solve_time', 0.0):.2f}ms  "
        f"max_raw={max_raw_joint}:{float(tau_motor_raw[max_raw_idx]):+.3f}  "
        f"max_cmd={max_cmd_joint}:{float(tau_motor_cmd[max_cmd_idx]):+.3f}"
    )
    print(
        "joint                 q_model   dq_model  tau_model  "
        "tau_raw  tau_clip   tau_cmd  fb_tau age_ms"
    )
    print("-" * 104)
    now = time.monotonic()
    for i, joint in enumerate(bridge.order):
        fb = cache[joint]
        if not fb.seen:
            print(f"{joint:20s}      n/a       n/a        n/a      n/a       n/a       n/a     n/a    n/a")
            continue
        q_model = bridge.motor_to_model(joint, fb.q)
        dq_model = bridge.motor_velocity_to_model(joint, fb.dq)
        age_ms = (now - last_seen[joint]) * 1e3 if last_seen.get(joint) is not None else math.inf
        print(
            f"{joint:20s} {q_model:+8.4f} {dq_model:+9.4f} "
            f"{tau_model_by_joint[joint]:+10.3f} "
            f"{float(tau_motor_raw[i]):+8.3f} {float(tau_motor_clipped[i]):+9.3f} "
            f"{float(tau_motor_cmd[i]):+9.3f} {fb.tau:+7.3f} {age_ms:6.1f}"
        )


def calibrate_prone_anchor_if_requested(bridge, cache, args):
    if not args.prone_calibrate_on_start:
        return bridge

    from vbot_real_affine import load_model_pose

    q_motor = cache_to_motor_dict(cache, bridge.order, "q")
    pose = load_model_pose(args.model_poses, args.prone_pose, bridge.order)
    calibrated = bridge.with_model_anchor(q_motor, pose)

    print(
        f"Prone anchor calibration: current motor feedback -> "
        f"{Path(args.model_poses).name}:{args.prone_pose}"
    )
    print("joint                    scale       bias    mapped_err")
    print("-" * 64)
    for joint in calibrated.order:
        affine = calibrated.joints[joint]
        err = calibrated.motor_to_model(joint, q_motor[joint]) - pose[joint]
        print(f"{joint:20s} {affine.scale:+10.4f} {affine.bias:+10.4f} {err:+12.3e}")
    return calibrated


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    add_import_paths(args)

    import numpy as np

    from convex_mpc import centroidal_mpc as centroidal_mpc_module
    from convex_mpc.com_trajectory import ComTraj
    from convex_mpc.gait import Gait
    from convex_mpc.leg_controller import LegController
    from convex_mpc.vbot_robot_data import PinVBotModel
    from lingzu_motor_protocol import (
        LINGZU_MOTOR_DISABLE_CODE,
        LINGZU_MOTOR_ENABLE_CODE,
    )
    from protocol_codec import RangeSpec
    from vbot_real_affine import MPC_JOINT_ORDER, VBotRealJointAffine
    from vbot_real_serial_utils import DualBus, Feedback

    np.set_printoptions(precision=4, suppress=True)
    mpc_q = centroidal_mpc_module.COST_MATRIX_Q.copy()
    mpc_q[5, 5] = args.yaw_weight
    mpc_q[11, 11] = args.yaw_rate_weight
    centroidal_mpc_module.COST_MATRIX_Q = mpc_q
    CentroidalMPC = centroidal_mpc_module.CentroidalMPC

    bridge = VBotRealJointAffine.from_yaml(args.affine)
    cache = {joint: Feedback() for joint in bridge.order}
    last_seen = {joint: None for joint in bridge.order}
    ranges = RangeSpec()
    stop = install_stop_handlers()

    print("EX31 real MPC suspended low-torque test")
    print(f"affine: {Path(args.affine)}")
    print(f"fatuDog serial helpers: {Path(args.fatudog_serial)}")
    print(f"ports: {args.port_a}, {args.port_b}  baudrate={args.baudrate}")
    print(
        "WARNING: this script can send nonzero torque. Use only while the robot is "
        "suspended or firmly supported."
    )
    print(
        "Yaw-free state is forced: base_yaw=0, base_wz=0, "
        "x_vel=y_vel=yaw_rate=0."
    )
    print(f"MPC yaw weights: yaw={args.yaw_weight:.3f}, yaw_rate={args.yaw_rate_weight:.3f}")
    print(f"tau_limit={args.tau_limit:.3f} Nm  duration={args.duration:.2f}s")

    bus = DualBus(args.port_a, args.port_b, args.baudrate)
    try:
        if not wait_for_feedback(bus, ranges, cache, last_seen, args, stop, bridge):
            missing = [joint for joint in bridge.order if not cache[joint].seen]
            raise RuntimeError(f"missing motor feedback: {', '.join(missing)}")

        bridge = calibrate_prone_anchor_if_requested(bridge, cache, args)

        if args.send_enable:
            send_motor_mode_all(bus, args, LINGZU_MOTOR_ENABLE_CODE)
        else:
            print("Assuming motors are already enabled by operator/request.")

        if args.prezero_seconds > 0.0:
            print(f"Sending zero torque for {args.prezero_seconds:.2f}s before MPC torque...")
            send_zero_torque_for(bus, ranges, args, bridge, args.prezero_seconds, stop)

        vbot = PinVBotModel()
        update_vbot_yaw_free_from_feedback(vbot, bridge, cache, args)

        gait = Gait(args.gait_hz, args.gait_duty)
        traj = ComTraj(vbot)
        traj.generate_traj(
            vbot,
            gait,
            0.0,
            0.0,
            0.0,
            args.base_height,
            0.0,
            time_step=gait.gait_period / args.horizon_segments,
        )
        mpc = CentroidalMPC(vbot, traj)
        leg_controller = LegController()

        tau_model = np.zeros(12, dtype=float)
        tau_motor_raw = np.zeros(12, dtype=float)
        tau_motor_clipped = np.zeros(12, dtype=float)
        tau_motor_cmd = np.zeros(12, dtype=float)
        clipped_count = 0
        alpha = 0.0

        cmd_period = 1.0 / args.cmd_hz
        mpc_period = 1.0 / args.mpc_hz
        print_period = math.inf if args.print_hz <= 0.0 else 1.0 / args.print_hz
        start_t = time.monotonic()
        deadline = start_t + args.duration
        next_cmd = 0.0
        next_mpc = 0.0
        next_print = 0.0

        while time.monotonic() < deadline and not stop["requested"]:
            now = time.monotonic()
            elapsed = now - start_t

            read_feedback_timed(bus, ranges, cache, last_seen)
            stale = stale_feedback_joints(last_seen, bridge.order, args.feedback_timeout_s)
            if stale:
                raise RuntimeError(
                    f"feedback timeout > {args.feedback_timeout_s:.3f}s: {', '.join(stale)}"
                )

            if now >= next_mpc:
                update_vbot_yaw_free_from_feedback(vbot, bridge, cache, args)
                tau_model, tau_motor_raw, _force = compute_yaw_free_tau(
                    vbot,
                    traj,
                    gait,
                    mpc,
                    leg_controller,
                    bridge,
                    elapsed,
                    args,
                )
                tau_motor_cmd, tau_motor_clipped, alpha, clipped_count = limit_and_ramp_tau(
                    tau_motor_raw,
                    args,
                    elapsed,
                )
                next_mpc = now + mpc_period

            if now >= next_cmd:
                send_torque_all(bus, ranges, args, bridge, tau_motor_cmd)
                next_cmd = now + cmd_period

            if now >= next_print:
                print_torque_table(
                    bridge,
                    cache,
                    last_seen,
                    tau_model,
                    tau_motor_raw,
                    tau_motor_clipped,
                    tau_motor_cmd,
                    elapsed,
                    mpc,
                    MPC_JOINT_ORDER,
                    alpha,
                    clipped_count,
                )
                next_print = now + print_period

            time.sleep(0.001)

        return 130 if stop["requested"] else 0
    finally:
        try:
            print(f"\nSending zero torque for {args.zero_on_exit_seconds:.2f}s...")
            send_zero_torque_for(bus, ranges, args, bridge, args.zero_on_exit_seconds, stop=None)
            if args.disable_on_exit:
                send_motor_mode_all(bus, args, LINGZU_MOTOR_DISABLE_CODE)
        finally:
            bus.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
