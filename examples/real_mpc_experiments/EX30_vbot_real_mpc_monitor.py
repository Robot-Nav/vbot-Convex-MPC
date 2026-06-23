"""EX30: monitor-only real VBot MPC bridge.

This script is the first real-MPC bring-up step. It reads direct-serial motor
feedback, maps motor coordinates into model coordinates with
``vbot_real_joint_affine.yaml``, updates the Pinocchio VBot model, runs the MPC
and leg controller, then prints the model-space and motor-space torque commands.

It does not enable motors and does not send nonzero control commands. By
default it sends zero-gain poll frames only to solicit feedback from the serial
gateway; pass ``--no-zero-gain-poll`` if your gateway streams feedback without
polling.
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
        description="Monitor real motor feedback through VBot MPC without sending control"
    )
    parser.add_argument("--port-a", default="/dev/myttyCAN0")
    parser.add_argument("--port-b", default="/dev/myttyCAN1")
    parser.add_argument("--baudrate", type=int, default=2_000_000)
    parser.add_argument("--channel", type=lambda x: int(x, 0), default=0x00)
    parser.add_argument("--fatudog-serial", default=default_fatudog_serial())
    parser.add_argument("--affine", default=str(DEFAULT_AFFINE))
    parser.add_argument("--model-poses", default=str(DEFAULT_MODEL_POSES))
    parser.add_argument(
        "--prone-calibrate-on-start",
        action="store_true",
        help="after feedback is available, map the current motor pose to --prone-pose",
    )
    parser.add_argument("--prone-pose", default="down")

    parser.add_argument("--duration", type=float, default=10.0, help="seconds; <=0 means forever")
    parser.add_argument("--read-hz", type=float, default=100.0)
    parser.add_argument("--mpc-hz", type=float, default=10.0)
    parser.add_argument("--print-hz", type=float, default=1.0)
    parser.add_argument("--wait-feedback-s", type=float, default=3.0)
    parser.add_argument("--zero-gain-poll", action="store_true", default=True)
    parser.add_argument("--no-zero-gain-poll", action="store_false", dest="zero_gain_poll")

    parser.add_argument("--base-height", type=float, default=0.462)
    parser.add_argument("--base-roll", type=float, default=0.0)
    parser.add_argument("--base-pitch", type=float, default=0.0)
    parser.add_argument("--base-yaw", type=float, default=0.0)
    parser.add_argument("--base-vx", type=float, default=0.0)
    parser.add_argument("--base-vy", type=float, default=0.0)
    parser.add_argument("--base-vz", type=float, default=0.0)
    parser.add_argument("--base-wx", type=float, default=0.0)
    parser.add_argument("--base-wy", type=float, default=0.0)
    parser.add_argument("--base-wz", type=float, default=0.0)

    parser.add_argument("--x-vel", type=float, default=0.0, help="desired body x velocity")
    parser.add_argument("--y-vel", type=float, default=0.0, help="desired body y velocity")
    parser.add_argument("--yaw-rate", type=float, default=0.0, help="desired yaw rate")
    parser.add_argument("--gait-hz", type=float, default=3.0)
    parser.add_argument("--gait-duty", type=float, default=0.6)
    parser.add_argument("--horizon-segments", type=int, default=16)
    parser.add_argument("--verbose-mpc", action="store_true")
    return parser


def validate_args(args):
    if args.read_hz <= 0.0 or args.mpc_hz <= 0.0 or args.print_hz < 0.0:
        raise RuntimeError("--read-hz/--mpc-hz must be positive and --print-hz non-negative")
    if args.wait_feedback_s <= 0.0:
        raise RuntimeError("--wait-feedback-s must be positive")
    if args.gait_hz <= 0.0 or not 0.0 < args.gait_duty < 1.0:
        raise RuntimeError("--gait-hz must be positive and --gait-duty must be in (0, 1)")
    if args.horizon_segments <= 0:
        raise RuntimeError("--horizon-segments must be positive")


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


def update_vbot_from_feedback(vbot, bridge, cache, args):
    q_motor = cache_to_motor_dict(cache, bridge.order, "q")
    dq_motor = cache_to_motor_dict(cache, bridge.order, "dq")

    q = vbot.current_config.get_q().copy()
    dq = vbot.current_config.get_dq().copy()

    q[0:3] = [0.0, 0.0, args.base_height]
    q[3:7] = rpy_to_xyzw(args.base_roll, args.base_pitch, args.base_yaw)
    q[7:19] = bridge.pin_joint_positions_from_motor(q_motor)

    dq[0:3] = [args.base_vx, args.base_vy, args.base_vz]
    dq[3:6] = [args.base_wx, args.base_wy, args.base_wz]
    dq[6:18] = bridge.pin_joint_velocities_from_motor(dq_motor)

    vbot.update_model(q, dq)
    return q_motor, dq_motor


def wait_for_feedback(bus, ranges, cache, args, stop, send_zero_gain_poll_all, joint_order):
    deadline = time.monotonic() + args.wait_feedback_s
    next_poll = 0.0
    poll_period = 1.0 / args.read_hz
    while time.monotonic() < deadline and not stop["requested"]:
        now = time.monotonic()
        if args.zero_gain_poll and now >= next_poll:
            send_zero_gain_poll_all(bus, ranges, args)
            next_poll = now + poll_period
        bus.read_feedback(ranges, cache)
        if all(cache[joint].seen for joint in joint_order):
            return True
        time.sleep(0.002)
    return all(cache[joint].seen for joint in joint_order)


def compute_monitor_tau(vbot, traj, gait, mpc, leg_controller, bridge, elapsed_s, args):
    import numpy as np

    traj.generate_traj(
        vbot,
        gait,
        elapsed_s,
        args.x_vel,
        args.y_vel,
        args.base_height,
        args.yaw_rate,
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

    tau_motor = bridge.motor_torque_commands_from_mpc(tau_model)
    return tau_model, tau_motor, mpc_force_now


def print_monitor_table(bridge, cache, tau_model, tau_motor, elapsed_s, mpc, mpc_joint_order):
    import numpy as np

    tau_model_by_joint = {
        joint: float(value) for joint, value in zip(mpc_joint_order, tau_model)
    }
    max_model_joint = max(tau_model_by_joint, key=lambda joint: abs(tau_model_by_joint[joint]))
    max_motor_idx = int(np.argmax(np.abs(tau_motor)))
    max_motor_joint = bridge.order[max_motor_idx]

    print(
        f"\nt={elapsed_s:.3f}s  base_source=fixed  "
        f"mpc_update={getattr(mpc, 'update_time', 0.0):.2f}ms  "
        f"mpc_solve={getattr(mpc, 'solve_time', 0.0):.2f}ms  "
        f"max_tau_model={max_model_joint}:{tau_model_by_joint[max_model_joint]:+.3f}  "
        f"max_tau_motor={max_motor_joint}:{float(tau_motor[max_motor_idx]):+.3f}"
    )
    print("joint                 q_model   dq_model  tau_model  tau_motor  q_motor seen")
    print("-" * 82)
    for i, joint in enumerate(bridge.order):
        fb = cache[joint]
        if not fb.seen:
            print(f"{joint:20s}      n/a       n/a        n/a        n/a      n/a    0")
            continue
        q_model = bridge.motor_to_model(joint, fb.q)
        dq_model = bridge.motor_velocity_to_model(joint, fb.dq)
        print(
            f"{joint:20s} {q_model:+8.4f} {dq_model:+9.4f} "
            f"{tau_model_by_joint[joint]:+10.3f} {float(tau_motor[i]):+10.3f} "
            f"{fb.q:+8.4f} {int(fb.seen):4d}"
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

    from convex_mpc.centroidal_mpc import CentroidalMPC
    from convex_mpc.com_trajectory import ComTraj
    from convex_mpc.gait import Gait
    from convex_mpc.leg_controller import LegController
    from convex_mpc.vbot_robot_data import PinVBotModel
    from protocol_codec import RangeSpec
    from vbot_real_affine import MPC_JOINT_ORDER, VBotRealJointAffine
    from vbot_real_serial_utils import (
        DualBus,
        Feedback,
        send_zero_gain_poll_all,
    )

    np.set_printoptions(precision=4, suppress=True)

    bridge = VBotRealJointAffine.from_yaml(args.affine)
    cache = {joint: Feedback() for joint in bridge.order}
    ranges = RangeSpec()
    stop = install_stop_handlers()

    print("EX30 real MPC monitor: read-only MPC computation")
    print(f"affine: {Path(args.affine)}")
    print(f"fatuDog serial helpers: {Path(args.fatudog_serial)}")
    print(f"ports: {args.port_a}, {args.port_b}  baudrate={args.baudrate}")
    print(
        "WARNING: base pose/velocity are fixed command-line values in this monitor; "
        "IMU/lowstate base feedback is not connected yet."
    )
    if args.zero_gain_poll:
        print("zero-gain polling is enabled: q=0,dq=0,kp=0,kd=0,tau=0 frames may be sent.")
    else:
        print("zero-gain polling is disabled; expecting feedback to stream already.")

    bus = DualBus(args.port_a, args.port_b, args.baudrate)
    try:
        if not wait_for_feedback(
            bus,
            ranges,
            cache,
            args,
            stop,
            send_zero_gain_poll_all,
            bridge.order,
        ):
            missing = [joint for joint in bridge.order if not cache[joint].seen]
            raise RuntimeError(f"missing motor feedback: {', '.join(missing)}")

        bridge = calibrate_prone_anchor_if_requested(bridge, cache, args)

        vbot = PinVBotModel()
        update_vbot_from_feedback(vbot, bridge, cache, args)

        gait = Gait(args.gait_hz, args.gait_duty)
        traj = ComTraj(vbot)
        traj.generate_traj(
            vbot,
            gait,
            0.0,
            args.x_vel,
            args.y_vel,
            args.base_height,
            args.yaw_rate,
            time_step=gait.gait_period / args.horizon_segments,
        )
        mpc = CentroidalMPC(vbot, traj)
        leg_controller = LegController()
        tau_model = np.zeros(12, dtype=float)
        tau_motor = np.zeros(12, dtype=float)

        read_period = 1.0 / args.read_hz
        mpc_period = 1.0 / args.mpc_hz
        print_period = math.inf if args.print_hz <= 0.0 else 1.0 / args.print_hz
        start_t = time.monotonic()
        deadline = math.inf if args.duration <= 0.0 else start_t + args.duration
        next_poll = 0.0
        next_mpc = 0.0
        next_print = 0.0

        while time.monotonic() < deadline and not stop["requested"]:
            now = time.monotonic()
            elapsed = now - start_t

            if args.zero_gain_poll and now >= next_poll:
                send_zero_gain_poll_all(bus, ranges, args)
                next_poll = now + read_period
            bus.read_feedback(ranges, cache)

            if now >= next_mpc:
                update_vbot_from_feedback(vbot, bridge, cache, args)
                tau_model, tau_motor, _force = compute_monitor_tau(
                    vbot,
                    traj,
                    gait,
                    mpc,
                    leg_controller,
                    bridge,
                    elapsed,
                    args,
                )
                next_mpc = now + mpc_period

            if now >= next_print:
                print_monitor_table(
                    bridge,
                    cache,
                    tau_model,
                    tau_motor,
                    elapsed,
                    mpc,
                    MPC_JOINT_ORDER,
                )
                next_print = now + print_period

            time.sleep(0.001)

        return 130 if stop["requested"] else 0
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
