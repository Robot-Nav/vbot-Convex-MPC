"""EX29: read-only DDS lowstate monitor for real VBot bring-up.

This script only subscribes to ``rt/lowstate`` and prints motor feedback mapped
through ``vbot_real_joint_affine.yaml``. It never publishes ``rt/lowcmd`` and
therefore does not command, enable, or move the robot.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "convex_mpc"
DEFAULT_AFFINE = REPO / "configs" / "vbot_real_joint_affine.yaml"
DEFAULT_MODEL_POSES = REPO / "configs" / "vbot_model_poses.yaml"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vbot_real_affine import VBotRealJointAffine, load_model_pose  # noqa: E402


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


def cache_to_motor_dict(cache, order, key: str) -> dict[str, float]:
    return {joint: float(cache[joint][key]) for joint in order}


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
            "temperature": int(field(state, "temperature")),
        }
    return cache


def imu_summary(msg) -> str:
    try:
        imu = field(msg, "imu_state")
        quat = field(imu, "quaternion")
        gyro = field(imu, "gyroscope")
        return (
            f"imu_quat=[{float(quat[0]):+.3f},{float(quat[1]):+.3f},"
            f"{float(quat[2]):+.3f},{float(quat[3]):+.3f}] "
            f"gyro=[{float(gyro[0]):+.3f},{float(gyro[1]):+.3f},{float(gyro[2]):+.3f}]"
        )
    except Exception:
        return "imu=n/a"


def print_table(bridge, cache, pose, elapsed_s: float, msg) -> None:
    q_motor = cache_to_motor_dict(cache, bridge.order, "q")
    dq_motor = cache_to_motor_dict(cache, bridge.order, "dq")
    q_model = bridge.motor_feedback_to_model_dict(q_motor)
    dq_model = bridge.motor_feedback_to_model_velocity_dict(dq_motor)

    lost_joints = [joint for joint in bridge.order if cache[joint]["lost"] != 0]
    max_pose_joint = None
    max_pose_err = None
    if pose is not None:
        max_pose_joint = max(bridge.order, key=lambda joint: abs(q_model[joint] - pose[joint]))
        max_pose_err = q_model[max_pose_joint] - pose[max_pose_joint]

    tick_text = ""
    try:
        tick_text = f" tick={field(msg, 'tick')}"
    except Exception:
        pass

    print(
        f"\nt={elapsed_s:.3f}s{tick_text}  "
        f"lost={','.join(lost_joints) if lost_joints else 'none'}  {imu_summary(msg)}"
    )
    if max_pose_joint is not None:
        print(f"max_pose_err={max_pose_joint}:{max_pose_err:+.4f} rad")

    print("joint                 q_motor    q_model   dq_motor   dq_model  tau_est temp lost")
    print("-" * 92)
    for joint in bridge.order:
        row = cache[joint]
        print(
            f"{joint:20s} {row['q']:+8.4f} {q_model[joint]:+9.4f} "
            f"{row['dq']:+9.4f} {dq_model[joint]:+9.4f} "
            f"{row['tau_est']:+8.3f} {row['temperature']:4d} {row['lost']:4d}"
        )


def calibrate_prone_anchor_if_requested(bridge, cache, args):
    if not args.prone_calibrate_on_start:
        return bridge
    q_motor = cache_to_motor_dict(cache, bridge.order, "q")
    pose = load_model_pose(args.model_poses, args.prone_pose, bridge.order)
    calibrated = bridge.with_model_anchor(q_motor, pose)

    print(
        f"Prone anchor calibration: current lowstate motor feedback -> "
        f"{Path(args.model_poses).name}:{args.prone_pose}"
    )
    print("joint                    scale       bias    mapped_err")
    print("-" * 64)
    for joint in calibrated.order:
        affine = calibrated.joints[joint]
        err = calibrated.motor_to_model(joint, q_motor[joint]) - pose[joint]
        print(f"{joint:20s} {affine.scale:+10.4f} {affine.bias:+10.4f} {err:+12.3e}")
    return calibrated


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Read-only VBot DDS rt/lowstate monitor; publishes no rt/lowcmd"
    )
    parser.add_argument("--network", default="lo")
    parser.add_argument("--affine", default=str(DEFAULT_AFFINE))
    parser.add_argument("--model-poses", default=str(DEFAULT_MODEL_POSES))
    parser.add_argument("--pose-check", default=None, help="optional pose name, e.g. down or stand")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds; <=0 means forever")
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--wait-lowstate-s", type=float, default=5.0)
    parser.add_argument(
        "--prone-calibrate-on-start",
        action="store_true",
        help="after first lowstate, map current motor pose to --prone-pose for display",
    )
    parser.add_argument("--prone-pose", default="down")
    return parser


def validate_args(args):
    if args.print_hz <= 0.0:
        raise RuntimeError("--print-hz must be positive")
    if args.wait_lowstate_s <= 0.0:
        raise RuntimeError("--wait-lowstate-s must be positive")


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    except Exception as exc:
        print("ERROR: unitree_sdk2py is required in this Python environment.")
        print(f"Import failure: {exc}")
        return 2

    bridge = VBotRealJointAffine.from_yaml(args.affine)
    pose = load_model_pose(args.model_poses, args.pose_check, bridge.order) if args.pose_check else None
    latest = {"msg": None, "t": 0.0}
    stop = install_stop_handlers()

    def on_lowstate(msg):
        latest["msg"] = msg
        latest["t"] = time.monotonic()

    print("EX29 DDS lowstate monitor: read-only")
    print(f"network: {args.network}")
    print(f"affine: {Path(args.affine)}")
    print("Subscribing rt/lowstate only. This script does not publish rt/lowcmd.")

    ChannelFactoryInitialize(0, args.network)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_lowstate, 1)

    deadline = time.monotonic() + args.wait_lowstate_s
    while latest["msg"] is None and time.monotonic() < deadline and not stop["requested"]:
        time.sleep(0.02)
    if latest["msg"] is None:
        raise RuntimeError("no rt/lowstate received; is dds_to_serial_gateway running on the same network?")

    first_cache = read_lowstate_motor_feedback(latest["msg"], bridge.order)
    bridge = calibrate_prone_anchor_if_requested(bridge, first_cache, args)

    start_t = time.monotonic()
    end_t = math.inf if args.duration <= 0.0 else start_t + args.duration
    print_period = 1.0 / args.print_hz
    next_print = 0.0

    while time.monotonic() < end_t and not stop["requested"]:
        now = time.monotonic()
        msg = latest["msg"]
        if msg is not None and now >= next_print:
            cache = read_lowstate_motor_feedback(msg, bridge.order)
            print_table(bridge, cache, pose, now - start_t, msg)
            next_print = now + print_period
        time.sleep(0.005)

    return 130 if stop["requested"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
