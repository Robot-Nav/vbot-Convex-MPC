"""EX32B: DDS stand-pose ramp for real VBot bring-up.

This script commands a slow joint-position ramp from the current mapped model
pose to a model pose such as ``stand``. It uses the same real/model affine:

    q_motor_cmd = (q_model_cmd - bias) / scale

It publishes DDS ``rt/lowcmd`` only; the existing C++ gateway owns serial/CAN.
Use supports or a harness during bring-up.
"""

from __future__ import annotations

import argparse
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


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


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


def cache_to_motor_dict(cache, order, key: str) -> dict[str, float]:
    return {joint: float(cache[joint][key]) for joint in order}


def lost_joints(cache, order) -> list[str]:
    return [joint for joint in order if int(cache[joint]["lost"]) != 0]


def dict_interpolate(a, b, alpha: float):
    return {joint: (1.0 - alpha) * a[joint] + alpha * b[joint] for joint in a}


def make_lowcmd(mode: int, q_motor_by_joint=None, order=()):
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

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
            motor.kp = 0.0
            motor.kd = 0.0
        else:
            motor.q = 0.0
            motor.kp = 0.0
            motor.kd = 0.0
        motor.dq = 0.0
        motor.tau = 0.0
    return cmd


def set_cmd_gains(cmd, count: int, kp: float, kd: float):
    for i in range(count):
        cmd.motor_cmd[i].kp = float(kp)
        cmd.motor_cmd[i].kd = float(kd)


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


def publish_for(pub, cmd, crc, hz: float, seconds: float, stop=None):
    period = 1.0 / hz
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline and not (stop and stop["requested"]):
        now = time.monotonic()
        publish_lowcmd(pub, cmd, crc)
        sleep_s = period - (time.monotonic() - now)
        if sleep_s > 0.0:
            time.sleep(sleep_s)


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


def print_pose_progress(bridge, cache, target_model, alpha: float, elapsed_s: float):
    q_motor = cache_to_motor_dict(cache, bridge.order, "q")
    q_model = bridge.motor_feedback_to_model_dict(q_motor)
    max_joint = max(bridge.order, key=lambda joint: abs(q_model[joint] - target_model[joint]))
    print(
        f"\nt={elapsed_s:.2f}s alpha={alpha:.3f} "
        f"max_err={max_joint}:{q_model[max_joint] - target_model[max_joint]:+.4f}"
    )
    print("joint                 q_model  q_target  err_model  q_motor  tau_est lost")
    print("-" * 82)
    for joint in bridge.order:
        err = q_model[joint] - target_model[joint]
        print(
            f"{joint:20s} {q_model[joint]:+8.4f} {target_model[joint]:+8.4f} "
            f"{err:+9.4f} {q_motor[joint]:+8.4f} {cache[joint]['tau_est']:+8.3f} "
            f"{int(cache[joint]['lost']):4d}"
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="DDS stand-pose q/kp/kd ramp")
    parser.add_argument("--network", default="lo")
    parser.add_argument("--affine", default=str(DEFAULT_AFFINE))
    parser.add_argument("--model-poses", default=str(DEFAULT_MODEL_POSES))
    parser.add_argument("--target-pose", default="stand")
    parser.add_argument("--prone-calibrate-on-start", action="store_true")
    parser.add_argument("--prone-pose", default="down")
    parser.add_argument("--wait-lowstate-s", type=float, default=5.0)
    parser.add_argument("--feedback-timeout-s", type=float, default=0.25)
    parser.add_argument("--cmd-hz", type=float, default=100.0)
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--ramp-seconds", type=float, default=6.0)
    parser.add_argument("--hold-seconds", type=float, default=10.0)
    parser.add_argument("--prezero-seconds", type=float, default=0.5)
    parser.add_argument("--disable-on-exit", action="store_true")
    parser.add_argument("--disable-seconds", type=float, default=0.5)
    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--kd", type=float, default=0.35)
    parser.add_argument("--allow-large-gains", action="store_true")
    parser.add_argument("--robot-supported", action="store_true")
    parser.add_argument("--i-accept-risk", action="store_true")
    return parser


def validate_args(args):
    if not args.robot_supported:
        raise RuntimeError("EX32B requires --robot-supported")
    if not args.i_accept_risk:
        raise RuntimeError("EX32B can move the robot; pass --i-accept-risk")
    if args.cmd_hz <= 0.0 or args.print_hz < 0.0:
        raise RuntimeError("--cmd-hz must be positive and --print-hz non-negative")
    if args.wait_lowstate_s <= 0.0 or args.feedback_timeout_s <= 0.0:
        raise RuntimeError("--wait-lowstate-s and --feedback-timeout-s must be positive")
    if args.ramp_seconds <= 0.0 or args.hold_seconds < 0.0:
        raise RuntimeError("--ramp-seconds must be positive and --hold-seconds non-negative")
    if args.prezero_seconds < 0.0 or args.disable_seconds < 0.0:
        raise RuntimeError("--prezero-seconds/--disable-seconds must be non-negative")
    if args.kp < 0.0 or args.kd < 0.0:
        raise RuntimeError("--kp/--kd must be non-negative")
    if (args.kp > 20.0 or args.kd > 1.0) and not args.allow_large_gains:
        raise RuntimeError("--kp > 20 or --kd > 1 requires --allow-large-gains")


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    except Exception as exc:
        print("ERROR: unitree_sdk2py is required in this Python environment.")
        print(f"Import failure: {exc}")
        return 2

    bridge = VBotRealJointAffine.from_yaml(args.affine)
    target_model = load_model_pose(args.model_poses, args.target_pose, bridge.order)
    latest = {"msg": None, "t": 0.0}
    stop = install_stop_handlers()

    def on_lowstate(msg):
        latest["msg"] = msg
        latest["t"] = time.monotonic()

    print("EX32B DDS stand-pose ramp")
    print(f"network: {args.network}")
    print(f"target_pose: {args.target_pose}")
    print(f"kp={args.kp:.3f} kd={args.kd:.3f} ramp={args.ramp_seconds:.2f}s hold={args.hold_seconds:.2f}s")
    print("WARNING: this can move/support the robot. Use support/harness during bring-up.")

    ChannelFactoryInitialize(0, args.network)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_lowstate, 1)
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc = make_crc()

    try:
        deadline = time.monotonic() + args.wait_lowstate_s
        while latest["msg"] is None and time.monotonic() < deadline and not stop["requested"]:
            time.sleep(0.02)
        if latest["msg"] is None:
            raise RuntimeError("no rt/lowstate received; start dds_to_serial_gateway first")

        zero = make_lowcmd(mode=0, order=bridge.order)
        print(f"Publishing mode=0 zero command for {args.prezero_seconds:.2f}s...")
        publish_for(pub, zero, crc, args.cmd_hz, args.prezero_seconds, stop=stop)

        print("Waiting for non-lost motor feedback...")
        deadline = time.monotonic() + args.wait_lowstate_s
        first_cache = None
        poll = make_lowcmd(mode=1, order=bridge.order)
        period = 1.0 / args.cmd_hz
        next_cmd = 0.0
        while time.monotonic() < deadline and not stop["requested"]:
            now = time.monotonic()
            if now >= next_cmd:
                publish_lowcmd(pub, poll, crc)
                next_cmd = now + period
            msg = latest["msg"]
            if msg is not None:
                cache = read_lowstate_motor_feedback(msg, bridge.order)
                if not lost_joints(cache, bridge.order):
                    first_cache = cache
                    break
            time.sleep(0.002)
        if first_cache is None:
            raise RuntimeError("motor feedback is still lost; run EX29B first and check gateway")

        bridge = calibrate_prone_anchor_if_requested(bridge, first_cache, args)
        q_start_motor = cache_to_motor_dict(first_cache, bridge.order, "q")
        q_start_model = bridge.motor_feedback_to_model_dict(q_start_motor)

        cmd_period = 1.0 / args.cmd_hz
        print_period = math.inf if args.print_hz <= 0.0 else 1.0 / args.print_hz
        total_s = args.ramp_seconds + args.hold_seconds
        start_t = time.monotonic()
        deadline = start_t + total_s
        next_cmd = 0.0
        next_print = 0.0
        last_cmd = None

        while time.monotonic() < deadline and not stop["requested"]:
            now = time.monotonic()
            elapsed = now - start_t

            if now - latest["t"] > args.feedback_timeout_s:
                raise RuntimeError(f"rt/lowstate timeout > {args.feedback_timeout_s:.3f}s")
            cache = read_lowstate_motor_feedback(latest["msg"], bridge.order)
            lost = lost_joints(cache, bridge.order)
            if lost:
                raise RuntimeError(f"motor feedback lost: {', '.join(lost)}")

            alpha = smoothstep(min(1.0, elapsed / args.ramp_seconds))
            q_cmd_model = dict_interpolate(q_start_model, target_model, alpha)
            q_cmd_motor = bridge.motor_position_commands_from_model_dict(q_cmd_model)

            if now >= next_cmd:
                last_cmd = make_lowcmd(mode=1, q_motor_by_joint=q_cmd_motor, order=bridge.order)
                set_cmd_gains(last_cmd, len(bridge.order), args.kp, args.kd)
                publish_lowcmd(pub, last_cmd, crc)
                next_cmd = now + cmd_period

            if now >= next_print:
                print_pose_progress(bridge, cache, q_cmd_model, alpha, elapsed)
                next_print = now + print_period

            time.sleep(0.001)

        if last_cmd is not None and args.hold_seconds <= 0.0:
            publish_lowcmd(pub, last_cmd, crc)
        return 130 if stop["requested"] else 0
    finally:
        if args.disable_on_exit:
            print(f"\nPublishing mode=0 disable command for {args.disable_seconds:.2f}s...")
            publish_for(pub, make_lowcmd(mode=0, order=bridge.order), crc, args.cmd_hz, args.disable_seconds)
        else:
            print("\nExit without mode=0 disable. Keep another controller/command source ready if the robot is loaded.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
