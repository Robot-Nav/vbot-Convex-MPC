"""EX26: DDS joint-PD pose hold for the real VBot.

This is the safe first hardware controller for fixed poses such as ``down`` or
``stand``.  It does not run the walking MPC.  It subscribes to ``rt/lowstate``,
ramps from the current measured model pose to a named model pose, converts the
targets through the real-joint affine map, and publishes ``rt/lowcmd``.

Typical use, with the robot supported and no other lowcmd publisher running:

    python3 examples/vbot_real_affine/EX26_vbot_real_pose_pd.py --pose down --i-accept-risk
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

from vbot_real_affine import DDS_JOINT_ORDER, VBotRealJointAffine  # noqa: E402


def load_yaml(path: Path):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pose(path: Path, pose_name: str) -> dict[str, float]:
    poses = load_yaml(path)
    if pose_name not in poses:
        raise RuntimeError(f"pose {pose_name!r} not found in {path}")
    pose = {joint: float(value) for joint, value in poses[pose_name].items()}
    missing = [joint for joint in DDS_JOINT_ORDER if joint not in pose]
    if missing:
        raise RuntimeError(f"pose {pose_name!r} is missing joints: {', '.join(missing)}")
    return pose


def field(obj, name: str):
    value = getattr(obj, name)
    return value() if callable(value) else value


def read_motor_feedback(lowstate, count: int):
    q_motor = []
    dq_motor = []
    tau_est = []
    lost = []
    motor_state = field(lowstate, "motor_state")
    for i in range(count):
        state = motor_state[i]
        q_motor.append(float(field(state, "q")))
        dq_motor.append(float(field(state, "dq")))
        tau_est.append(float(field(state, "tau_est")))
        lost.append(int(field(state, "lost")))
    return q_motor, dq_motor, tau_est, lost


def make_lowcmd():
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        motor = cmd.motor_cmd[i]
        motor.mode = 0
        motor.q = 0.0
        motor.dq = 0.0
        motor.kp = 0.0
        motor.kd = 0.0
        motor.tau = 0.0
    return cmd


def maybe_crc(cmd, crc):
    if crc is not None:
        cmd.crc = crc.Crc(cmd)


def disable_all(pub, hz: float, seconds: float):
    cmd = make_lowcmd()
    crc = make_crc()
    dt = 1.0 / hz
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        maybe_crc(cmd, crc)
        pub.Write(cmd)
        time.sleep(dt)


def make_crc():
    try:
        from unitree_sdk2py.utils.crc import CRC

        return CRC()
    except Exception:
        return None


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def lerp_pose(a: dict[str, float], b: dict[str, float], alpha: float) -> dict[str, float]:
    return {joint: a[joint] + alpha * (b[joint] - a[joint]) for joint in DDS_JOINT_ORDER}


def print_target_table(
    bridge: VBotRealJointAffine,
    current_model: dict[str, float],
    target_model: dict[str, float],
):
    print("joint                 q_model_now  q_model_cmd  delta_model  q_motor_cmd")
    print("-" * 78)
    target_motor = bridge.motor_position_commands_from_model_dict(target_model)
    for joint in DDS_JOINT_ORDER:
        delta = target_model[joint] - current_model[joint]
        print(
            f"{joint:20s} {current_model[joint]:+11.5f} "
            f"{target_model[joint]:+11.5f} {delta:+12.5f} "
            f"{target_motor[joint]:+12.5f}"
        )


def print_feedback_summary(
    bridge: VBotRealJointAffine,
    q_motor,
    dq_motor,
    tau_est,
    lost,
    target_model: dict[str, float],
):
    q_model = bridge.motor_feedback_to_model_dict(dict(zip(DDS_JOINT_ORDER, q_motor)))
    dq_model = bridge.motor_feedback_to_model_velocity_dict(dict(zip(DDS_JOINT_ORDER, dq_motor)))
    max_err_joint = max(DDS_JOINT_ORDER, key=lambda joint: abs(q_model[joint] - target_model[joint]))
    max_dq_joint = max(DDS_JOINT_ORDER, key=lambda joint: abs(dq_model[joint]))
    max_tau_i = max(range(len(tau_est)), key=lambda i: abs(tau_est[i]))
    lost_joints = [joint for joint, value in zip(DDS_JOINT_ORDER, lost) if value != 0]
    print(
        f"max_err={max_err_joint}:{q_model[max_err_joint] - target_model[max_err_joint]:+.4f} rad  "
        f"max_dq={max_dq_joint}:{dq_model[max_dq_joint]:+.4f} rad/s  "
        f"max_tau_est={DDS_JOINT_ORDER[max_tau_i]}:{tau_est[max_tau_i]:+.3f} Nm  "
        f"lost={','.join(lost_joints) if lost_joints else 'none'}"
    )


def fill_pose_cmd(
    cmd,
    bridge: VBotRealJointAffine,
    target_model: dict[str, float],
    kp: float,
    kd: float,
):
    target_motor = bridge.motor_position_commands_from_model_dict(target_model)
    for i, joint in enumerate(DDS_JOINT_ORDER):
        motor = cmd.motor_cmd[i]
        motor.mode = 0x01
        motor.q = float(target_motor[joint])
        motor.dq = 0.0
        motor.kp = float(kp)
        motor.kd = float(kd)
        motor.tau = 0.0


def wait_for_lowstate(args, latest, stop):
    print("Waiting for rt/lowstate feedback...")
    deadline = time.monotonic() + args.wait_lowstate_s
    while latest["msg"] is None and time.monotonic() < deadline and not stop["requested"]:
        time.sleep(0.02)
    if latest["msg"] is None:
        raise RuntimeError("no rt/lowstate received")


def run_pose_hold(args, bridge: VBotRealJointAffine, pose: dict[str, float], latest, pub, stop):
    wait_for_lowstate(args, latest, stop)
    q0, dq0, tau0, lost0 = read_motor_feedback(latest["msg"], len(DDS_JOINT_ORDER))
    lost_joints = [joint for joint, value in zip(DDS_JOINT_ORDER, lost0) if value != 0]
    if lost_joints and args.refuse_lost:
        raise RuntimeError(f"feedback lost on joints: {', '.join(lost_joints)}")

    current_model = bridge.motor_feedback_to_model_dict(dict(zip(DDS_JOINT_ORDER, q0)))
    print_target_table(bridge, current_model, pose)

    max_delta = max(abs(pose[joint] - current_model[joint]) for joint in DDS_JOINT_ORDER)
    if max_delta > args.max_start_delta and not args.allow_large_delta:
        raise RuntimeError(
            f"largest requested model-space move is {max_delta:.3f} rad; "
            "increase --max-start-delta or pass --allow-large-delta if this is intentional"
        )

    cmd = make_lowcmd()
    crc = make_crc()
    dt = 1.0 / args.cmd_hz
    print_period = math.inf if args.print_hz <= 0.0 else 1.0 / args.print_hz
    next_print = 0.0
    start_t = time.monotonic()
    deadline = math.inf if args.duration <= 0.0 else start_t + args.duration
    last_rx_t = latest["t"]

    print(
        f"\nPublishing rt/lowcmd pose={args.pose!r}, kp={args.kp}, kd={args.kd}, "
        f"hz={args.cmd_hz}, ramp={args.ramp_seconds}s"
    )
    if args.duration <= 0.0:
        print("Duration: forever. Press Ctrl+C to stop.")
    else:
        print(f"Duration: {args.duration:.2f}s")

    while time.monotonic() < deadline and not stop["requested"]:
        now = time.monotonic()
        if latest["t"] != last_rx_t:
            last_rx_t = latest["t"]
        if now - last_rx_t > args.lowstate_timeout_s:
            raise RuntimeError(f"rt/lowstate timeout > {args.lowstate_timeout_s:.3f}s")

        alpha = smoothstep((now - start_t) / max(args.ramp_seconds, 1.0e-6))
        target_now = lerp_pose(current_model, pose, alpha)
        fill_pose_cmd(cmd, bridge, target_now, args.kp, args.kd)
        maybe_crc(cmd, crc)
        pub.Write(cmd)

        if now >= next_print:
            q, dq, tau, lost = read_motor_feedback(latest["msg"], len(DDS_JOINT_ORDER))
            print_feedback_summary(bridge, q, dq, tau, lost, pose)
            next_print = now + print_period

        sleep_t = start_t + (int((now - start_t) / dt) + 1) * dt - time.monotonic()
        if sleep_t > 0.0:
            time.sleep(sleep_t)


def parse_args():
    parser = argparse.ArgumentParser(description="Hold a named VBot model pose over DDS lowcmd")
    parser.add_argument("--network", default="lo", help="DDS network interface")
    parser.add_argument("--affine", default=str(DEFAULT_AFFINE))
    parser.add_argument("--model-poses", default=str(DEFAULT_MODEL_POSES))
    parser.add_argument("--pose", default="down", help="pose name in vbot_model_poses.yaml")
    parser.add_argument("--cmd-hz", type=float, default=200.0)
    parser.add_argument("--print-hz", type=float, default=2.0, help="0 disables periodic summaries")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; <=0 means hold forever")
    parser.add_argument("--ramp-seconds", type=float, default=5.0)
    parser.add_argument("--kp", type=float, default=6.0)
    parser.add_argument("--kd", type=float, default=0.8)
    parser.add_argument("--wait-lowstate-s", type=float, default=3.0)
    parser.add_argument("--lowstate-timeout-s", type=float, default=0.25)
    parser.add_argument("--max-start-delta", type=float, default=1.5)
    parser.add_argument("--allow-large-delta", action="store_true")
    parser.add_argument("--refuse-lost", action="store_true", default=True)
    parser.add_argument("--no-refuse-lost", action="store_false", dest="refuse_lost")
    parser.add_argument("--disable-on-exit", action="store_true", default=True)
    parser.add_argument("--no-disable-on-exit", action="store_false", dest="disable_on_exit")
    parser.add_argument("--disable-seconds", type=float, default=0.25)
    parser.add_argument("--i-accept-risk", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if not args.i_accept_risk:
        raise RuntimeError("hardware command mode requires --i-accept-risk")
    if args.cmd_hz <= 0.0:
        raise RuntimeError("--cmd-hz must be positive")
    if args.print_hz < 0.0:
        raise RuntimeError("--print-hz must be non-negative")
    if args.duration < 0.0:
        raise RuntimeError("--duration must be >= 0")
    if args.ramp_seconds < 0.0:
        raise RuntimeError("--ramp-seconds must be >= 0")
    if args.kp < 0.0 or args.kd < 0.0:
        raise RuntimeError("--kp and --kd must be non-negative")
    if args.lowstate_timeout_s <= 0.0:
        raise RuntimeError("--lowstate-timeout-s must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    bridge = VBotRealJointAffine.from_yaml(args.affine)
    pose = load_pose(Path(args.model_poses), args.pose)

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    except Exception as exc:
        print("ERROR: unitree_sdk2py is required on the robot/control computer.")
        print(f"Import failure: {exc}")
        return 2

    latest = {"msg": None, "t": 0.0}

    def on_lowstate(msg):
        latest["msg"] = msg
        latest["t"] = time.monotonic()

    stop = {"requested": False}

    def on_signal(_signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    ChannelFactoryInitialize(0, args.network)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_lowstate, 1)
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    print(f"Affine: {Path(args.affine)}")
    print(f"Pose file: {Path(args.model_poses)}")
    print(f"DDS network: {args.network}")
    print("Make sure the robot is supported and no other rt/lowcmd publisher is active.")

    try:
        run_pose_hold(args, bridge, pose, latest, pub, stop)
    finally:
        if args.disable_on_exit:
            print("\nPublishing mode=0 for all motors...")
            disable_all(pub, args.cmd_hz, args.disable_seconds)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
