"""DDS test helper for VBot real-joint affine calibration.

Default mode is read-only: subscribe rt/lowstate, map motor feedback into the
model joint convention, and optionally compare against a named model pose.

Command mode is intentionally explicit and single-joint only. Use it with the
robot safely suspended/supported and no other rt/lowcmd publisher running.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_AFFINE = REPO / "configs" / "vbot_real_joint_affine.yaml"
DEFAULT_MODEL_POSES = REPO / "configs" / "vbot_model_poses.yaml"


def load_yaml(path: Path):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def field(obj, name: str):
    value = getattr(obj, name)
    return value() if callable(value) else value


def set_field(obj, name: str, value):
    attr = getattr(obj, name)
    if callable(attr):
        attr(value)
    else:
        setattr(obj, name, value)


def indexed_field(obj, name: str, index: int):
    return field(obj, name)[index]


def load_affine(path: Path):
    raw = load_yaml(path)
    order = list(raw["dds_joint_order"])
    joints = raw["joints"]
    missing = [name for name in order if name not in joints]
    if missing:
        raise RuntimeError(f"affine file is missing joints: {', '.join(missing)}")
    return order, joints


def load_pose(path: Path, pose_name: str | None):
    if pose_name is None:
        return None
    poses = load_yaml(path)
    if pose_name not in poses:
        raise RuntimeError(f"pose {pose_name!r} not found in {path}")
    return {name: float(value) for name, value in poses[pose_name].items()}


def motor_to_model(joint_cfg, q_motor: float) -> float:
    return float(joint_cfg["scale"]) * q_motor + float(joint_cfg["bias"])


def model_to_motor(joint_cfg, q_model: float) -> float:
    scale = float(joint_cfg["scale"])
    if abs(scale) < 1.0e-9:
        raise RuntimeError("joint affine scale is too close to zero")
    return (q_model - float(joint_cfg["bias"])) / scale


def read_motor_feedback(lowstate, count: int):
    q = []
    dq = []
    tau = []
    lost = []
    motor_state = field(lowstate, "motor_state")
    for i in range(count):
        ms = motor_state[i]
        q.append(float(field(ms, "q")))
        dq.append(float(field(ms, "dq")))
        tau.append(float(field(ms, "tau_est")))
        lost.append(int(field(ms, "lost")))
    return q, dq, tau, lost


def make_lowcmd():
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0
    return cmd


def maybe_crc(cmd, crc):
    if crc is not None:
        cmd.crc = crc.Crc(cmd)


def print_table(order, joints, q_motor, dq_motor, tau, lost, expected_pose):
    print("joint                 q_motor    q_model    err_pose   dq_model   tau_est  lost")
    print("-" * 86)
    for i, name in enumerate(order):
        q_model = motor_to_model(joints[name], q_motor[i])
        dq_model = float(joints[name]["scale"]) * dq_motor[i]
        if expected_pose is None or name not in expected_pose:
            err_text = "     n/a"
        else:
            err_text = f"{q_model - expected_pose[name]:+9.4f}"
        print(
            f"{name:20s} {q_motor[i]:+8.4f} {q_model:+9.4f} "
            f"{err_text} {dq_model:+9.4f} {tau[i]:+8.3f} {lost[i]:4d}"
        )


def model_vector(order, joints, q_motor):
    return [motor_to_model(joints[name], q_motor[i]) for i, name in enumerate(order)]


def run_manual_probe(args, order, joints, latest, stop):
    print("Waiting for rt/lowstate feedback...")
    deadline = time.monotonic() + args.wait_lowstate_s
    while latest["msg"] is None and time.monotonic() < deadline and not stop["requested"]:
        time.sleep(0.02)
    if latest["msg"] is None:
        raise RuntimeError("no rt/lowstate received")

    q0, _, _, lost0 = read_motor_feedback(latest["msg"], len(order))
    lost_joints = [name for i, name in enumerate(order) if lost0[i] != 0]
    if lost_joints:
        print(f"WARNING: feedback lost on joints: {', '.join(lost_joints)}")
    q_model0 = model_vector(order, joints, q0)

    print("\nManual probe mode: no rt/lowcmd will be published.")
    print("Move one joint by hand a small amount; the largest model-space delta should be that joint.")
    print("Press Ctrl+C to stop.\n")

    period = 1.0 / args.print_hz
    next_print = 0.0
    deadline = math.inf if args.duration <= 0 else time.monotonic() + args.duration
    while time.monotonic() < deadline and not stop["requested"]:
        msg = latest["msg"]
        if msg is None:
            time.sleep(0.005)
            continue
        now = time.monotonic()
        if now >= next_print:
            q, dq, tau, lost = read_motor_feedback(msg, len(order))
            q_model = model_vector(order, joints, q)
            delta = [q_model[i] - q_model0[i] for i in range(len(order))]
            max_i = max(range(len(order)), key=lambda i: abs(delta[i]))
            print(
                f"largest={order[max_i]:20s} "
                f"dq_model={delta[max_i]:+9.5f} "
                f"q_model={q_model[max_i]:+9.5f} "
                f"q_motor={q[max_i]:+9.5f} "
                f"lost={lost[max_i]}"
            )
            if args.verbose_manual:
                for i, name in enumerate(order):
                    if abs(delta[i]) >= args.manual_threshold:
                        print(f"  {name:20s} delta_model={delta[i]:+9.5f}")
            next_print = now + period
        time.sleep(0.005)


def print_command_preview(joint_name, joint_cfg, q_motor_now, q_model_now, step_model):
    q_model_cmd = q_model_now + step_model
    q_motor_cmd = model_to_motor(joint_cfg, q_model_cmd)
    print("\nCommand preview")
    print(f"  joint:       {joint_name}")
    print(f"  q_motor_now: {q_motor_now:+.6f}")
    print(f"  q_model_now: {q_model_now:+.6f}")
    print(f"  q_model_cmd: {q_model_cmd:+.6f}  (step {step_model:+.6f})")
    print(f"  q_motor_cmd: {q_motor_cmd:+.6f}  (delta {q_motor_cmd - q_motor_now:+.6f})")
    return q_model_cmd, q_motor_cmd


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def disable_all(pub, hz: float, seconds: float):
    cmd = make_lowcmd()
    dt = 1.0 / hz
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pub.Write(cmd)
        time.sleep(dt)


def run_monitor(args, order, joints, expected_pose, latest, stop):
    period = 1.0 / args.print_hz
    next_print = 0.0
    deadline = math.inf if args.duration <= 0 else time.monotonic() + args.duration
    while time.monotonic() < deadline and not stop["requested"]:
        msg = latest["msg"]
        if msg is None:
            time.sleep(0.02)
            continue
        now = time.monotonic()
        if now >= next_print:
            q, dq, tau, lost = read_motor_feedback(msg, len(order))
            print()
            print_table(order, joints, q, dq, tau, lost, expected_pose)
            next_print = now + period
        time.sleep(0.005)


def run_command_step(args, order, joints, latest, pub, stop):
    if args.joint not in order:
        raise RuntimeError(f"--joint must be one of: {', '.join(order)}")
    if not args.i_accept_risk:
        raise RuntimeError("command mode requires --i-accept-risk")

    print("Waiting for rt/lowstate feedback...")
    deadline = time.monotonic() + args.wait_lowstate_s
    while latest["msg"] is None and time.monotonic() < deadline and not stop["requested"]:
        time.sleep(0.02)
    if latest["msg"] is None:
        raise RuntimeError("no rt/lowstate received")

    joint_index = order.index(args.joint)
    q, dq, tau, lost = read_motor_feedback(latest["msg"], len(order))
    q_hold = list(q)
    if lost[joint_index] != 0:
        raise RuntimeError(f"{args.joint} feedback lost={lost[joint_index]}; refusing command")
    if not args.passive_others:
        lost_joints = [name for i, name in enumerate(order) if lost[i] != 0]
        if lost_joints:
            raise RuntimeError(f"feedback lost on joints: {', '.join(lost_joints)}; refusing hold-all command")

    q_model_now = motor_to_model(joints[args.joint], q[joint_index])
    _, q_motor_cmd = print_command_preview(
        args.joint,
        joints[args.joint],
        q[joint_index],
        q_model_now,
        args.step_model,
    )

    print(
        f"\nSending lowcmd: step_kp={args.kp}, step_kd={args.kd}, "
        f"hold_kp={args.hold_kp}, hold_kd={args.hold_kd}, "
        f"duration={args.command_seconds}s, hz={args.cmd_hz}"
    )
    cmd = make_lowcmd()
    crc = None
    try:
        from unitree_sdk2py.utils.crc import CRC

        crc = CRC()
    except Exception:
        pass

    dt = 1.0 / args.cmd_hz
    start_t = time.monotonic()
    command_deadline = time.monotonic() + args.command_seconds
    while time.monotonic() < command_deadline and not stop["requested"]:
        elapsed = time.monotonic() - start_t
        alpha = smoothstep(elapsed / max(args.ramp_seconds, 1.0e-6))
        q_step_now = q_hold[joint_index] + alpha * (q_motor_cmd - q_hold[joint_index])

        for i in range(len(order)):
            m = cmd.motor_cmd[i]
            if args.passive_others and i != joint_index:
                m.mode = 0
                m.q = 0.0
                m.dq = 0.0
                m.kp = 0.0
                m.kd = 0.0
                m.tau = 0.0
                continue
            m.mode = 0x01
            m.q = float(q_hold[i])
            m.dq = 0.0
            m.kp = float(args.hold_kp)
            m.kd = float(args.hold_kd)
            m.tau = 0.0

        m = cmd.motor_cmd[joint_index]
        m.mode = 0x01
        m.q = float(q_step_now)
        m.dq = 0.0
        m.kp = float(args.kp)
        m.kd = float(args.kd)
        m.tau = 0.0
        maybe_crc(cmd, crc)
        pub.Write(cmd)
        time.sleep(dt)

    q_after, dq_after, tau_after, lost_after = read_motor_feedback(latest["msg"], len(order))
    q_model_after = motor_to_model(joints[args.joint], q_after[joint_index])
    print("\nResult")
    print(
        f"  {args.joint}: q_model_before={q_model_now:+.6f}, "
        f"q_model_after={q_model_after:+.6f}, "
        f"observed_step={q_model_after - q_model_now:+.6f}, "
        f"lost={lost_after[joint_index]}"
    )
    print(
        f"  motor feedback: q={q_after[joint_index]:+.6f}, "
        f"dq={dq_after[joint_index]:+.6f}, tau_est={tau_after[joint_index]:+.3f}"
    )

    if args.disable_on_exit:
        print("\nPublishing mode=0 for all motors...")
        disable_all(pub, args.cmd_hz, args.disable_seconds)


def parse_args():
    parser = argparse.ArgumentParser(description="Test VBot real joint affine mapping over DDS")
    parser.add_argument("--network", default="lo", help="DDS network interface")
    parser.add_argument("--affine", default=str(DEFAULT_AFFINE))
    parser.add_argument("--model-poses", default=str(DEFAULT_MODEL_POSES))
    parser.add_argument("--expect-pose", choices=("stand", "down"), default=None)
    parser.add_argument("--duration", type=float, default=0.0, help="monitor duration; <=0 means forever")
    parser.add_argument("--print-hz", type=float, default=1.0)
    parser.add_argument("--manual-probe", action="store_true", help="read-only delta view for moving joints by hand")
    parser.add_argument("--manual-threshold", type=float, default=0.003)
    parser.add_argument("--verbose-manual", action="store_true")

    parser.add_argument("--command-step", action="store_true", help="send one small model-space step to one joint")
    parser.add_argument("--joint", default="FR_hip_joint")
    parser.add_argument("--step-model", type=float, default=0.01)
    parser.add_argument("--kp", type=float, default=3.0)
    parser.add_argument("--kd", type=float, default=0.2)
    parser.add_argument("--hold-kp", type=float, default=1.0)
    parser.add_argument("--hold-kd", type=float, default=0.2)
    parser.add_argument("--cmd-hz", type=float, default=200.0)
    parser.add_argument("--command-seconds", type=float, default=1.0)
    parser.add_argument("--ramp-seconds", type=float, default=0.5)
    parser.add_argument("--wait-lowstate-s", type=float, default=3.0)
    parser.add_argument("--passive-others", action="store_true", help="leave non-tested motors disabled")
    parser.add_argument("--disable-on-exit", action="store_true", default=True)
    parser.add_argument("--no-disable-on-exit", action="store_false", dest="disable_on_exit")
    parser.add_argument("--disable-seconds", type=float, default=0.25)
    parser.add_argument("--i-accept-risk", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_hz <= 0.0 or args.cmd_hz <= 0.0:
        raise RuntimeError("--print-hz and --cmd-hz must be positive")

    order, joints = load_affine(Path(args.affine))
    expected_pose = load_pose(Path(args.model_poses), args.expect_pose)

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    except Exception as exc:
        print("ERROR: unitree_sdk2py is required on the robot/control computer.")
        print(f"Import failure: {exc}")
        return 2

    latest = {"msg": None}

    def on_lowstate(msg):
        latest["msg"] = msg

    stop = {"requested": False}

    def on_signal(_signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    ChannelFactoryInitialize(0, args.network)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_lowstate, 1)

    pub = None
    if args.command_step and args.manual_probe:
        raise RuntimeError("choose only one of --command-step or --manual-probe")

    if args.command_step:
        pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        pub.Init()

    print(f"Affine: {Path(args.affine)}")
    print(f"DDS network: {args.network}")
    print(f"Joint order: {', '.join(order)}")
    if args.manual_probe:
        print("Mode: manual-probe. No rt/lowcmd will be published.")
        run_manual_probe(args, order, joints, latest, stop)
    elif args.command_step:
        print("Mode: command-step. Make sure the robot is supported and no other lowcmd publisher is active.")
        run_command_step(args, order, joints, latest, pub, stop)
    else:
        print("Mode: monitor only. No rt/lowcmd will be published.")
        run_monitor(args, order, joints, expected_pose, latest, stop)

    if stop["requested"] and pub is not None and args.disable_on_exit:
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
