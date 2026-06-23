"""EX29B: DDS zero-lowcmd poll for real VBot feedback bring-up.

This script publishes zero-gain, zero-torque ``rt/lowcmd`` frames so the C++
``dds_to_serial_gateway`` sends type1 frames and receives type2 motor feedback.

It is not an MPC controller. It is the bridge check between read-only DDS
lowstate monitoring and any real controller:

    rt/lowcmd(mode=1, q=0, dq=0, kp=0, kd=0, tau=0)
      -> dds_to_serial_gateway
      -> motor serial type1
      -> type2 feedback
      -> rt/lowstate(lost=0)

Because ``mode=1`` can enable motors through the gateway, this script requires
explicit risk flags. Use only while the robot is safely supported.
"""

from __future__ import annotations

import argparse
import math
import signal
import time


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


def make_lowcmd(mode: int):
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        motor = cmd.motor_cmd[i]
        motor.mode = int(mode) if i < 12 else 0
        motor.q = 0.0
        motor.dq = 0.0
        motor.kp = 0.0
        motor.kd = 0.0
        motor.tau = 0.0
    return cmd


def make_crc():
    try:
        from unitree_sdk2py.utils.crc import CRC

        return CRC()
    except Exception:
        return None


def maybe_crc(cmd, crc):
    if crc is not None:
        cmd.crc = crc.Crc(cmd)


def read_lost_summary(msg, count: int = 12):
    if msg is None:
        return "lowstate=n/a"
    motor_state = field(msg, "motor_state")
    lost = []
    q_values = []
    tau_values = []
    for i in range(count):
        state = motor_state[i]
        lost.append(int(field(state, "lost")))
        q_values.append(float(field(state, "q")))
        tau_values.append(float(field(state, "tau_est")))
    lost_count = sum(1 for value in lost if value != 0)
    max_tau = max(tau_values, key=abs) if tau_values else 0.0
    return (
        f"lost={lost_count}/{count} "
        f"q0={q_values[0]:+.4f} q5={q_values[5]:+.4f} "
        f"max_tau_est={max_tau:+.3f}"
    )


def publish_for(pub, cmd, crc, hz: float, seconds: float, stop=None, latest=None, print_hz: float = 2.0):
    dt = 1.0 / hz
    deadline = time.monotonic() + max(0.0, seconds)
    next_print = 0.0
    while time.monotonic() < deadline and not (stop and stop["requested"]):
        maybe_crc(cmd, crc)
        pub.Write(cmd)
        now = time.monotonic()
        if latest is not None and now >= next_print:
            print(read_lost_summary(latest["msg"]))
            next_print = now + (math.inf if print_hz <= 0.0 else 1.0 / print_hz)
        sleep_s = dt - (time.monotonic() - now)
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Publish zero-gain rt/lowcmd to make the C++ gateway poll motor feedback"
    )
    parser.add_argument("--network", default="lo")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--cmd-hz", type=float, default=100.0)
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--wait-lowstate-s", type=float, default=3.0)
    parser.add_argument("--prezero-seconds", type=float, default=0.25)
    parser.add_argument("--disable-seconds", type=float, default=0.5)
    parser.add_argument("--robot-supported", action="store_true")
    parser.add_argument("--i-accept-risk", action="store_true")
    return parser


def validate_args(args):
    if not args.robot_supported:
        raise RuntimeError("pass --robot-supported after physically supporting/suspending the robot")
    if not args.i_accept_risk:
        raise RuntimeError("EX29B can enable motors through the gateway; pass --i-accept-risk")
    if args.duration <= 0.0:
        raise RuntimeError("--duration must be positive")
    if args.cmd_hz <= 0.0:
        raise RuntimeError("--cmd-hz must be positive")
    if args.print_hz < 0.0:
        raise RuntimeError("--print-hz must be non-negative")
    if args.wait_lowstate_s <= 0.0:
        raise RuntimeError("--wait-lowstate-s must be positive")
    if args.prezero_seconds < 0.0 or args.disable_seconds < 0.0:
        raise RuntimeError("--prezero-seconds/--disable-seconds must be non-negative")


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

    latest = {"msg": None, "t": 0.0}
    stop = install_stop_handlers()

    def on_lowstate(msg):
        latest["msg"] = msg
        latest["t"] = time.monotonic()

    ChannelFactoryInitialize(0, args.network)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_lowstate, 1)
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    print("EX29B zero-lowcmd poll")
    print("Publishing q=0,dq=0,kp=0,kd=0,tau=0. This can set motor mode=1 through the gateway.")
    print(f"network={args.network} cmd_hz={args.cmd_hz} duration={args.duration:.2f}s")

    deadline = time.monotonic() + args.wait_lowstate_s
    while latest["msg"] is None and time.monotonic() < deadline and not stop["requested"]:
        time.sleep(0.02)
    if latest["msg"] is None:
        raise RuntimeError("no rt/lowstate received; start dds_to_serial_gateway first")
    print("initial", read_lost_summary(latest["msg"]))

    crc = make_crc()
    cmd_disable = make_lowcmd(mode=0)
    cmd_poll = make_lowcmd(mode=1)

    try:
        if args.prezero_seconds > 0.0:
            print(f"Publishing mode=0 for {args.prezero_seconds:.2f}s before polling...")
            publish_for(pub, cmd_disable, crc, args.cmd_hz, args.prezero_seconds, stop=stop)

        print("Publishing zero-gain mode=1 poll frames...")
        publish_for(
            pub,
            cmd_poll,
            crc,
            args.cmd_hz,
            args.duration,
            stop=stop,
            latest=latest,
            print_hz=args.print_hz,
        )
    finally:
        print(f"Publishing mode=0 for {args.disable_seconds:.2f}s on exit...")
        publish_for(pub, cmd_disable, crc, args.cmd_hz, args.disable_seconds)

    print("final", read_lost_summary(latest["msg"]))
    return 130 if stop["requested"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
