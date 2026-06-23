"""EX28: print raw real motor feedback angles.

This script opens the two direct-serial CAN ports, polls motor feedback with
zero-gain frames by default, and prints the raw motor output angle for every
joint. It does not enable motors and does not send nonzero commands.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
DEFAULT_FATUDOG_SERIAL = WORKSPACE / "fatuDog" / "serial_dds_gateway"


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


_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--fatudog-serial", default=default_fatudog_serial())
_pre_args, _ = _pre_parser.parse_known_args()
FATUDOG_SERIAL = Path(_pre_args.fatudog_serial).expanduser().resolve()

if str(FATUDOG_SERIAL) not in sys.path:
    sys.path.insert(0, str(FATUDOG_SERIAL))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from protocol_codec import RangeSpec  # noqa: E402
from vbot_real_serial_utils import (  # noqa: E402
    BUS_A_IDS,
    BUS_B_IDS,
    DualBus,
    Feedback,
    JOINT_ORDER,
    JOINT_TO_CAN_ID,
    send_zero_gain_poll_all,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read and print raw VBot real motor output angles"
    )
    parser.add_argument("--port-a", default="/dev/myttyCAN0")
    parser.add_argument("--port-b", default="/dev/myttyCAN1")
    parser.add_argument("--baudrate", type=int, default=2_000_000)
    parser.add_argument("--channel", type=lambda x: int(x, 0), default=0x00)
    parser.add_argument("--fatudog-serial", default=str(FATUDOG_SERIAL))
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; <=0 means forever")
    parser.add_argument("--read-hz", type=float, default=100.0)
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--wait-feedback-s", type=float, default=3.0)
    parser.add_argument("--once", action="store_true", help="print one table and exit")
    parser.add_argument("--zero-gain-poll", action="store_true", default=True)
    parser.add_argument("--no-zero-gain-poll", action="store_false", dest="zero_gain_poll")
    return parser


def validate_args(args) -> None:
    if args.read_hz <= 0.0:
        raise RuntimeError("--read-hz must be positive")
    if args.print_hz <= 0.0:
        raise RuntimeError("--print-hz must be positive")
    if args.wait_feedback_s <= 0.0:
        raise RuntimeError("--wait-feedback-s must be positive")


def install_stop_handlers():
    stop = {"requested": False}

    def on_signal(_signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    return stop


def bus_name_for_motor_id(motor_id: int) -> str:
    if motor_id in BUS_A_IDS:
        return "A"
    if motor_id in BUS_B_IDS:
        return "B"
    return "?"


def count_seen(cache: dict[str, Feedback]) -> int:
    return sum(1 for joint in JOINT_ORDER if cache[joint].seen)


def print_motor_angle_table(cache: dict[str, Feedback], elapsed_s: float) -> None:
    print(f"\nt={elapsed_s:.3f}s  seen={count_seen(cache)}/{len(JOINT_ORDER)}")
    print("joint                 id bus    q_motor(rad)  dq_motor(rad/s)      tau   temp seen")
    print("-" * 88)
    for joint in JOINT_ORDER:
        motor_id = JOINT_TO_CAN_ID[joint]
        fb = cache[joint]
        if not fb.seen:
            print(f"{joint:20s} {motor_id:2d}  {bus_name_for_motor_id(motor_id):>1s}          n/a             n/a      n/a    n/a    0")
            continue
        print(
            f"{joint:20s} {motor_id:2d}  {bus_name_for_motor_id(motor_id):>1s} "
            f"{fb.q:+14.6f} {fb.dq:+15.6f} {fb.tau:+8.3f} "
            f"{fb.temp_c:6.1f} {int(fb.seen):4d}"
        )


def poll_once(bus: DualBus, ranges: RangeSpec, cache: dict[str, Feedback], args) -> None:
    if args.zero_gain_poll:
        send_zero_gain_poll_all(bus, ranges, args)
    bus.read_feedback(ranges, cache)


def wait_initial_feedback(bus: DualBus, ranges: RangeSpec, cache, args, stop) -> None:
    deadline = time.monotonic() + args.wait_feedback_s
    period = 1.0 / args.read_hz
    next_poll = 0.0
    while time.monotonic() < deadline and not stop["requested"]:
        now = time.monotonic()
        if now >= next_poll:
            poll_once(bus, ranges, cache, args)
            next_poll = now + period
        else:
            bus.read_feedback(ranges, cache)
        if count_seen(cache) == len(JOINT_ORDER):
            return
        time.sleep(0.002)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    stop = install_stop_handlers()
    ranges = RangeSpec()
    cache = {joint: Feedback() for joint in JOINT_ORDER}

    print("EX28 real motor angle reader")
    print(f"fatuDog serial helpers: {Path(args.fatudog_serial)}")
    print(f"ports: {args.port_a}, {args.port_b}  baudrate={args.baudrate}")
    if args.zero_gain_poll:
        print("zero-gain polling is enabled: q=0,dq=0,kp=0,kd=0,tau=0 frames may be sent.")
    else:
        print("zero-gain polling is disabled; expecting feedback to stream already.")

    bus = DualBus(args.port_a, args.port_b, args.baudrate)
    try:
        start_t = time.monotonic()
        wait_initial_feedback(bus, ranges, cache, args, stop)
        print_motor_angle_table(cache, time.monotonic() - start_t)
        if args.once or stop["requested"]:
            return 130 if stop["requested"] else 0

        read_period = 1.0 / args.read_hz
        print_period = 1.0 / args.print_hz
        deadline = math.inf if args.duration <= 0.0 else start_t + args.duration
        next_read = time.monotonic()
        next_print = time.monotonic() + print_period

        while time.monotonic() < deadline and not stop["requested"]:
            now = time.monotonic()
            if now >= next_read:
                poll_once(bus, ranges, cache, args)
                next_read = now + read_period
            else:
                bus.read_feedback(ranges, cache)

            if now >= next_print:
                print_motor_angle_table(cache, now - start_t)
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
