"""Shared helpers for VBot direct-serial real-joint affine tests."""

from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
SRC = REPO / "src" / "convex_mpc"
DEFAULT_AFFINE = REPO / "configs" / "vbot_real_joint_affine.yaml"
DEFAULT_MODEL_POSES = REPO / "configs" / "vbot_model_poses.yaml"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vbot_real_affine import DDS_JOINT_ORDER, JOINTS, VBotRealJointAffine  # noqa: E402

from lingzu_motor_protocol import (  # noqa: E402
    LINGZU_MOTOR_DISABLE_CODE,
    LINGZU_MOTOR_ENABLE_CODE,
    build_motor_mode_frame,
    decode_type2_serial_frame,
    encode_type1_standard_serial_frame,
)
from motor_map import CAN_ID_TO_JOINT, JOINT_ORDER, JOINT_TO_CAN_ID  # noqa: E402
from protocol_codec import RangeSpec, Type1Command  # noqa: E402
from serial_framer import SerialFramer  # noqa: E402


BUS_A_IDS = {11, 21, 31, 13, 23, 33}  # FR + RR
BUS_B_IDS = {12, 22, 32, 14, 24, 34}  # FL + RL
LEG_JOINTS = {leg: list(joints) for leg, joints in JOINTS.items()}
LEG_JOINTS["ALL"] = list(DDS_JOINT_ORDER)


def load_yaml(path: Path):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_pose(path: Path, pose_name: str) -> dict[str, float]:
    poses = load_yaml(path)
    if pose_name not in poses:
        raise RuntimeError(f"pose {pose_name!r} not found in {path}")
    pose = {name: float(value) for name, value in poses[pose_name].items()}
    missing = [name for name in DDS_JOINT_ORDER if name not in pose]
    if missing:
        raise RuntimeError(f"pose {pose_name!r} is missing joints: {', '.join(missing)}")
    return pose


def load_serial_affine(path: Path, sign_only: bool = False) -> VBotRealJointAffine:
    raw = load_yaml(path)
    mapping = VBotRealJointAffine.from_yaml(path)
    if tuple(mapping.order) != tuple(JOINT_ORDER):
        raise RuntimeError(
            "affine dds_joint_order does not match fatuDog motor_map.JOINT_ORDER"
        )

    for joint in mapping.order:
        scale = mapping.joints[joint].scale
        if sign_only and abs(abs(scale) - 1.0) > 1.0e-9:
            raise RuntimeError(
                f"invalid joint mapping scale {scale:+.9f}; "
                "expected +1 or -1 because q_motor is the joint output angle"
            )

    convention = raw.get("convention", {})
    strict_anchor_check = bool(convention.get("strict_anchor_check", True))
    anchor_tolerance = float(convention.get("anchor_tolerance", 1.0e-4))
    if strict_anchor_check:
        for joint in mapping.order:
            cfg = raw["joints"][joint]
            if "q_motor_stand" not in cfg or "q_model_stand" not in cfg:
                continue
            q_model_stand = mapping.motor_to_model(joint, float(cfg["q_motor_stand"]))
            expected = float(cfg["q_model_stand"])
            if abs(q_model_stand - expected) > anchor_tolerance:
                raise RuntimeError(
                    f"{joint} stand mismatch: mapped {q_model_stand:+.9f}, "
                    f"expected {expected:+.9f}"
                )
    return mapping


@dataclass
class Feedback:
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0
    temp_c: float = 0.0
    seen: bool = False


@dataclass(frozen=True)
class JointSample:
    q_motor: float
    q_model: float
    dq_motor: float
    dq_model: float
    tau: float
    temp_c: float


class DualBus:
    def __init__(self, port_a: str, port_b: str, baudrate: int):
        self.a = SerialFramer(port=port_a, baudrate=baudrate)
        self.b = SerialFramer(port=port_b, baudrate=baudrate)

    def close(self):
        self.a.close()
        self.b.close()

    def framer_for_motor(self, motor_id: int):
        if motor_id in BUS_A_IDS:
            return self.a
        if motor_id in BUS_B_IDS:
            return self.b
        raise RuntimeError(f"unknown motor id {motor_id}")

    def write_motor_frame(self, motor_id: int, frame):
        self.framer_for_motor(motor_id).write_frame(frame)

    def read_feedback(self, ranges: RangeSpec, cache: dict[str, Feedback]):
        for framer in (self.a, self.b):
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


def require_joint(name: str):
    if name not in JOINT_ORDER:
        raise RuntimeError(f"unknown joint {name!r}")


def sample_joint(
    mapping: VBotRealJointAffine,
    cache: dict[str, Feedback],
    joint: str,
) -> JointSample:
    fb = cache[joint]
    if not fb.seen:
        raise RuntimeError(f"no feedback for {joint}")
    return JointSample(
        q_motor=fb.q,
        q_model=mapping.motor_to_model(joint, fb.q),
        dq_motor=fb.dq,
        dq_model=mapping.motor_velocity_to_model(joint, fb.dq),
        tau=fb.tau,
        temp_c=fb.temp_c,
    )


def current_model_targets(
    mapping: VBotRealJointAffine,
    cache: dict[str, Feedback],
    joints: list[str] | None = None,
) -> dict[str, float]:
    targets = {}
    target_joints = mapping.order if joints is None else joints
    for joint in target_joints:
        if not cache[joint].seen:
            raise RuntimeError(f"no feedback for {joint}; cannot form current targets")
        targets[joint] = mapping.motor_to_model(joint, cache[joint].q)
    return targets


def motor_mode_joints(args):
    if args.enable_joint is not None:
        require_joint(args.enable_joint)
        return [args.enable_joint]
    if args.mode in ("leg-pose", "pose-cycle"):
        return list(LEG_JOINTS[args.leg])
    if args.single_only:
        require_joint(args.joint)
        return [args.joint]
    return list(JOINT_ORDER)


def send_enable_disable(bus: DualBus, args, mode_code: int):
    joints = motor_mode_joints(args)
    mode_name = "enable" if mode_code == LINGZU_MOTOR_ENABLE_CODE else "disable"
    bursts = args.enable_bursts if mode_code == LINGZU_MOTOR_ENABLE_CODE else args.disable_bursts
    interval = args.motor_mode_interval
    print(f"{mode_name} joints: {', '.join(joints)}  bursts={bursts}")
    for burst in range(bursts):
        for joint in joints:
            motor_id = JOINT_TO_CAN_ID[joint]
            frame = build_motor_mode_frame(
                channel=args.channel,
                master_id=args.master_id,
                motor_id=motor_id,
                mode_code=mode_code,
            )
            bus.write_motor_frame(motor_id, frame)
        if burst + 1 < bursts:
            time.sleep(interval)
    if mode_code == LINGZU_MOTOR_ENABLE_CODE and args.enable_settle > 0.0:
        time.sleep(args.enable_settle)


def send_zero_gain_poll_all(bus: DualBus, ranges: RangeSpec, args):
    for joint in JOINT_ORDER:
        motor_id = JOINT_TO_CAN_ID[joint]
        cmd = Type1Command(
            motor_id=motor_id,
            q=0.0,
            dq=0.0,
            kp=0.0,
            kd=0.0,
            tau=0.0,
        )
        frame = encode_type1_standard_serial_frame(args.channel, cmd, ranges)
        bus.write_motor_frame(motor_id, frame)


def send_joint_hold(
    bus: DualBus,
    mapping: VBotRealJointAffine,
    ranges: RangeSpec,
    args,
    joint: str,
    q_model_target: float,
):
    motor_id = JOINT_TO_CAN_ID[joint]
    cmd = Type1Command(
        motor_id=motor_id,
        q=mapping.model_to_motor(joint, q_model_target),
        dq=0.0,
        kp=args.kp,
        kd=args.kd,
        tau=args.tau,
    )
    frame = encode_type1_standard_serial_frame(args.channel, cmd, ranges)
    bus.write_motor_frame(motor_id, frame)


def send_step_targets(
    bus: DualBus,
    mapping: VBotRealJointAffine,
    ranges: RangeSpec,
    args,
    targets: dict[str, float],
):
    if args.single_only:
        send_joint_hold(bus, mapping, ranges, args, args.joint, targets[args.joint])
        return

    for joint in mapping.order:
        if joint == args.joint:
            kp = args.kp
            kd = args.kd
            tau = args.tau
        else:
            kp = args.other_kp
            kd = args.other_kd
            tau = args.other_tau
        motor_id = JOINT_TO_CAN_ID[joint]
        cmd = Type1Command(
            motor_id=motor_id,
            q=mapping.model_to_motor(joint, targets[joint]),
            dq=0.0,
            kp=kp,
            kd=kd,
            tau=tau,
        )
        frame = encode_type1_standard_serial_frame(args.channel, cmd, ranges)
        bus.write_motor_frame(motor_id, frame)


def send_pose_targets(
    bus: DualBus,
    mapping: VBotRealJointAffine,
    ranges: RangeSpec,
    args,
    targets: dict[str, float],
):
    for joint, q_model in targets.items():
        send_joint_hold(bus, mapping, ranges, args, joint, q_model)


def wait_feedback(
    bus: DualBus,
    ranges: RangeSpec,
    cache: dict[str, Feedback],
    seconds: float,
    args=None,
    zero_gain_poll: bool = False,
    stop=None,
):
    deadline = time.monotonic() + seconds
    next_poll = 0.0
    poll_period = 1.0 / args.tx_hz if args is not None and args.tx_hz > 0.0 else 0.02
    while time.monotonic() < deadline and not (stop and stop["requested"]):
        now = time.monotonic()
        if zero_gain_poll and args is not None and now >= next_poll:
            send_zero_gain_poll_all(bus, ranges, args)
            next_poll = now + poll_period
        bus.read_feedback(ranges, cache)
        if all(cache[joint].seen for joint in JOINT_ORDER):
            return True
        time.sleep(0.002)
    return all(cache[joint].seen for joint in JOINT_ORDER)


def print_feedback(mapping: VBotRealJointAffine, cache: dict[str, Feedback]):
    print("joint                 q_motor    q_model   dq_motor   dq_model      tau   temp seen")
    print("-" * 88)
    for joint in mapping.order:
        fb = cache[joint]
        if not fb.seen:
            print(f"{joint:20s}      n/a       n/a       n/a       n/a       n/a    n/a    0")
            continue
        q_model = mapping.motor_to_model(joint, fb.q)
        dq_model = mapping.motor_velocity_to_model(joint, fb.dq)
        print(
            f"{joint:20s} {fb.q:+8.4f} {q_model:+9.4f} "
            f"{fb.dq:+9.4f} {dq_model:+9.4f} {fb.tau:+8.3f} "
            f"{fb.temp_c:6.1f} {int(fb.seen):4d}"
        )


def print_pose_check(mapping: VBotRealJointAffine, cache: dict[str, Feedback], pose: dict[str, float]):
    print("joint                 q_model   expected      error    q_motor seen")
    print("-" * 74)
    max_abs = 0.0
    missing = []
    for joint in mapping.order:
        fb = cache[joint]
        if not fb.seen:
            missing.append(joint)
            print(f"{joint:20s}       n/a {pose[joint]:+9.4f}       n/a       n/a    0")
            continue
        q_model = mapping.motor_to_model(joint, fb.q)
        expected = pose[joint]
        err = q_model - expected
        max_abs = max(max_abs, abs(err))
        print(
            f"{joint:20s} {q_model:+9.4f} {expected:+9.4f} "
            f"{err:+9.4f} {fb.q:+9.4f} {int(fb.seen):4d}"
        )
    print(f"\nmax_abs_error: {max_abs:.6f}")
    if missing:
        print(f"missing_feedback: {', '.join(missing)}")


def print_targets(
    mapping: VBotRealJointAffine,
    targets: dict[str, float],
    cache: dict[str, Feedback],
):
    print("joint                 q_model_cmd  q_motor_cmd  q_motor_now  q_motor_delta")
    print("-" * 78)
    target_order = [joint for joint in mapping.order if joint in targets]
    for joint in target_order:
        q_motor_cmd = mapping.model_to_motor(joint, targets[joint])
        if not cache[joint].seen:
            print(f"{joint:20s} {targets[joint]:+11.5f} {q_motor_cmd:+12.5f}          n/a            n/a")
        else:
            q_motor_now = cache[joint].q
            print(
                f"{joint:20s} {targets[joint]:+11.5f} {q_motor_cmd:+12.5f} "
                f"{q_motor_now:+12.5f} {q_motor_cmd - q_motor_now:+14.5f}"
            )


def print_target_feedback(
    mapping: VBotRealJointAffine,
    targets: dict[str, float],
    cache: dict[str, Feedback],
):
    print(
        "joint                 "
        "q_model_cmd  q_model_fb  model_err  "
        "q_motor_cmd  q_motor_fb  motor_err seen"
    )
    print("-" * 106)
    target_order = [joint for joint in mapping.order if joint in targets]
    for joint in target_order:
        q_model_cmd = targets[joint]
        q_motor_cmd = mapping.model_to_motor(joint, q_model_cmd)
        fb = cache[joint]
        if not fb.seen:
            print(
                f"{joint:20s} {q_model_cmd:+11.5f}        n/a        n/a  "
                f"{q_motor_cmd:+11.5f}        n/a        n/a    0"
            )
            continue

        q_model_fb = mapping.motor_to_model(joint, fb.q)
        print(
            f"{joint:20s} {q_model_cmd:+11.5f} {q_model_fb:+11.5f} "
            f"{q_model_fb - q_model_cmd:+10.5f}  "
            f"{q_motor_cmd:+11.5f} {fb.q:+11.5f} "
            f"{fb.q - q_motor_cmd:+10.5f} {int(fb.seen):4d}"
        )


def open_csv(path_text: str | None):
    if not path_text:
        return None, None
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "t",
            "phase",
            "joint",
            "q_motor",
            "q_model",
            "dq_motor",
            "dq_model",
            "tau",
            "temp_c",
            "q_model_cmd",
            "q_motor_cmd",
            "q_model_err",
            "q_motor_err",
            "alpha",
        ],
    )
    writer.writeheader()
    return f, writer


def log_sample(writer, t0: float, phase: str, joint: str, sample: JointSample):
    if writer is None:
        return
    writer.writerow(
        {
            "t": f"{time.monotonic() - t0:.6f}",
            "phase": phase,
            "joint": joint,
            "q_motor": f"{sample.q_motor:.9f}",
            "q_model": f"{sample.q_model:.9f}",
            "dq_motor": f"{sample.dq_motor:.9f}",
            "dq_model": f"{sample.dq_model:.9f}",
            "tau": f"{sample.tau:.9f}",
            "temp_c": f"{sample.temp_c:.3f}",
        }
    )


def log_command_sample(
    writer,
    t0: float,
    phase: str,
    joint: str,
    sample: JointSample,
    q_model_cmd: float,
    q_motor_cmd: float,
    alpha: float | None = None,
):
    if writer is None:
        return
    writer.writerow(
        {
            "t": f"{time.monotonic() - t0:.6f}",
            "phase": phase,
            "joint": joint,
            "q_motor": f"{sample.q_motor:.9f}",
            "q_model": f"{sample.q_model:.9f}",
            "dq_motor": f"{sample.dq_motor:.9f}",
            "dq_model": f"{sample.dq_model:.9f}",
            "tau": f"{sample.tau:.9f}",
            "temp_c": f"{sample.temp_c:.3f}",
            "q_model_cmd": f"{q_model_cmd:.9f}",
            "q_motor_cmd": f"{q_motor_cmd:.9f}",
            "q_model_err": f"{sample.q_model - q_model_cmd:.9f}",
            "q_motor_err": f"{sample.q_motor - q_motor_cmd:.9f}",
            "alpha": "" if alpha is None else f"{alpha:.9f}",
        }
    )


def run_print(args, bus: DualBus, mapping: VBotRealJointAffine, ranges: RangeSpec, cache, stop):
    tx_period = 1.0 / args.tx_hz
    print_period = 1.0 / args.print_hz
    next_tx = 0.0
    next_print = 0.0
    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline and not stop["requested"]:
        now = time.monotonic()
        if args.zero_gain_poll and now >= next_tx:
            send_zero_gain_poll_all(bus, ranges, args)
            next_tx = now + tx_period
        bus.read_feedback(ranges, cache)
        if now >= next_print:
            print()
            print_feedback(mapping, cache)
            next_print = now + print_period
        time.sleep(0.002)


def run_pose_check(args, bus: DualBus, mapping: VBotRealJointAffine, ranges: RangeSpec, cache, stop):
    wait_feedback(bus, ranges, cache, args.duration, args, zero_gain_poll=args.zero_gain_poll, stop=stop)
    pose = load_model_pose(Path(args.model_poses), args.pose_check)
    print_pose_check(mapping, cache, pose)


def run_step_or_verify(args, bus: DualBus, mapping: VBotRealJointAffine, ranges: RangeSpec, cache, stop):
    require_joint(args.joint)
    wait_feedback(bus, ranges, cache, args.settle_time, args, zero_gain_poll=args.zero_gain_poll, stop=stop)
    before = sample_joint(mapping, cache, args.joint)

    target_joints = [args.joint] if args.single_only else list(mapping.order)
    targets = current_model_targets(mapping, cache, target_joints)
    targets[args.joint] = before.q_model + args.step_model
    q_motor_cmd = mapping.model_to_motor(args.joint, targets[args.joint])
    print(
        f"step target: {args.joint} q_model_0={before.q_model:+.6f}, "
        f"step={args.step_model:+.6f}, q_model_cmd={targets[args.joint]:+.6f}, "
        f"q_motor_cmd={q_motor_cmd:+.6f}, q_motor_delta_cmd={q_motor_cmd - before.q_motor:+.6f}"
    )
    print_targets(mapping, targets, cache)

    csv_file, writer = open_csv(args.log_csv)
    t0 = time.monotonic()
    log_sample(writer, t0, "before", args.joint, before)
    try:
        tx_period = 1.0 / args.tx_hz
        print_period = 1.0 / args.print_hz
        next_tx = 0.0
        next_print = 0.0
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline and not stop["requested"]:
            now = time.monotonic()
            if now >= next_tx:
                send_step_targets(bus, mapping, ranges, args, targets)
                next_tx = now + tx_period

            bus.read_feedback(ranges, cache)
            if cache[args.joint].seen:
                sample = sample_joint(mapping, cache, args.joint)
                log_command_sample(
                    writer,
                    t0,
                    "step",
                    args.joint,
                    sample,
                    targets[args.joint],
                    mapping.model_to_motor(args.joint, targets[args.joint]),
                )

            if now >= next_print:
                print()
                print_target_feedback(mapping, targets, cache)
                next_print = now + print_period
            time.sleep(0.001)

        after = sample_joint(mapping, cache, args.joint)
        log_sample(writer, t0, "after_step", args.joint, after)

        if args.verify_return and not stop["requested"]:
            deadline = time.monotonic() + args.return_duration
            while time.monotonic() < deadline and not stop["requested"]:
                send_joint_hold(bus, mapping, ranges, args, args.joint, before.q_model)
                bus.read_feedback(ranges, cache)
                if cache[args.joint].seen:
                    sample = sample_joint(mapping, cache, args.joint)
                    log_command_sample(
                        writer,
                        t0,
                        "return",
                        args.joint,
                        sample,
                        before.q_model,
                        mapping.model_to_motor(args.joint, before.q_model),
                    )
                time.sleep(tx_period)

        if args.mode == "verify":
            print_verify_result(args, before, after)
    finally:
        if csv_file is not None:
            csv_file.close()


def run_leg_pose(args, bus: DualBus, mapping: VBotRealJointAffine, ranges: RangeSpec, cache, stop):
    leg_joints = LEG_JOINTS[args.leg]
    wait_feedback(bus, ranges, cache, args.settle_time, args, zero_gain_poll=args.zero_gain_poll, stop=stop)

    current = current_model_targets(mapping, cache, leg_joints)
    pose = load_model_pose(Path(args.model_poses), args.pose)
    targets = {}
    for joint in leg_joints:
        delta = args.pose_alpha * (pose[joint] - current[joint])
        if args.max_pose_step > 0.0:
            delta = max(-args.max_pose_step, min(args.max_pose_step, delta))
        targets[joint] = current[joint] + delta

    print(f"leg-pose target: leg={args.leg}, pose={args.pose}, alpha={args.pose_alpha:.3f}")
    if args.max_pose_step > 0.0:
        print(f"max_pose_step: {args.max_pose_step:.3f} rad")
    print_targets(mapping, targets, cache)

    csv_file, writer = open_csv(args.log_csv)
    t0 = time.monotonic()
    for joint in leg_joints:
        log_sample(writer, t0, "before", joint, sample_joint(mapping, cache, joint))

    try:
        tx_period = 1.0 / args.tx_hz
        print_period = 1.0 / args.print_hz
        next_tx = 0.0
        next_print = 0.0
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline and not stop["requested"]:
            now = time.monotonic()
            if now >= next_tx:
                send_pose_targets(bus, mapping, ranges, args, targets)
                next_tx = now + tx_period

            bus.read_feedback(ranges, cache)
            for joint in leg_joints:
                if cache[joint].seen:
                    sample = sample_joint(mapping, cache, joint)
                    log_command_sample(
                        writer,
                        t0,
                        "leg_pose",
                        joint,
                        sample,
                        targets[joint],
                        mapping.model_to_motor(joint, targets[joint]),
                    )

            if now >= next_print:
                print()
                print_target_feedback(mapping, targets, cache)
                next_print = now + print_period
            time.sleep(0.001)

        print("\nLEG-POSE result")
        print(
            "joint                 "
            "q_model_0  q_model_cmd  q_model_end  model_err  "
            "q_motor_0  q_motor_cmd  q_motor_end  motor_err"
        )
        print("-" * 126)
        for joint in leg_joints:
            end = sample_joint(mapping, cache, joint)
            log_sample(writer, t0, "after_leg_pose", joint, end)

            q_motor_0 = mapping.model_to_motor(joint, current[joint])
            q_motor_cmd = mapping.model_to_motor(joint, targets[joint])
            q_motor_end = end.q_motor

            print(
                f"{joint:20s} "
                f"{current[joint]:+10.5f} {targets[joint]:+12.5f} "
                f"{end.q_model:+12.5f} {end.q_model - targets[joint]:+10.5f}  "
                f"{q_motor_0:+10.5f} {q_motor_cmd:+12.5f} "
                f"{q_motor_end:+12.5f} {q_motor_end - q_motor_cmd:+10.5f}"
            )
    finally:
        if csv_file is not None:
            csv_file.close()


def print_verify_result(args, before: JointSample, after: JointSample):
    commanded_delta = args.step_model
    observed_delta = after.q_model - before.q_model
    observed_motor_delta = after.q_motor - before.q_motor
    ratio = abs(observed_delta / commanded_delta) if abs(commanded_delta) > 1.0e-12 else 0.0
    sign_ok = observed_delta * commanded_delta > 0.0
    ratio_ok = args.min_ratio <= ratio <= args.max_ratio
    pass_ok = sign_ok and ratio_ok

    print("\nVERIFY result")
    print(f"joint               : {args.joint}")
    print(f"q_model_0           : {before.q_model:+.9f}")
    print(f"q_model_after_step  : {after.q_model:+.9f}")
    print(f"q_motor_0           : {before.q_motor:+.9f}")
    print(f"q_motor_after_step  : {after.q_motor:+.9f}")
    print(f"commanded_delta     : {commanded_delta:+.9f}")
    print(f"observed_delta      : {observed_delta:+.9f}")
    print(f"observed_motor_delta: {observed_motor_delta:+.9f}")
    print(f"abs ratio           : {ratio:.6f}")
    print(f"sign_ok             : {sign_ok}")
    print(f"ratio_ok            : {ratio_ok}")
    print(f"PASS                : {pass_ok}")


def smooth_alpha(x: float, profile: str) -> float:
    x = max(0.0, min(1.0, x))
    if profile == "linear":
        return x
    if profile == "smoothstep":
        return x * x * (3.0 - 2.0 * x)
    if profile == "smootherstep":
        return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
    raise RuntimeError(f"unknown ramp profile {profile!r}")


def interpolate_targets(a: dict[str, float], b: dict[str, float], alpha: float) -> dict[str, float]:
    return {joint: a[joint] + alpha * (b[joint] - a[joint]) for joint in a.keys()}


def run_pose_cycle(args, bus: DualBus, mapping: VBotRealJointAffine, ranges: RangeSpec, cache, stop):
    joints = LEG_JOINTS[args.leg]
    wait_feedback(bus, ranges, cache, args.settle_time, args, zero_gain_poll=args.zero_gain_poll, stop=stop)

    start_targets = current_model_targets(mapping, cache, joints)
    first_pose_all = load_model_pose(Path(args.model_poses), args.first_pose)
    second_pose_all = load_model_pose(Path(args.model_poses), args.second_pose)
    first_targets = {joint: first_pose_all[joint] for joint in joints}
    second_targets = {joint: second_pose_all[joint] for joint in joints}

    first_time = args.first_time if args.first_time > 0.0 else args.duration * args.first_ratio
    first_time = max(0.001, min(args.duration - 0.001, first_time))
    second_time = args.duration - first_time

    print(
        f"pose-cycle target: leg={args.leg}, {args.first_pose} for {first_time:.2f}s, "
        f"then {args.second_pose} for {second_time:.2f}s, profile={args.ramp_profile}"
    )
    print("first pose targets:")
    print_targets(mapping, first_targets, cache)
    print("second pose targets:")
    print_targets(mapping, second_targets, cache)

    csv_file, writer = open_csv(args.log_csv)
    t0 = time.monotonic()
    for joint in joints:
        log_sample(writer, t0, "before_cycle", joint, sample_joint(mapping, cache, joint))

    try:
        tx_period = 1.0 / args.tx_hz
        print_period = 1.0 / args.print_hz
        next_tx = 0.0
        next_print = 0.0
        cycle_start = time.monotonic()
        deadline = cycle_start + args.duration

        while time.monotonic() < deadline and not stop["requested"]:
            now = time.monotonic()
            elapsed = now - cycle_start

            if elapsed <= first_time:
                phase = "to_first_pose"
                alpha = smooth_alpha(elapsed / first_time, args.ramp_profile)
                targets = interpolate_targets(start_targets, first_targets, alpha)
            else:
                phase = "to_second_pose"
                alpha = smooth_alpha((elapsed - first_time) / second_time, args.ramp_profile)
                targets = interpolate_targets(first_targets, second_targets, alpha)

            if now >= next_tx:
                send_pose_targets(bus, mapping, ranges, args, targets)
                next_tx = now + tx_period

            bus.read_feedback(ranges, cache)
            for joint in joints:
                if cache[joint].seen:
                    sample = sample_joint(mapping, cache, joint)
                    log_command_sample(
                        writer,
                        t0,
                        phase,
                        joint,
                        sample,
                        targets[joint],
                        mapping.model_to_motor(joint, targets[joint]),
                        alpha=alpha,
                    )

            if now >= next_print:
                print(f"\nPOSE-CYCLE phase={phase} elapsed={elapsed:.2f}s alpha={alpha:.3f}")
                print_feedback(mapping, cache)
                next_print = now + print_period
            time.sleep(0.001)

        send_pose_targets(bus, mapping, ranges, args, second_targets)
        wait_feedback(bus, ranges, cache, 0.05, args, zero_gain_poll=False, stop=stop)

        print("\nPOSE-CYCLE result")
        print(
            "joint                 "
            "q_model_cmd  q_model_end  model_err  "
            "q_motor_cmd  q_motor_end  motor_err"
        )
        print("-" * 104)
        for joint in joints:
            end = sample_joint(mapping, cache, joint)
            log_sample(writer, t0, "after_cycle", joint, end)
            q_model_cmd = second_targets[joint]
            q_motor_cmd = mapping.model_to_motor(joint, q_model_cmd)
            print(
                f"{joint:20s} "
                f"{q_model_cmd:+12.5f} {end.q_model:+12.5f} {end.q_model - q_model_cmd:+10.5f}  "
                f"{q_motor_cmd:+12.5f} {end.q_motor:+12.5f} {end.q_motor - q_motor_cmd:+10.5f}"
            )
    finally:
        if csv_file is not None:
            csv_file.close()


def build_serial_arg_parser(
    description: str,
    fatu_serial_path: Path,
    include_pose_cycle: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--port-a", default="/dev/myttyCAN0")
    parser.add_argument("--port-b", default="/dev/myttyCAN1")
    parser.add_argument("--baudrate", type=int, default=2_000_000)
    parser.add_argument("--channel", type=lambda x: int(x, 0), default=0x00)
    parser.add_argument("--master-id", type=lambda x: int(x, 0), default=0x00FD)
    parser.add_argument("--fatudog-serial", default=str(fatu_serial_path))
    parser.add_argument("--affine", default=str(DEFAULT_AFFINE))
    parser.add_argument("--model-poses", default=str(DEFAULT_MODEL_POSES))
    modes = ["print", "pose-check", "step", "verify", "leg-pose"]
    if include_pose_cycle:
        modes.append("pose-cycle")
    parser.add_argument("--mode", choices=tuple(modes), default="print")

    parser.add_argument("--pose-check", choices=("stand", "down"), default="stand")
    parser.add_argument("--pose", choices=("stand", "down"), default="stand")
    if include_pose_cycle:
        parser.add_argument("--first-pose", choices=("stand", "down"), default="down")
        parser.add_argument("--second-pose", choices=("stand", "down"), default="stand")
        parser.add_argument(
            "--first-time",
            type=float,
            default=0.0,
            help="seconds for current -> first-pose in pose-cycle; 0 uses --first-ratio",
        )
        parser.add_argument(
            "--first-ratio",
            type=float,
            default=0.5,
            help="duration fraction for current -> first-pose when --first-time is 0",
        )
        parser.add_argument(
            "--ramp-profile",
            choices=("linear", "smoothstep", "smootherstep"),
            default="smootherstep",
        )
    parser.add_argument("--leg", choices=tuple(LEG_JOINTS.keys()), default="FR")
    parser.add_argument("--pose-alpha", type=float, default=0.25)
    parser.add_argument("--max-pose-step", type=float, default=0.20)
    parser.add_argument("--joint", choices=JOINT_ORDER, default="FR_hip_joint")
    parser.add_argument("--step-model", type=float, default=0.005)
    parser.add_argument("--kp", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--other-kp", type=float, default=0.0)
    parser.add_argument("--other-kd", type=float, default=0.0)
    parser.add_argument("--other-tau", type=float, default=0.0)
    parser.add_argument("--single-only", action="store_true", default=True)
    parser.add_argument("--all-joints-command", dest="single_only", action="store_false")

    parser.add_argument("--tx-hz", type=float, default=50.0)
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--return-duration", type=float, default=0.5)
    parser.add_argument("--zero-gain-poll", action="store_true", default=True)
    parser.add_argument("--no-zero-gain-poll", dest="zero_gain_poll", action="store_false")
    parser.add_argument("--send-enable", action="store_true")
    parser.add_argument("--enable-bursts", type=int, default=3)
    parser.add_argument("--disable-bursts", type=int, default=3)
    parser.add_argument("--motor-mode-interval", type=float, default=0.05)
    parser.add_argument("--enable-settle", type=float, default=0.1)
    parser.add_argument(
        "--enable-joint",
        choices=JOINT_ORDER,
        default=None,
        help="joint affected by --send-enable/--disable-on-exit; default: --joint when --single-only, otherwise all",
    )
    parser.add_argument("--disable-on-exit", action="store_true")
    parser.add_argument("--verify-return", action="store_true")
    parser.add_argument("--min-ratio", type=float, default=0.25)
    parser.add_argument("--max-ratio", type=float, default=1.75)
    parser.add_argument("--log-csv", default=None)
    parser.add_argument(
        "--lowstate-capture-bin",
        default=None,
        help="optional lowstate_capture_yaml executable to run alongside this test",
    )
    parser.add_argument(
        "--lowstate-capture-output",
        default=None,
        help="YAML output for --lowstate-capture-bin; default is logs/<mode>_lowstate_trajectory.yaml",
    )
    parser.add_argument("--lowstate-capture-network", default="lo")
    parser.add_argument("--lowstate-capture-hz", type=float, default=100.0)
    parser.add_argument("--lowstate-capture-extra-s", type=float, default=1.0)
    parser.add_argument("--lowstate-capture-pose", default=None)
    risk_modes = "step/verify/leg-pose/pose-cycle" if include_pose_cycle else "step/verify/leg-pose"
    parser.add_argument(
        "--i-accept-risk",
        action="store_true",
        help=f"required for --mode {risk_modes} or --send-enable",
    )
    return parser


def validate_serial_args(args, include_pose_cycle: bool = False):
    if args.tx_hz <= 0.0 or args.print_hz <= 0.0:
        raise RuntimeError("--tx-hz and --print-hz must be positive")
    if args.duration <= 0.0:
        raise RuntimeError("--duration must be positive")
    if args.settle_time < 0.0 or args.return_duration < 0.0:
        raise RuntimeError("--settle-time and --return-duration must be non-negative")
    if args.enable_bursts <= 0 or args.disable_bursts <= 0:
        raise RuntimeError("--enable-bursts and --disable-bursts must be positive")
    if args.motor_mode_interval < 0.0 or args.enable_settle < 0.0:
        raise RuntimeError("--motor-mode-interval and --enable-settle must be non-negative")
    if args.lowstate_capture_hz <= 0.0:
        raise RuntimeError("--lowstate-capture-hz must be positive")
    if args.lowstate_capture_extra_s < 0.0:
        raise RuntimeError("--lowstate-capture-extra-s must be non-negative")
    if args.mode in ("step", "verify") and abs(args.step_model) < 1.0e-12:
        raise RuntimeError("--step-model must be non-zero for step/verify")
    if not 0.0 <= args.pose_alpha <= 1.0:
        raise RuntimeError("--pose-alpha must be in [0, 1]")
    if args.max_pose_step < 0.0:
        raise RuntimeError("--max-pose-step must be non-negative")

    grouped_modes = ["leg-pose"]
    risk_modes = ["step", "verify", "leg-pose"]
    if include_pose_cycle:
        grouped_modes.append("pose-cycle")
        risk_modes.append("pose-cycle")
        if not 0.0 < args.first_ratio < 1.0:
            raise RuntimeError("--first-ratio must be in (0, 1)")
        if args.first_time < 0.0:
            raise RuntimeError("--first-time must be non-negative")
        if args.mode == "pose-cycle" and args.duration <= 0.1:
            raise RuntimeError("--duration must be larger than 0.1 for pose-cycle")

    if args.mode in grouped_modes and args.enable_joint is not None:
        modes = "/".join(grouped_modes)
        raise RuntimeError(f"--enable-joint is not supported with --mode {modes}; the selected leg is enabled")
    if (args.mode in risk_modes or args.send_enable) and not args.i_accept_risk:
        modes = "/".join(risk_modes)
        raise RuntimeError(f"--mode {modes} or --send-enable requires --i-accept-risk")


def install_stop_handlers():
    stop = {"requested": False}

    def on_signal(_signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    return stop


def default_lowstate_capture_output(args) -> str:
    return str(Path("logs") / f"{args.mode}_lowstate_trajectory.yaml")


def start_lowstate_capture(args):
    if not args.lowstate_capture_bin:
        return None

    output = args.lowstate_capture_output or default_lowstate_capture_output(args)
    pose_name = args.lowstate_capture_pose or f"{args.mode}_trajectory"
    duration = args.duration + args.lowstate_capture_extra_s
    command = [
        str(Path(args.lowstate_capture_bin).expanduser()),
        "--network",
        args.lowstate_capture_network,
        "--pose",
        pose_name,
        "--output",
        output,
        "--sample-hz",
        str(args.lowstate_capture_hz),
        "--duration",
        str(duration),
        "--trajectory",
        "--note",
        f"{args.mode} capture from EX24/serial mapping test",
    ]
    Path(output).expanduser().parent.mkdir(parents=True, exist_ok=True)
    print(f"lowstate capture: {' '.join(command)}")
    return subprocess.Popen(command)


def stop_lowstate_capture(process):
    if process is None:
        return
    try:
        code = process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            code = process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
    if code != 0:
        print(f"WARNING: lowstate_capture_yaml exited with code {code}")


def run_serial_mapping_tool(
    args,
    fatu_serial_path: Path,
    include_pose_cycle: bool = False,
    sign_only: bool = False,
) -> int:
    mapping = load_serial_affine(Path(args.affine), sign_only=sign_only)
    ranges = RangeSpec()
    cache = {joint: Feedback() for joint in JOINT_ORDER}
    stop = install_stop_handlers()

    bus = DualBus(args.port_a, args.port_b, args.baudrate)
    capture_process = None
    try:
        print(f"affine: {mapping.path}")
        print(f"fatuDog serial helpers: {fatu_serial_path}")
        print(f"port A: {args.port_a}  FR/RR")
        print(f"port B: {args.port_b}  FL/RL")
        print(f"mode: {args.mode}")
        capture_process = start_lowstate_capture(args)

        if args.send_enable:
            send_enable_disable(bus, args, LINGZU_MOTOR_ENABLE_CODE)

        if args.mode == "print":
            run_print(args, bus, mapping, ranges, cache, stop)
        elif args.mode == "pose-check":
            run_pose_check(args, bus, mapping, ranges, cache, stop)
        elif args.mode in ("step", "verify"):
            run_step_or_verify(args, bus, mapping, ranges, cache, stop)
        elif args.mode == "leg-pose":
            run_leg_pose(args, bus, mapping, ranges, cache, stop)
        elif include_pose_cycle and args.mode == "pose-cycle":
            run_pose_cycle(args, bus, mapping, ranges, cache, stop)
        else:
            raise RuntimeError(f"unsupported mode {args.mode!r}")

        if stop["requested"]:
            return 130
        return 0
    finally:
        if args.disable_on_exit:
            send_enable_disable(bus, args, LINGZU_MOTOR_DISABLE_CODE)
        stop_lowstate_capture(capture_process)
        bus.close()
