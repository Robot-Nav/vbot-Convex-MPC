"""EX30B: monitor-only DDS real VBot MPC bridge.

This is the DDS counterpart of EX30. It subscribes to ``rt/lowstate``, maps real
motor feedback into model coordinates, updates the Pinocchio VBot model, solves
MPC, and prints model-space and motor-space torque commands.

It never publishes MPC torque. If ``--publish-zero-poll`` is enabled, it only
publishes zero-gain, zero-torque ``rt/lowcmd`` frames to make the C++
``dds_to_serial_gateway`` poll motor feedback:

    mode=1, q=0, dq=0, kp=0, kd=0, tau=0

Use this while the robot is safely supported. Stop immediately if the gateway
or robot behaves unexpectedly.
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

from vbot_real_affine import MPC_JOINT_ORDER, VBotRealJointAffine, load_model_pose  # noqa: E402


MPC_LEG_ORDER = ("FL", "FR", "RL", "RR")
LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}


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
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


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


def cache_to_motor_dict(cache, order, key: str) -> dict[str, float]:
    return {joint: float(cache[joint][key]) for joint in order}


def lost_joints(cache, order) -> list[str]:
    return [joint for joint in order if int(cache[joint]["lost"]) != 0]


def update_vbot_from_lowstate(vbot, bridge, cache, args):
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


def print_monitor_table(bridge, cache, tau_model, tau_motor, elapsed_s, mpc):
    import numpy as np

    q_motor = cache_to_motor_dict(cache, bridge.order, "q")
    dq_motor = cache_to_motor_dict(cache, bridge.order, "dq")
    q_model = bridge.motor_feedback_to_model_dict(q_motor)
    dq_model = bridge.motor_feedback_to_model_velocity_dict(dq_motor)
    tau_model_by_joint = {
        joint: float(value) for joint, value in zip(MPC_JOINT_ORDER, tau_model)
    }
    max_model_joint = max(tau_model_by_joint, key=lambda joint: abs(tau_model_by_joint[joint]))
    max_motor_idx = int(np.argmax(np.abs(tau_motor)))
    max_motor_joint = bridge.order[max_motor_idx]
    lost = lost_joints(cache, bridge.order)

    print(
        f"\nt={elapsed_s:.3f}s  source=DDS-lowstate  "
        f"lost={len(lost)}/{len(bridge.order)}  "
        f"mpc_update={getattr(mpc, 'update_time', 0.0):.2f}ms  "
        f"mpc_solve={getattr(mpc, 'solve_time', 0.0):.2f}ms  "
        f"max_tau_model={max_model_joint}:{tau_model_by_joint[max_model_joint]:+.3f}  "
        f"max_tau_motor={max_motor_joint}:{float(tau_motor[max_motor_idx]):+.3f}"
    )
    if lost:
        print(f"lost_joints={', '.join(lost)}")
    print("joint                 q_model   dq_model  tau_model  tau_motor  q_motor lost")
    print("-" * 86)
    for i, joint in enumerate(bridge.order):
        print(
            f"{joint:20s} {q_model[joint]:+8.4f} {dq_model[joint]:+9.4f} "
            f"{tau_model_by_joint[joint]:+10.3f} {float(tau_motor[i]):+10.3f} "
            f"{q_motor[joint]:+8.4f} {int(cache[joint]['lost']):4d}"
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


def publish_lowcmd(pub, cmd, crc):
    maybe_crc(cmd, crc)
    pub.Write(cmd)


def publish_for(pub, cmd, crc, hz: float, seconds: float):
    if seconds <= 0.0:
        publish_lowcmd(pub, cmd, crc)
        return
    period = 1.0 / hz
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        now = time.monotonic()
        publish_lowcmd(pub, cmd, crc)
        sleep_s = period - (time.monotonic() - now)
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Monitor DDS lowstate through VBot MPC without publishing MPC torque"
    )
    parser.add_argument("--network", default="lo")
    parser.add_argument("--affine", default=str(DEFAULT_AFFINE))
    parser.add_argument("--model-poses", default=str(DEFAULT_MODEL_POSES))
    parser.add_argument(
        "--prone-calibrate-on-start",
        action="store_true",
        help="after first non-lost lowstate, map current motor pose to --prone-pose",
    )
    parser.add_argument("--prone-pose", default="down")

    parser.add_argument("--duration", type=float, default=10.0, help="seconds; <=0 means forever")
    parser.add_argument("--mpc-hz", type=float, default=10.0)
    parser.add_argument("--print-hz", type=float, default=1.0)
    parser.add_argument("--wait-lowstate-s", type=float, default=5.0)
    parser.add_argument("--feedback-timeout-s", type=float, default=0.5)

    parser.add_argument("--publish-zero-poll", action="store_true")
    parser.add_argument("--cmd-hz", type=float, default=100.0)
    parser.add_argument("--prezero-seconds", type=float, default=0.25)
    parser.add_argument("--disable-on-exit", action="store_true", default=True)
    parser.add_argument("--disable-seconds", type=float, default=0.5)
    parser.add_argument("--robot-supported", action="store_true")
    parser.add_argument("--i-accept-risk", action="store_true")

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

    parser.add_argument("--x-vel", type=float, default=0.0)
    parser.add_argument("--y-vel", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--gait-hz", type=float, default=3.0)
    parser.add_argument("--gait-duty", type=float, default=0.6)
    parser.add_argument("--horizon-segments", type=int, default=16)
    parser.add_argument("--verbose-mpc", action="store_true")
    return parser


def validate_args(args):
    if args.mpc_hz <= 0.0 or args.print_hz < 0.0 or args.cmd_hz <= 0.0:
        raise RuntimeError("--mpc-hz/--cmd-hz must be positive and --print-hz non-negative")
    if args.wait_lowstate_s <= 0.0 or args.feedback_timeout_s <= 0.0:
        raise RuntimeError("--wait-lowstate-s and --feedback-timeout-s must be positive")
    if args.gait_hz <= 0.0 or not 0.0 < args.gait_duty < 1.0:
        raise RuntimeError("--gait-hz must be positive and --gait-duty must be in (0, 1)")
    if args.horizon_segments <= 0:
        raise RuntimeError("--horizon-segments must be positive")
    if args.prezero_seconds < 0.0 or args.disable_seconds < 0.0:
        raise RuntimeError("--prezero-seconds/--disable-seconds must be non-negative")
    if args.publish_zero_poll and not args.robot_supported:
        raise RuntimeError("--publish-zero-poll requires --robot-supported")
    if args.publish_zero_poll and not args.i_accept_risk:
        raise RuntimeError("--publish-zero-poll can enable motors through the gateway; pass --i-accept-risk")


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)

    import numpy as np

    from convex_mpc.centroidal_mpc import CentroidalMPC
    from convex_mpc.com_trajectory import ComTraj
    from convex_mpc.gait import Gait
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

    bridge = VBotRealJointAffine.from_yaml(args.affine)
    latest = {"msg": None, "t": 0.0}
    stop = install_stop_handlers()

    def on_lowstate(msg):
        latest["msg"] = msg
        latest["t"] = time.monotonic()

    print("EX30B DDS real MPC monitor: computes MPC torque, publishes no MPC torque")
    print(f"network: {args.network}")
    print(f"affine: {Path(args.affine)}")
    print(
        "WARNING: base pose/velocity are fixed command-line values in this monitor; "
        "IMU base feedback is not connected to MPC state yet."
    )
    if args.publish_zero_poll:
        print("zero-poll is enabled: publishing mode=1,q=0,dq=0,kp=0,kd=0,tau=0 lowcmd frames.")
    else:
        print("zero-poll is disabled: expecting gateway lowstate feedback to already be live.")

    ChannelFactoryInitialize(0, args.network)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_lowstate, 1)
    pub = None
    cmd_poll = None
    cmd_disable = None
    crc = None
    if args.publish_zero_poll:
        pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        pub.Init()
        cmd_poll = make_lowcmd(mode=1)
        cmd_disable = make_lowcmd(mode=0)
        crc = make_crc()

    try:
        deadline = time.monotonic() + args.wait_lowstate_s
        while latest["msg"] is None and time.monotonic() < deadline and not stop["requested"]:
            time.sleep(0.02)
        if latest["msg"] is None:
            raise RuntimeError("no rt/lowstate received; start dds_to_serial_gateway first")

        if args.publish_zero_poll and args.prezero_seconds > 0.0:
            print(f"Publishing mode=0 for {args.prezero_seconds:.2f}s before zero-poll...")
            publish_for(pub, cmd_disable, crc, args.cmd_hz, args.prezero_seconds)

        print("Waiting for non-lost motor feedback...")
        deadline = time.monotonic() + args.wait_lowstate_s
        first_cache = None
        next_poll = 0.0
        cmd_period = 1.0 / args.cmd_hz
        while time.monotonic() < deadline and not stop["requested"]:
            now = time.monotonic()
            if args.publish_zero_poll and now >= next_poll:
                publish_lowcmd(pub, cmd_poll, crc)
                next_poll = now + cmd_period
            msg = latest["msg"]
            if msg is not None:
                cache = read_lowstate_motor_feedback(msg, bridge.order)
                if not lost_joints(cache, bridge.order):
                    first_cache = cache
                    break
            time.sleep(0.002)
        if first_cache is None:
            raise RuntimeError("motor feedback is still lost; run EX29B and check gateway type2_frames")

        bridge = calibrate_prone_anchor_if_requested(bridge, first_cache, args)

        vbot = PinVBotModel()
        update_vbot_from_lowstate(vbot, bridge, first_cache, args)

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

        mpc_period = 1.0 / args.mpc_hz
        print_period = math.inf if args.print_hz <= 0.0 else 1.0 / args.print_hz
        start_t = time.monotonic()
        run_deadline = math.inf if args.duration <= 0.0 else start_t + args.duration
        next_mpc = 0.0
        next_print = 0.0
        next_cmd = 0.0

        while time.monotonic() < run_deadline and not stop["requested"]:
            now = time.monotonic()
            elapsed = now - start_t

            if args.publish_zero_poll and now >= next_cmd:
                publish_lowcmd(pub, cmd_poll, crc)
                next_cmd = now + cmd_period

            if now - latest["t"] > args.feedback_timeout_s:
                raise RuntimeError(f"rt/lowstate timeout > {args.feedback_timeout_s:.3f}s")

            msg = latest["msg"]
            cache = read_lowstate_motor_feedback(msg, bridge.order)
            lost = lost_joints(cache, bridge.order)
            if lost:
                raise RuntimeError(f"motor feedback lost: {', '.join(lost)}")

            if now >= next_mpc:
                update_vbot_from_lowstate(vbot, bridge, cache, args)
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
                print_monitor_table(bridge, cache, tau_model, tau_motor, elapsed, mpc)
                next_print = now + print_period

            time.sleep(0.001)

        return 130 if stop["requested"] else 0
    finally:
        if args.publish_zero_poll and args.disable_on_exit:
            print(f"\nPublishing mode=0 for {args.disable_seconds:.2f}s on exit...")
            publish_for(pub, cmd_disable, crc, args.cmd_hz, args.disable_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
