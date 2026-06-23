"""EX31B: suspended low-torque DDS VBot MPC test.

This is the first DDS real-control test. It subscribes to ``rt/lowstate``,
maps motor feedback into model coordinates, computes MPC torque, maps model
torque back to motor torque, clips/ramp-limits it, and publishes ``rt/lowcmd``.

Use only with the robot suspended or firmly supported.
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
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


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


def update_vbot_yaw_free_from_lowstate(vbot, bridge, cache, args):
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
    return tau_model, tau_motor_raw


def limit_and_ramp_tau(tau_motor_raw, args, elapsed_s):
    import numpy as np

    clipped = np.clip(tau_motor_raw, -args.tau_limit, args.tau_limit)
    alpha = 1.0 if args.ramp_seconds <= 1.0e-9 else max(0.0, min(1.0, elapsed_s / args.ramp_seconds))
    cmd = alpha * clipped
    clipped_count = int(np.count_nonzero(np.abs(tau_motor_raw) > args.tau_limit + 1.0e-12))
    return cmd, clipped, alpha, clipped_count


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


def make_lowcmd(mode: int, tau_motor=None):
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
        motor.tau = float(tau_motor[i]) if tau_motor is not None and i < 12 else 0.0
    return cmd


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


def publish_zero_for(pub, crc, hz: float, seconds: float, mode: int = 1, stop=None):
    cmd = make_lowcmd(mode=mode)
    period = 1.0 / hz
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline and not (stop and stop["requested"]):
        now = time.monotonic()
        publish_lowcmd(pub, cmd, crc)
        sleep_s = period - (time.monotonic() - now)
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def print_torque_table(
    bridge,
    cache,
    tau_model,
    tau_motor_raw,
    tau_motor_clipped,
    tau_motor_cmd,
    elapsed_s,
    mpc,
    alpha,
    clipped_count,
):
    import numpy as np

    q_motor = cache_to_motor_dict(cache, bridge.order, "q")
    dq_motor = cache_to_motor_dict(cache, bridge.order, "dq")
    q_model = bridge.motor_feedback_to_model_dict(q_motor)
    dq_model = bridge.motor_feedback_to_model_velocity_dict(dq_motor)
    tau_model_by_joint = {joint: float(value) for joint, value in zip(MPC_JOINT_ORDER, tau_model)}
    max_raw_idx = int(np.argmax(np.abs(tau_motor_raw)))
    max_cmd_idx = int(np.argmax(np.abs(tau_motor_cmd)))
    print(
        f"\nt={elapsed_s:.3f}s  source=DDS-lowstate yaw_free  "
        f"ramp={alpha:.2f}  clipped={clipped_count}/{len(bridge.order)}  "
        f"mpc_solve={getattr(mpc, 'solve_time', 0.0):.2f}ms  "
        f"max_raw={bridge.order[max_raw_idx]}:{float(tau_motor_raw[max_raw_idx]):+.3f}  "
        f"max_cmd={bridge.order[max_cmd_idx]}:{float(tau_motor_cmd[max_cmd_idx]):+.3f}"
    )
    print("joint                 q_model   dq_model  tau_model  tau_raw  tau_clip   tau_cmd  tau_est")
    print("-" * 96)
    for i, joint in enumerate(bridge.order):
        print(
            f"{joint:20s} {q_model[joint]:+8.4f} {dq_model[joint]:+9.4f} "
            f"{tau_model_by_joint[joint]:+10.3f} {float(tau_motor_raw[i]):+8.3f} "
            f"{float(tau_motor_clipped[i]):+9.3f} {float(tau_motor_cmd[i]):+9.3f} "
            f"{float(cache[joint]['tau_est']):+8.3f}"
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Suspended low-torque DDS VBot MPC hardware test"
    )
    parser.add_argument("--network", default="lo")
    parser.add_argument("--affine", default=str(DEFAULT_AFFINE))
    parser.add_argument("--model-poses", default=str(DEFAULT_MODEL_POSES))
    parser.add_argument("--prone-calibrate-on-start", action="store_true")
    parser.add_argument("--prone-pose", default="down")

    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--cmd-hz", type=float, default=100.0)
    parser.add_argument("--mpc-hz", type=float, default=10.0)
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--wait-lowstate-s", type=float, default=5.0)
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

    parser.add_argument("--tau-limit", type=float, default=0.20)
    parser.add_argument("--allow-large-tau-limit", action="store_true")
    parser.add_argument("--ramp-seconds", type=float, default=1.0)
    parser.add_argument("--prezero-seconds", type=float, default=0.5)
    parser.add_argument("--zero-on-exit-seconds", type=float, default=0.8)
    parser.add_argument("--disable-on-exit", action="store_true", default=True)
    parser.add_argument("--no-disable-on-exit", action="store_false", dest="disable_on_exit")

    parser.add_argument("--robot-is-suspended", action="store_true")
    parser.add_argument("--i-accept-risk", action="store_true")
    parser.add_argument("--allow-long-duration", action="store_true")
    return parser


def validate_args(args):
    if not args.robot_is_suspended:
        raise RuntimeError("EX31B requires --robot-is-suspended")
    if not args.i_accept_risk:
        raise RuntimeError("EX31B sends nonzero torque; pass --i-accept-risk")
    if args.duration <= 0.0:
        raise RuntimeError("--duration must be positive")
    if args.duration > 10.0 and not args.allow_long_duration:
        raise RuntimeError("--duration > 10s requires --allow-long-duration")
    if args.cmd_hz <= 0.0 or args.mpc_hz <= 0.0 or args.print_hz < 0.0:
        raise RuntimeError("--cmd-hz/--mpc-hz must be positive and --print-hz non-negative")
    if args.wait_lowstate_s <= 0.0 or args.feedback_timeout_s <= 0.0:
        raise RuntimeError("--wait-lowstate-s and --feedback-timeout-s must be positive")
    if args.gait_hz <= 0.0 or not 0.0 < args.gait_duty < 1.0:
        raise RuntimeError("--gait-hz must be positive and --gait-duty must be in (0, 1)")
    if args.horizon_segments <= 0:
        raise RuntimeError("--horizon-segments must be positive")
    if args.yaw_weight < 0.0 or args.yaw_rate_weight < 0.0:
        raise RuntimeError("--yaw-weight and --yaw-rate-weight must be non-negative")
    if args.tau_limit <= 0.0:
        raise RuntimeError("--tau-limit must be positive")
    if args.tau_limit > 1.0 and not args.allow_large_tau_limit:
        raise RuntimeError("--tau-limit > 1.0 Nm requires --allow-large-tau-limit")
    if args.ramp_seconds < 0.0 or args.prezero_seconds < 0.0 or args.zero_on_exit_seconds < 0.0:
        raise RuntimeError("--ramp/zero timing values must be non-negative")


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)

    import numpy as np

    from convex_mpc import centroidal_mpc as centroidal_mpc_module
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
    mpc_q = centroidal_mpc_module.COST_MATRIX_Q.copy()
    mpc_q[5, 5] = args.yaw_weight
    mpc_q[11, 11] = args.yaw_rate_weight
    centroidal_mpc_module.COST_MATRIX_Q = mpc_q
    CentroidalMPC = centroidal_mpc_module.CentroidalMPC

    bridge = VBotRealJointAffine.from_yaml(args.affine)
    latest = {"msg": None, "t": 0.0}
    stop = install_stop_handlers()

    def on_lowstate(msg):
        latest["msg"] = msg
        latest["t"] = time.monotonic()

    print("EX31B DDS real MPC suspended low-torque test")
    print(f"network: {args.network}")
    print(f"affine: {Path(args.affine)}")
    print(f"tau_limit={args.tau_limit:.3f} Nm  duration={args.duration:.2f}s")
    print("WARNING: this publishes nonzero motor torque. Use only while suspended/supported.")
    print("Yaw-free fixed base state: yaw=0, wz=0, x_vel=y_vel=yaw_rate=0.")

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

        print(f"Publishing mode=0 zero command for {args.prezero_seconds:.2f}s...")
        publish_zero_for(pub, crc, args.cmd_hz, args.prezero_seconds, mode=0, stop=stop)

        print("Waiting for non-lost motor feedback with zero torque...")
        deadline = time.monotonic() + args.wait_lowstate_s
        first_cache = None
        period = 1.0 / args.cmd_hz
        next_cmd = 0.0
        while time.monotonic() < deadline and not stop["requested"]:
            now = time.monotonic()
            if now >= next_cmd:
                publish_lowcmd(pub, make_lowcmd(mode=1), crc)
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

        vbot = PinVBotModel()
        update_vbot_yaw_free_from_lowstate(vbot, bridge, first_cache, args)
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
        alpha = 0.0
        clipped_count = 0

        cmd_period = 1.0 / args.cmd_hz
        mpc_period = 1.0 / args.mpc_hz
        print_period = math.inf if args.print_hz <= 0.0 else 1.0 / args.print_hz
        start_t = time.monotonic()
        run_deadline = start_t + args.duration
        next_cmd = 0.0
        next_mpc = 0.0
        next_print = 0.0

        while time.monotonic() < run_deadline and not stop["requested"]:
            now = time.monotonic()
            elapsed = now - start_t

            if now - latest["t"] > args.feedback_timeout_s:
                raise RuntimeError(f"rt/lowstate timeout > {args.feedback_timeout_s:.3f}s")

            cache = read_lowstate_motor_feedback(latest["msg"], bridge.order)
            lost = lost_joints(cache, bridge.order)
            if lost:
                raise RuntimeError(f"motor feedback lost: {', '.join(lost)}")

            if now >= next_mpc:
                update_vbot_yaw_free_from_lowstate(vbot, bridge, cache, args)
                tau_model, tau_motor_raw = compute_yaw_free_tau(
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
                publish_lowcmd(pub, make_lowcmd(mode=1, tau_motor=tau_motor_cmd), crc)
                next_cmd = now + cmd_period

            if now >= next_print:
                print_torque_table(
                    bridge,
                    cache,
                    tau_model,
                    tau_motor_raw,
                    tau_motor_clipped,
                    tau_motor_cmd,
                    elapsed,
                    mpc,
                    alpha,
                    clipped_count,
                )
                next_print = now + print_period

            time.sleep(0.001)

        return 130 if stop["requested"] else 0
    finally:
        print(f"\nPublishing zero torque for {args.zero_on_exit_seconds:.2f}s...")
        publish_zero_for(pub, crc, args.cmd_hz, args.zero_on_exit_seconds, mode=1, stop=None)
        if args.disable_on_exit:
            print("Publishing mode=0 disable command on exit...")
            publish_zero_for(pub, crc, args.cmd_hz, args.zero_on_exit_seconds, mode=0, stop=None)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
