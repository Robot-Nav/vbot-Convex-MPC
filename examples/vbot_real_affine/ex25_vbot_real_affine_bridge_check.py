"""Print VBot real-affine bridge conventions for hardware MPC bring-up.

This script is read-only. It does not open serial ports and does not publish
motor commands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "convex_mpc"
DEFAULT_AFFINE = REPO / "configs" / "vbot_real_joint_affine.yaml"
DEFAULT_POSES = REPO / "configs" / "vbot_model_poses.yaml"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vbot_real_affine import (  # noqa: E402
    DDS_JOINT_ORDER,
    MPC_JOINT_ORDER,
    PIN_JOINT_ORDER,
    VBotRealJointAffine,
)


def load_yaml(path: Path):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_orders():
    print("Vector orders")
    print("  DDS/CAN feedback and motor command order:")
    print("   ", ", ".join(DDS_JOINT_ORDER))
    print("  Pinocchio joint order after floating base q[7:19] / dq[6:18]:")
    print("   ", ", ".join(PIN_JOINT_ORDER))
    print("  MPC torque vector order:")
    print("   ", ", ".join(MPC_JOINT_ORDER))


def print_affine_table(bridge: VBotRealJointAffine, raw):
    print("\nAffine and torque transform")
    print(
        "joint                   scale       bias     tau_motor = scale * tau_model"
    )
    print("-" * 76)
    for joint in bridge.order:
        affine = bridge.joints[joint]
        cfg = raw["joints"][joint]
        q_motor_stand = cfg.get("q_motor_stand", cfg.get("calib_q_motor_stand"))
        q_motor_down = cfg.get("q_motor_down", cfg.get("calib_q_motor_down"))
        q_model_stand_ref = cfg.get("q_model_stand", cfg.get("calib_q_model_stand"))
        q_model_down_ref = cfg.get("q_model_down", cfg.get("calib_q_model_down"))
        if None in (q_motor_stand, q_motor_down, q_model_stand_ref, q_model_down_ref):
            stand_err_text = "stand_err=n/a"
            down_err_text = "down_err=n/a"
        else:
            q_model_stand = affine.motor_to_model(float(q_motor_stand))
            q_model_down = affine.motor_to_model(float(q_motor_down))
            stand_err = q_model_stand - float(q_model_stand_ref)
            down_err = q_model_down - float(q_model_down_ref)
            stand_err_text = f"stand_err={stand_err:+.2e}"
            down_err_text = f"down_err={down_err:+.2e}"
        sign = "flip" if affine.scale < 0.0 else "same"
        print(
            f"{joint:20s} {affine.scale:+10.6f} {affine.bias:+10.6f}"
            f"   {sign:4s}   {stand_err_text} {down_err_text}"
        )


def print_pose_vectors(bridge: VBotRealJointAffine, pose_file: Path, pose_name: str):
    poses = load_yaml(pose_file)
    if pose_name not in poses:
        raise RuntimeError(f"pose {pose_name!r} not found in {pose_file}")
    pose = {joint: float(value) for joint, value in poses[pose_name].items()}
    missing = [joint for joint in DDS_JOINT_ORDER if joint not in pose]
    if missing:
        raise RuntimeError(f"pose {pose_name!r} is missing joints: {', '.join(missing)}")

    q_motor = bridge.motor_position_commands_from_model_dict(pose)
    print(f"\nPose {pose_name!r}")
    print("  model q in DDS/CAN order:")
    print("   ", " ".join(f"{pose[joint]:+.5f}" for joint in DDS_JOINT_ORDER))
    print("  motor q command in DDS/CAN order:")
    print("   ", " ".join(f"{q_motor[joint]:+.5f}" for joint in DDS_JOINT_ORDER))
    print("  model q in Pinocchio joint order:")
    print("   ", " ".join(f"{pose[joint]:+.5f}" for joint in PIN_JOINT_ORDER))


def print_tau_example(bridge: VBotRealJointAffine):
    tau_model = [1.0] * 12
    tau_motor = bridge.motor_torque_commands_from_mpc(tau_model)
    print("\nTorque sanity example")
    print("  If tau_model_mpc_order is all +1 Nm, motor tau command in DDS/CAN order is:")
    print("   ", " ".join(f"{value:+.5f}" for value in tau_motor))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check VBot real affine bridge conventions")
    parser.add_argument("--affine", default=str(DEFAULT_AFFINE))
    parser.add_argument("--pose-file", default=str(DEFAULT_POSES))
    parser.add_argument("--pose", choices=("stand", "down"), default="stand")
    parser.add_argument("--no-pose", action="store_true")
    args = parser.parse_args()

    affine_path = Path(args.affine)
    raw = load_yaml(affine_path)
    bridge = VBotRealJointAffine.from_yaml(affine_path)

    print_orders()
    print_affine_table(bridge, raw)
    print_tau_example(bridge)
    if not args.no_pose:
        print_pose_vectors(bridge, Path(args.pose_file), args.pose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
