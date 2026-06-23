"""Solve real-motor to model-joint affine calibration.

Given real down/stand captures and model down/stand poses, solve

    q_model = scale * q_motor - bias
    dq_model = scale * dq_motor
    q_motor_cmd = (q_model_cmd + bias) / scale

The output is written as YAML for later real-robot adapters.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_POSE_FILE = REPO / "configs" / "vbot_model_poses.yaml"
DEFAULT_OUTPUT = REPO / "configs" / "vbot_real_joint_affine.yaml"

DDS_JOINT_ORDER = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)


def load_yaml(path: Path):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to solve calibration") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to write calibration") from exc
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_capture(path: Path, pose_name: str) -> dict[str, float]:
    raw = load_yaml(path)
    order = raw["dds_joint_order"]
    if tuple(order) != DDS_JOINT_ORDER:
        raise RuntimeError(f"{path} joint order does not match expected DDS order")
    pose = raw["poses"][pose_name]
    if any(int(value) != 0 for value in pose["lost"]):
        raise RuntimeError(f"{path} pose {pose_name} has lost != 0")
    return dict(zip(DDS_JOINT_ORDER, [float(value) for value in pose["q_motor"]]))


def require_pose(poses, name: str) -> dict[str, float]:
    if name not in poses:
        raise RuntimeError(f"model pose {name!r} missing")
    pose = {joint: float(value) for joint, value in poses[name].items()}
    missing = [joint for joint in DDS_JOINT_ORDER if joint not in pose]
    if missing:
        raise RuntimeError(f"model pose {name!r} missing joints: {', '.join(missing)}")
    return pose


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve VBot real joint affine calibration")
    parser.add_argument("--real-down", default=str(REPO.parent / "down.yaml"))
    parser.add_argument("--real-stand", default=str(REPO.parent / "stand.yaml"))
    parser.add_argument("--model-poses", default=str(DEFAULT_POSE_FILE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--min-motor-delta", type=float, default=0.02)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    real_down = load_capture(Path(args.real_down), "down")
    real_stand = load_capture(Path(args.real_stand), "stand")
    model_poses = load_yaml(Path(args.model_poses))
    model_down = require_pose(model_poses, "down")
    model_stand = require_pose(model_poses, "stand")

    joints = {}
    print("joint                 scale       bias        motor_delta  model_delta")
    print("-" * 76)
    for joint in DDS_JOINT_ORDER:
        motor_delta = real_down[joint] - real_stand[joint]
        model_delta = model_down[joint] - model_stand[joint]
        if abs(motor_delta) < args.min_motor_delta:
            raise RuntimeError(f"{joint}: motor delta too small ({motor_delta:+.6f})")
        scale = model_delta / motor_delta
        bias = scale * real_stand[joint] - model_stand[joint]
        joints[joint] = {
            "scale": float(scale),
            "bias": float(bias),
            "q_motor_stand": float(real_stand[joint]),
            "q_motor_down": float(real_down[joint]),
            "q_model_stand": float(model_stand[joint]),
            "q_model_down": float(model_down[joint]),
        }
        print(f"{joint:20s} {scale:+10.6f} {bias:+10.6f} {motor_delta:+12.6f} {model_delta:+12.6f}")

    output = {
        "convention": {
            "q_model": "scale * q_motor - bias",
            "dq_model": "scale * dq_motor",
            "q_motor_cmd": "(q_model_cmd + bias) / scale",
        },
        "dds_joint_order": list(DDS_JOINT_ORDER),
        "real_down": str(Path(args.real_down)),
        "real_stand": str(Path(args.real_stand)),
        "model_poses": str(Path(args.model_poses)),
        "joints": joints,
    }

    if args.write:
        save_yaml(Path(args.output), output)
        print(f"\nWrote affine calibration to {args.output}")
    else:
        print("\nDry run only. Add --write to save YAML.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
