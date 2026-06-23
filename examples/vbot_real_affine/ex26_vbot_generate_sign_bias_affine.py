"""Generate VBot real-joint sign+bias mapping.

This generator assumes the real motor feedback is already in joint-angle units,
so the mapping scale only encodes direction:

    q_model = sign * q_motor + bias

where sign is +1 or -1.  The sign is inferred from the stand->down delta, and
the bias is shared by joint type (hip/thigh/calf) by averaging per-joint bias
estimates.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
DEFAULT_REAL_STAND = WORKSPACE / "stand.yaml"
DEFAULT_REAL_DOWN = WORKSPACE / "down.yaml"
DEFAULT_MODEL_POSES = REPO / "configs" / "vbot_model_poses.yaml"
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
        raise RuntimeError("PyYAML is required") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_capture(path: Path, pose_name: str) -> dict[str, float]:
    raw = load_yaml(path)
    order = tuple(raw["dds_joint_order"])
    if order != DDS_JOINT_ORDER:
        raise RuntimeError(f"{path}: dds_joint_order does not match expected order")
    pose = raw["poses"][pose_name]
    return dict(zip(DDS_JOINT_ORDER, [float(value) for value in pose["q_motor"]]))


def load_model_pose(path: Path, pose_name: str) -> dict[str, float]:
    poses = load_yaml(path)
    if pose_name not in poses:
        raise RuntimeError(f"pose {pose_name!r} not found in {path}")
    pose = {name: float(value) for name, value in poses[pose_name].items()}
    missing = [name for name in DDS_JOINT_ORDER if name not in pose]
    if missing:
        raise RuntimeError(f"pose {pose_name!r} is missing joints: {', '.join(missing)}")
    return pose


def joint_type(joint_name: str) -> str:
    # FR_hip_joint -> hip
    return joint_name.split("_", 2)[1]


def sign_from_deltas(joint: str, motor_delta: float, model_delta: float, min_delta: float) -> float:
    if abs(motor_delta) < min_delta:
        raise RuntimeError(f"{joint}: motor delta too small ({motor_delta:+.6f})")
    if abs(model_delta) < min_delta:
        raise RuntimeError(f"{joint}: model delta too small ({model_delta:+.6f})")
    return 1.0 if motor_delta * model_delta > 0.0 else -1.0


def bias_samples_for_mode(
    bias_source: str,
    sign: float,
    real_stand: float,
    real_down: float,
    model_stand: float,
    model_down: float,
) -> list[float]:
    samples = []
    if bias_source in ("stand", "both"):
        samples.append(model_stand - sign * real_stand)
    if bias_source in ("down", "both"):
        samples.append(model_down - sign * real_down)
    return samples


def generate_mapping(args):
    real_stand = load_capture(Path(args.real_stand), "stand")
    real_down = load_capture(Path(args.real_down), "down")
    model_stand = load_model_pose(Path(args.model_poses), "stand")
    model_down = load_model_pose(Path(args.model_poses), "down")

    signs = {}
    per_joint_bias_samples = {}
    type_bias_samples = {"hip": [], "thigh": [], "calf": []}

    for joint in DDS_JOINT_ORDER:
        motor_delta = real_down[joint] - real_stand[joint]
        model_delta = model_down[joint] - model_stand[joint]
        sign = sign_from_deltas(joint, motor_delta, model_delta, args.min_delta)
        signs[joint] = sign

        samples = bias_samples_for_mode(
            args.bias_source,
            sign,
            real_stand[joint],
            real_down[joint],
            model_stand[joint],
            model_down[joint],
        )
        per_joint_bias_samples[joint] = samples
        type_bias_samples[joint_type(joint)].extend(samples)

    type_bias = {
        name: statistics.fmean(samples)
        for name, samples in type_bias_samples.items()
    }

    joints = {}
    print(
        "joint                 sign  type_bias  stand_err   down_err"
        "   raw_bias_samples"
    )
    print("-" * 92)
    for joint in DDS_JOINT_ORDER:
        sign = signs[joint]
        bias = type_bias[joint_type(joint)]
        mapped_stand = sign * real_stand[joint] + bias
        mapped_down = sign * real_down[joint] + bias
        stand_err = mapped_stand - model_stand[joint]
        down_err = mapped_down - model_down[joint]
        raw_samples = ", ".join(f"{value:+.6f}" for value in per_joint_bias_samples[joint])
        print(
            f"{joint:20s} {sign:+4.0f} {bias:+10.6f}"
            f" {stand_err:+10.6f} {down_err:+10.6f}   {raw_samples}"
        )
        joints[joint] = {
            "scale": float(sign),
            "bias": float(bias),
            "q_motor_stand": float(real_stand[joint]),
            "q_motor_down": float(real_down[joint]),
            "q_model_stand": float(model_stand[joint]),
            "q_model_down": float(model_down[joint]),
        }

    return {
        "convention": {
            "q_model": "scale * q_motor + bias",
            "dq_model": "scale * dq_motor",
            "q_motor_cmd": "(q_model_cmd - bias) / scale",
            "tau_motor_cmd": "scale * tau_model_cmd",
            "kp_motor_cmd": "kp_model_cmd",
            "kd_motor_cmd": "kd_model_cmd",
            "scale_mode": "sign_only",
            "bias_mode": "joint_type_average",
            "bias_source": args.bias_source,
            "strict_anchor_check": False,
            "note": (
                "scale is sign-only (+1/-1); bias is averaged by joint type."
            ),
        },
        "dds_joint_order": list(DDS_JOINT_ORDER),
        "real_down": str(Path(args.real_down)),
        "real_stand": str(Path(args.real_stand)),
        "model_poses": str(Path(args.model_poses)),
        "bias_by_joint_type": {name: float(value) for name, value in type_bias.items()},
        "joints": joints,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate sign-only VBot real joint mapping")
    parser.add_argument("--real-down", default=str(DEFAULT_REAL_DOWN))
    parser.add_argument("--real-stand", default=str(DEFAULT_REAL_STAND))
    parser.add_argument("--model-poses", default=str(DEFAULT_MODEL_POSES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--bias-source",
        choices=("stand", "down", "both"),
        default="stand",
        help="which calibration pose(s) to average for same-type bias",
    )
    parser.add_argument("--min-delta", type=float, default=1.0e-6)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    mapping = generate_mapping(args)
    output = Path(args.output)
    if args.write:
        save_yaml(output, mapping)
        print(f"\nWrote sign-only mapping to {output}")
    else:
        print(f"\nDry run only. Add --write to update {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
