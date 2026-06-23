"""Print a circular-angle VBot real-joint sign+bias mapping.

This helper treats joint angles as periodic values.  It keeps scale as a pure
direction (+1/-1), infers that direction from the circular stand->down motion,
and computes each joint bias from the model ``down`` pose:

    q_model ~= wrap(scale * q_motor + bias)

The script is read-only.  It prints a table and a YAML candidate to stdout.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
DEFAULT_REAL_STAND = WORKSPACE / "stand.yaml"
DEFAULT_REAL_DOWN = WORKSPACE / "down.yaml"
DEFAULT_MODEL_POSES = REPO / "configs" / "vbot_model_poses.yaml"

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

TAU = 2.0 * math.pi


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def circular_delta(to_angle: float, from_angle: float) -> float:
    return wrap_to_pi(to_angle - from_angle)


def equivalent_near(angle: float, reference: float) -> float:
    return angle + TAU * round((reference - angle) / TAU)


def load_yaml(path: Path):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data) -> str:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required") from exc
    return yaml.safe_dump(data, sort_keys=False)


def load_capture(path: Path, pose_name: str) -> dict[str, float]:
    raw = load_yaml(path)
    order = tuple(raw["dds_joint_order"])
    if order != DDS_JOINT_ORDER:
        raise RuntimeError(f"{path}: dds_joint_order does not match expected order")
    pose = raw["poses"][pose_name]
    if any(int(value) != 0 for value in pose.get("lost", [])):
        raise RuntimeError(f"{path}: pose {pose_name!r} has lost != 0")
    return dict(zip(DDS_JOINT_ORDER, [float(value) for value in pose["q_motor"]]))


def load_model_pose(path: Path, pose_name: str) -> dict[str, float]:
    poses = load_yaml(path)
    if pose_name not in poses:
        raise RuntimeError(f"pose {pose_name!r} not found in {path}")
    pose = {joint: float(value) for joint, value in poses[pose_name].items()}
    missing = [joint for joint in DDS_JOINT_ORDER if joint not in pose]
    if missing:
        raise RuntimeError(f"pose {pose_name!r} is missing joints: {', '.join(missing)}")
    return pose


def infer_sign(joint: str, motor_delta: float, model_delta: float, min_delta: float) -> float:
    if abs(motor_delta) < min_delta:
        raise RuntimeError(f"{joint}: circular motor delta too small ({motor_delta:+.6f})")
    if abs(model_delta) < min_delta:
        raise RuntimeError(f"{joint}: circular model delta too small ({model_delta:+.6f})")
    return 1.0 if motor_delta * model_delta > 0.0 else -1.0


def generate_mapping(args):
    real_stand = load_capture(Path(args.real_stand), "stand")
    real_down = load_capture(Path(args.real_down), "down")
    model_stand = load_model_pose(Path(args.model_poses), "stand")
    model_down = load_model_pose(Path(args.model_poses), "down")

    joints = {}
    rows = []
    for joint in DDS_JOINT_ORDER:
        motor_delta = circular_delta(real_down[joint], real_stand[joint])
        model_delta = circular_delta(model_down[joint], model_stand[joint])
        sign = infer_sign(joint, motor_delta, model_delta, args.min_delta)

        bias = wrap_to_pi(model_down[joint] - sign * real_down[joint])
        stand_err = wrap_to_pi(sign * real_stand[joint] + bias - model_stand[joint])
        down_err = wrap_to_pi(sign * real_down[joint] + bias - model_down[joint])
        q_motor_stand_cmd = equivalent_near(
            (model_stand[joint] - bias) / sign,
            real_stand[joint],
        )
        q_motor_down_cmd = equivalent_near(
            (model_down[joint] - bias) / sign,
            real_down[joint],
        )

        rows.append(
            {
                "joint": joint,
                "sign": sign,
                "bias": bias,
                "motor_delta": motor_delta,
                "model_delta": model_delta,
                "stand_err": stand_err,
                "down_err": down_err,
                "q_motor_stand_cmd": q_motor_stand_cmd,
                "q_motor_down_cmd": q_motor_down_cmd,
            }
        )
        joints[joint] = {
            "scale": float(sign),
            "bias": float(bias),
            "q_motor_stand": float(real_stand[joint]),
            "q_motor_down": float(real_down[joint]),
            "q_model_stand": float(model_stand[joint]),
            "q_model_down": float(model_down[joint]),
            "stand_circular_error": float(stand_err),
            "down_circular_error": float(down_err),
            "q_motor_stand_cmd_near_capture": float(q_motor_stand_cmd),
            "q_motor_down_cmd_near_capture": float(q_motor_down_cmd),
        }

    mapping = {
        "convention": {
            "q_model": "wrap_to_pi(scale * q_motor + bias)",
            "dq_model": "scale * dq_motor",
            "q_motor_cmd": "nearest_equivalent((q_model_cmd - bias) / scale)",
            "tau_motor_cmd": "scale * tau_model_cmd",
            "scale_mode": "sign_only",
            "bias_mode": "per_joint_down_circular",
            "sign_source": "circular stand_to_down delta",
            "bias_source": "down",
            "note": (
                "Angles are treated on a circle; bias aligns the real down "
                "capture to the model down pose modulo 2*pi."
            ),
        },
        "dds_joint_order": list(DDS_JOINT_ORDER),
        "real_down": str(Path(args.real_down)),
        "real_stand": str(Path(args.real_stand)),
        "model_poses": str(Path(args.model_poses)),
        "joints": joints,
    }
    return rows, mapping


def print_table(rows):
    print(
        "joint                 sign       bias  motor_dlt  model_dlt  "
        "stand_err   down_err  q_down_cmd"
    )
    print("-" * 100)
    for row in rows:
        print(
            f"{row['joint']:20s} {row['sign']:+4.0f}"
            f" {row['bias']:+10.6f}"
            f" {row['motor_delta']:+10.6f}"
            f" {row['model_delta']:+10.6f}"
            f" {row['stand_err']:+10.6f}"
            f" {row['down_err']:+10.6f}"
            f" {row['q_motor_down_cmd']:+11.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print circular down-bias VBot real joint mapping"
    )
    parser.add_argument("--real-down", default=str(DEFAULT_REAL_DOWN))
    parser.add_argument("--real-stand", default=str(DEFAULT_REAL_STAND))
    parser.add_argument("--model-poses", default=str(DEFAULT_MODEL_POSES))
    parser.add_argument("--min-delta", type=float, default=1.0e-6)
    parser.add_argument(
        "--no-yaml",
        action="store_true",
        help="only print the summary table",
    )
    args = parser.parse_args()

    rows, mapping = generate_mapping(args)
    print_table(rows)
    if not args.no_yaml:
        print("\nYAML candidate:")
        print(dump_yaml(mapping), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
