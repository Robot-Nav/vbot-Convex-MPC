"""VBot real-motor to model-joint affine bridge.

This module keeps the real hardware angle convention separate from the model
and MPC vector conventions.  The affine is

    q_model = scale * q_motor - bias

so velocities use the same scale, and virtual work gives

    tau_motor = scale * tau_model

The bias sign matches ``joint_prone_bias.fatu.txt``:

    bias = scale * q_motor_prone - q_model_prone
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

try:
    import numpy as np
except Exception:  # pragma: no cover - used on small bring-up systems without numpy.
    np = None


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

PIN_LEG_ORDER = ("FR", "FL", "RR", "RL")
MPC_LEG_ORDER = ("FL", "FR", "RL", "RR")

JOINTS = {
    "FR": ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
    "FL": ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
    "RR": ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
    "RL": ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
}

PIN_JOINT_ORDER = tuple(joint for leg in PIN_LEG_ORDER for joint in JOINTS[leg])
MPC_JOINT_ORDER = tuple(joint for leg in MPC_LEG_ORDER for joint in JOINTS[leg])


def _load_yaml(path: Path):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load affine YAML") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_joint_dict(values, order) -> dict[str, float]:
    if isinstance(values, Mapping):
        return {name: float(values[name]) for name in order}
    arr = list(values)
    if len(arr) != len(order):
        raise RuntimeError(f"expected {len(order)} values, got {len(arr)}")
    return {name: float(value) for name, value in zip(order, arr)}


def _vector(values):
    values = [float(value) for value in values]
    if np is None:
        return values
    return np.array(values, dtype=float)


@dataclass(frozen=True)
class JointAffine:
    scale: float
    bias: float

    def __post_init__(self):
        if abs(self.scale) < 1.0e-9:
            raise RuntimeError("joint affine scale is too close to zero")

    def motor_to_model(self, q_motor: float) -> float:
        return self.scale * q_motor - self.bias

    def model_to_motor(self, q_model: float) -> float:
        return (q_model + self.bias) / self.scale

    def motor_velocity_to_model(self, dq_motor: float) -> float:
        return self.scale * dq_motor

    def model_velocity_to_motor(self, dq_model: float) -> float:
        return dq_model / self.scale

    def model_torque_to_motor(self, tau_model: float) -> float:
        return self.scale * tau_model

    def motor_torque_to_model(self, tau_motor: float) -> float:
        return tau_motor / self.scale


class VBotRealJointAffine:
    def __init__(self, order, joints: Mapping[str, JointAffine], path: Path | None = None):
        self.order = tuple(order)
        self.joints = dict(joints)
        self.path = path

        missing = [name for name in self.order if name not in self.joints]
        if missing:
            raise RuntimeError(f"affine is missing joints: {', '.join(missing)}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VBotRealJointAffine":
        path = Path(path)
        raw = _load_yaml(path)
        order = tuple(raw["dds_joint_order"])
        if order != DDS_JOINT_ORDER:
            raise RuntimeError(
                "affine dds_joint_order does not match the expected DDS/CAN order"
            )
        joints = {
            name: JointAffine(
                scale=float(raw["joints"][name]["scale"]),
                bias=float(raw["joints"][name]["bias"]),
            )
            for name in order
        }
        return cls(order, joints, path=path)

    def motor_to_model(self, joint: str, q_motor: float) -> float:
        return self.joints[joint].motor_to_model(q_motor)

    def model_to_motor(self, joint: str, q_model: float) -> float:
        return self.joints[joint].model_to_motor(q_model)

    def motor_velocity_to_model(self, joint: str, dq_motor: float) -> float:
        return self.joints[joint].motor_velocity_to_model(dq_motor)

    def motor_vel_to_model(self, joint: str, dq_motor: float) -> float:
        return self.motor_velocity_to_model(joint, dq_motor)

    def model_velocity_to_motor(self, joint: str, dq_model: float) -> float:
        return self.joints[joint].model_velocity_to_motor(dq_model)

    def model_torque_to_motor(self, joint: str, tau_model: float) -> float:
        return self.joints[joint].model_torque_to_motor(tau_model)

    def motor_torque_to_model(self, joint: str, tau_motor: float) -> float:
        return self.joints[joint].motor_torque_to_model(tau_motor)

    def motor_feedback_to_model_dict(self, q_motor_by_joint) -> dict[str, float]:
        q_motor = _as_joint_dict(q_motor_by_joint, self.order)
        return {
            joint: self.motor_to_model(joint, q_motor[joint])
            for joint in self.order
        }

    def motor_feedback_to_model_velocity_dict(self, dq_motor_by_joint) -> dict[str, float]:
        dq_motor = _as_joint_dict(dq_motor_by_joint, self.order)
        return {
            joint: self.motor_velocity_to_model(joint, dq_motor[joint])
            for joint in self.order
        }

    def pin_joint_positions_from_motor(self, q_motor_by_joint) -> np.ndarray:
        q_model = self.motor_feedback_to_model_dict(q_motor_by_joint)
        return _vector(q_model[joint] for joint in PIN_JOINT_ORDER)

    def pin_joint_velocities_from_motor(self, dq_motor_by_joint) -> np.ndarray:
        dq_model = self.motor_feedback_to_model_velocity_dict(dq_motor_by_joint)
        return _vector(dq_model[joint] for joint in PIN_JOINT_ORDER)

    def mpc_joint_positions_from_motor(self, q_motor_by_joint) -> np.ndarray:
        q_model = self.motor_feedback_to_model_dict(q_motor_by_joint)
        return _vector(q_model[joint] for joint in MPC_JOINT_ORDER)

    def motor_position_commands_from_model_dict(self, q_model_by_joint) -> dict[str, float]:
        q_model = _as_joint_dict(q_model_by_joint, self.order)
        return {
            joint: self.model_to_motor(joint, q_model[joint])
            for joint in self.order
        }

    def motor_torque_commands_from_mpc(self, tau_model_mpc_order) -> np.ndarray:
        tau_model = _as_joint_dict(tau_model_mpc_order, MPC_JOINT_ORDER)
        return _vector(
            self.model_torque_to_motor(joint, tau_model[joint]) for joint in self.order
        )

    def model_torque_feedback_to_mpc(self, tau_motor_by_joint) -> np.ndarray:
        tau_motor = _as_joint_dict(tau_motor_by_joint, self.order)
        tau_model = {
            joint: self.motor_torque_to_model(joint, tau_motor[joint])
            for joint in self.order
        }
        return _vector(tau_model[joint] for joint in MPC_JOINT_ORDER)

    def with_model_anchor(self, q_motor_by_joint, q_model_by_joint) -> "VBotRealJointAffine":
        """Return a copy whose bias maps one measured real pose to one model pose."""
        q_motor = _as_joint_dict(q_motor_by_joint, self.order)
        q_model = _as_joint_dict(q_model_by_joint, self.order)
        joints = {
            joint: JointAffine(
                scale=self.joints[joint].scale,
                bias=self.joints[joint].scale * q_motor[joint] - q_model[joint],
            )
            for joint in self.order
        }
        return VBotRealJointAffine(self.order, joints, path=self.path)


def load_model_pose(path: str | Path, pose_name: str, order=DDS_JOINT_ORDER) -> dict[str, float]:
    poses = _load_yaml(Path(path))
    if pose_name not in poses:
        raise RuntimeError(f"pose {pose_name!r} not found in {path}")
    pose = {joint: float(value) for joint, value in poses[pose_name].items()}
    missing = [joint for joint in order if joint not in pose]
    if missing:
        raise RuntimeError(f"pose {pose_name!r} is missing joints: {', '.join(missing)}")
    return pose


__all__ = [
    "DDS_JOINT_ORDER",
    "PIN_LEG_ORDER",
    "MPC_LEG_ORDER",
    "PIN_JOINT_ORDER",
    "MPC_JOINT_ORDER",
    "JOINTS",
    "JointAffine",
    "VBotRealJointAffine",
    "load_model_pose",
]
