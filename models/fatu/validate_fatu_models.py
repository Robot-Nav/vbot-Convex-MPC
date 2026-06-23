#!/usr/bin/env python3
"""Validate kinematic alignment between fatu.urdf and fatu.xml."""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

TOL = 1e-3
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_URDF = SCRIPT_DIR / "fatu.urdf"
DEFAULT_XML = SCRIPT_DIR / "fatu.xml"

JOINT_ORDER = [
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
]

HOME_JOINT_POS = {
    "F.*_hip_joint": 0.0,
    "R.*_hip_joint": 0.0,
    "F.*_thigh_joint": 1.1,
    "R.*_thigh_joint": 1.3,
    ".*_calf_joint": -1.8,
}

EXPECTED = {
    "spawn_z": 0.29,
    "base_mass": 9.016326,
    "hip_effort": 12.0,
    "thigh_effort": 12.0,
    "calf_effort": 24.0,
    "foot_mass": 0.01,
    "foot_radius": 0.021,
    "hip_limits": (-0.733, 0.733),
    "front_thigh_limits": (-1.559, 3.129),
    "back_thigh_limits": (-0.512, 4.177),
    "calf_limits": (-2.638, -0.785),
    "hip_attachments": {
        "FR_hip": (0.18453, -0.051, 0.0),
        "FL_hip": (0.18453, 0.051, 0.0),
        "RR_hip": (-0.18453, -0.051, 0.0),
        "RL_hip": (-0.18453, 0.051, 0.0),
    },
    "thigh_offset_y": {
        "FR": -0.0975,
        "FL": 0.0975,
        "RR": -0.0975,
        "RL": 0.0975,
    },
    "calf_offset_z": -0.1985,
    "foot_offset_z": -0.214,
}


@dataclass
class Issue:
    severity: str
    message: str


def parse_xyz(text: str) -> tuple[float, float, float]:
    parts = [float(v) for v in text.split()]
    if len(parts) == 1:
        return parts[0], parts[0], parts[0]
    if len(parts) == 2:
        return parts[0], parts[1], 0.0
    return parts[0], parts[1], parts[2]


def parse_range(text: str) -> tuple[float, float]:
    lo, hi = [float(v) for v in text.split()]
    return lo, hi


def nearly_equal(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def vec_equal(a: tuple[float, ...], b: tuple[float, ...], tol: float = TOL) -> bool:
    return all(nearly_equal(x, y, tol) for x, y in zip(a, b))


def load_urdf_joints(urdf_path: Path) -> dict[str, dict]:
    root = ET.parse(urdf_path).getroot()
    joints: dict[str, dict] = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        name = joint.get("name")
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        dynamics = joint.find("dynamics")
        joints[name] = {
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "origin": parse_xyz(origin.get("xyz", "0 0 0")) if origin is not None else (0.0, 0.0, 0.0),
            "axis": parse_xyz(axis.get("xyz", "0 1 0")) if axis is not None else (0.0, 1.0, 0.0),
            "lower": float(limit.get("lower")),
            "upper": float(limit.get("upper")),
            "effort": float(limit.get("effort")),
            "velocity": float(limit.get("velocity", "0")),
            "damping": float(dynamics.get("damping")) if dynamics is not None else None,
            "friction": float(dynamics.get("friction")) if dynamics is not None else None,
        }
    return joints


def load_urdf_link_masses(urdf_path: Path) -> dict[str, float]:
    root = ET.parse(urdf_path).getroot()
    masses: dict[str, float] = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = inertial.find("mass")
        if mass is None:
            continue
        masses[link.get("name")] = float(mass.get("value"))
    return masses


def load_xml_model(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    return root


def find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None


def build_default_joint_attrs(root: ET.Element) -> dict[str, dict]:
    attrs: dict[str, dict] = {}

    def visit(default_elem: ET.Element, inherited: dict) -> None:
        cls = default_elem.get("class")
        local = dict(inherited)
        joint = default_elem.find("joint")
        if joint is not None:
            if joint.get("range"):
                local["range"] = parse_range(joint.get("range"))
            if joint.get("axis"):
                local["axis"] = parse_xyz(joint.get("axis"))
        if cls:
            attrs[cls] = dict(local)
        for child in default_elem.findall("default"):
            visit(child, local)

    for default in root.findall("./default/default"):
        visit(default, {"axis": parse_xyz("0 1 0")})
    return attrs


def get_joint_info(root: ET.Element, joint_name: str) -> dict:
    default_attrs = build_default_joint_attrs(root)
    fatu_defaults = default_attrs.get("fatu", {"axis": parse_xyz("0 1 0")})
    for joint in root.iter("joint"):
        if joint.get("name") != joint_name:
            continue
        cls = joint.get("class", "fatu")
        inherited = dict(fatu_defaults)
        inherited.update(default_attrs.get(cls, {}))
        axis = parse_xyz(joint.get("axis")) if joint.get("axis") else inherited.get("axis", (0.0, 1.0, 0.0))
        info: dict = {"axis": axis}
        if joint.get("range"):
            info["lower"], info["upper"] = parse_range(joint.get("range"))
        elif "range" in inherited:
            info["lower"], info["upper"] = inherited["range"]
        return info
    raise KeyError(joint_name)


def get_body_pos(body: ET.Element) -> tuple[float, float, float]:
    return parse_xyz(body.get("pos", "0 0 0"))


def get_spawn_z(root: ET.Element) -> float:
    base = find_body(root, "base_link")
    if base is None:
        raise RuntimeError("base_link body not found in XML")
    return get_body_pos(base)[2]


def get_motor_range(root: ET.Element, class_name: str) -> tuple[float, float] | None:
    for default in root.iter("default"):
        if default.get("class") != class_name:
            continue
        motor = default.find("motor")
        if motor is not None and motor.get("ctrlrange"):
            return parse_range(motor.get("ctrlrange"))
    return None


def get_foot_defaults(root: ET.Element) -> dict:
    for default in root.iter("default"):
        if default.get("class") != "foot":
            continue
        geom = default.find("geom")
        if geom is None:
            continue
        return {
            "type": geom.get("type"),
            "size": float(geom.get("size")),
            "condim": int(geom.get("condim", "3")),
        }
    return {}


def parse_keyframe_home(root: ET.Element) -> tuple[list[float], list[float]] | None:
    keyframe = root.find("keyframe")
    if keyframe is None:
        return None
    for key in keyframe.findall("key"):
        if key.get("name") == "home":
            qpos = [float(v) for v in key.get("qpos").split()]
            ctrl = [float(v) for v in key.get("ctrl").split()]
            return qpos, ctrl
    return None


def validate(urdf_path: Path, xml_path: Path) -> list[Issue]:
    issues: list[Issue] = []
    urdf_joints = load_urdf_joints(urdf_path)
    urdf_masses = load_urdf_link_masses(urdf_path)
    xml_root = load_xml_model(xml_path)

    if find_body(xml_root, "base_link") is None:
        issues.append(Issue("error", "XML missing body name base_link"))

    spawn_z = get_spawn_z(xml_root)
    if not nearly_equal(spawn_z, EXPECTED["spawn_z"]):
        issues.append(
            Issue(
                "error",
                f"XML spawn height z={spawn_z:.4f}, expected {EXPECTED['spawn_z']:.4f}",
            )
        )

    if not nearly_equal(urdf_masses.get("base_link", 0.0), EXPECTED["base_mass"]):
        issues.append(Issue("warn", "URDF base_link mass differs from expected canonical value"))

    for joint_name in JOINT_ORDER:
        if joint_name not in urdf_joints:
            issues.append(Issue("error", f"URDF missing joint {joint_name}"))
            continue
        try:
            xml_joint = get_joint_info(xml_root, joint_name)
        except KeyError:
            issues.append(Issue("error", f"XML missing joint {joint_name}"))
            continue

        urdf = urdf_joints[joint_name]
        if not vec_equal((urdf["lower"], urdf["upper"]), (xml_joint["lower"], xml_joint["upper"]), tol=5e-3):
            issues.append(
                Issue(
                    "error",
                    f"{joint_name} limits URDF [{urdf['lower']}, {urdf['upper']}] "
                    f"!= XML [{xml_joint['lower']}, {xml_joint['upper']}]",
                )
            )

        if not vec_equal(urdf["axis"], xml_joint["axis"]):
            issues.append(Issue("error", f"{joint_name} axis mismatch"))

        leg_prefix = joint_name.split("_")[0]
        if joint_name.endswith("_hip_joint"):
            body = find_body(xml_root, f"{leg_prefix}_hip")
            expected = EXPECTED["hip_attachments"][f"{leg_prefix}_hip"]
            actual = get_body_pos(body) if body is not None else None
            if actual is None or not vec_equal(actual, expected, tol=1e-4):
                issues.append(
                    Issue(
                        "error",
                        f"{leg_prefix}_hip attachment XML {actual} != URDF {expected}",
                    )
                )
            if not nearly_equal(urdf["effort"], EXPECTED["hip_effort"]):
                issues.append(Issue("warn", f"{joint_name} URDF effort={urdf['effort']}, expected 12"))
        elif joint_name.endswith("_thigh_joint"):
            body = find_body(xml_root, f"{leg_prefix}_thigh")
            expected_y = EXPECTED["thigh_offset_y"][leg_prefix]
            actual = get_body_pos(body) if body is not None else None
            if actual is None or not nearly_equal(actual[1], expected_y, 1e-4):
                issues.append(
                    Issue(
                        "error",
                        f"{leg_prefix}_thigh y-offset XML {actual} expected y={expected_y}",
                    )
                )
            expected_limits = (
                EXPECTED["front_thigh_limits"]
                if leg_prefix in ("FR", "FL")
                else EXPECTED["back_thigh_limits"]
            )
            if not vec_equal((urdf["lower"], urdf["upper"]), expected_limits, tol=5e-3):
                issues.append(
                    Issue(
                        "error",
                        f"{joint_name} URDF limits {urdf['lower'], urdf['upper']} "
                        f"!= expected {expected_limits}",
                    )
                )
        elif joint_name.endswith("_calf_joint"):
            body = find_body(xml_root, f"{leg_prefix}_calf")
            actual = get_body_pos(body) if body is not None else None
            if actual is None or not nearly_equal(actual[2], EXPECTED["calf_offset_z"], 1e-4):
                issues.append(
                    Issue(
                        "error",
                        f"{leg_prefix}_calf z-offset XML {actual} expected z={EXPECTED['calf_offset_z']}",
                    )
                )
            if not nearly_equal(urdf["effort"], EXPECTED["calf_effort"]):
                issues.append(Issue("error", f"{joint_name} URDF effort={urdf['effort']}, expected 24"))

    knee_motor = get_motor_range(xml_root, "knee")
    if knee_motor is None or not vec_equal(knee_motor, (-24.0, 24.0)):
        issues.append(Issue("error", f"XML knee motor ctrlrange={knee_motor}, expected (-24, 24)"))

    foot_defaults = get_foot_defaults(xml_root)
    if foot_defaults.get("type") != "sphere":
        issues.append(Issue("error", "XML foot collision should be sphere"))
    if foot_defaults.get("size") != EXPECTED["foot_radius"]:
        issues.append(Issue("error", f"XML foot radius={foot_defaults.get('size')}, expected 0.021"))
    if foot_defaults.get("condim") != 6:
        issues.append(Issue("warn", f"XML foot condim={foot_defaults.get('condim')}, recommended 6"))

    for leg in ("FR", "FL", "RR", "RL"):
        foot_body = find_body(xml_root, f"{leg}_foot")
        if foot_body is None:
            issues.append(Issue("error", f"XML missing {leg}_foot body"))
            continue
        if not vec_equal(get_body_pos(foot_body), (0.0, 0.0, EXPECTED["foot_offset_z"]), tol=1e-4):
            issues.append(Issue("error", f"{leg}_foot offset mismatch"))
        inertial = foot_body.find("inertial")
        if inertial is None or not nearly_equal(float(inertial.get("mass")), EXPECTED["foot_mass"]):
            issues.append(Issue("error", f"{leg}_foot missing 0.01 kg inertial in XML"))

        urdf_foot_mass = urdf_masses.get(f"{leg}_foot")
        if urdf_foot_mass is None or not nearly_equal(urdf_foot_mass, EXPECTED["foot_mass"]):
            issues.append(Issue("error", f"URDF {leg}_foot mass != 0.01"))

    home = parse_keyframe_home(xml_root)
    if home is None:
        issues.append(Issue("error", "XML missing keyframe home"))
    else:
        qpos, ctrl = home
        if len(qpos) != 19 or len(ctrl) != 12:
            issues.append(Issue("error", f"home keyframe size qpos={len(qpos)} ctrl={len(ctrl)}"))
        else:
            if not nearly_equal(qpos[2], EXPECTED["spawn_z"]):
                issues.append(Issue("error", "home keyframe qpos z != spawn height"))
            expected_ctrl = [0, 1.1, -1.8, 0, 1.1, -1.8, 0, 1.3, -1.8, 0, 1.3, -1.8]
            if not vec_equal(tuple(ctrl), tuple(expected_ctrl), tol=1e-6):
                issues.append(Issue("error", f"home ctrl {ctrl} != expected {expected_ctrl}"))

    imu_site = None
    for site in xml_root.iter("site"):
        if site.get("name") == "trunk_imu":
            imu_site = parse_xyz(site.get("pos", "0 0 0"))
            break
    if imu_site != (0.0, 0.0, 0.0):
        issues.append(Issue("error", f"trunk_imu site at {imu_site}, expected origin"))

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    args = parser.parse_args()

    if not args.urdf.is_file():
        print(f"URDF not found: {args.urdf}", file=sys.stderr)
        return 2
    if not args.xml.is_file():
        print(f"XML not found: {args.xml}", file=sys.stderr)
        return 2

    issues = validate(args.urdf, args.xml)
    errors = [i for i in issues if i.severity == "error"]
    warns = [i for i in issues if i.severity == "warn"]

    for issue in issues:
        print(f"[{issue.severity.upper()}] {issue.message}")

    print()
    if errors:
        print(f"FAILED: {len(errors)} error(s), {len(warns)} warning(s)")
        return 1
    if warns:
        print(f"PASSED with {len(warns)} warning(s)")
        return 0
    print("PASSED: URDF and XML are aligned for policy transfer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
