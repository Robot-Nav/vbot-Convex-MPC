"""Solve a symmetric VBot down pose from MuJoCo kinematics.

This tool is for sim2real calibration only. It does not run MPC, does not step
physics, and does not publish commands. It searches a symmetric model pose whose
four feet are on the ground while the base is near a requested down height.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco as mj
import mujoco.viewer


REPO = Path(__file__).resolve().parents[2]
XML_PATH = REPO / "models" / "MJCF" / "vbot" / "vbot_mpc_scene.xml"
DEFAULT_POSE_FILE = REPO / "configs" / "vbot_model_poses.yaml"
FLOATING_JOINT = "joint_fixed_world"

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

FOOT_GEOMS = ("FR", "FL", "RR", "RL")
LOW_BAR_GEOMS = (
    "collision_low_bar_FR_box",
    "collision_low_bar_FL_box",
    "collision_low_bar_RR_box",
    "collision_low_bar_RL_box",
)
KNEE_BODIES = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")


def load_yaml(path: Path):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load/write pose YAML") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load/write pose YAML") from exc
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def joint_qpos_addr(model, joint_name: str) -> int:
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"joint not found in MuJoCo model: {joint_name}")
    return int(model.jnt_qposadr[jid])


def base_qpos_addr(model) -> int:
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, FLOATING_JOINT)
    if jid < 0:
        raise RuntimeError(f"base joint not found in MuJoCo model: {FLOATING_JOINT}")
    return int(model.jnt_qposadr[jid])


def joint_range(model, joint_name: str) -> tuple[float, float]:
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"joint not found in MuJoCo model: {joint_name}")
    lower, upper = model.jnt_range[jid]
    return float(lower), float(upper)


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def make_pose(hip: float, thigh: float, calf: float) -> dict[str, float]:
    return {
        "FR_hip_joint": hip,
        "FR_thigh_joint": thigh,
        "FR_calf_joint": calf,
        "FL_hip_joint": -hip,
        "FL_thigh_joint": thigh,
        "FL_calf_joint": calf,
        "RR_hip_joint": hip,
        "RR_thigh_joint": thigh,
        "RR_calf_joint": calf,
        "RL_hip_joint": -hip,
        "RL_thigh_joint": thigh,
        "RL_calf_joint": calf,
    }


def set_model_state(model, data, pose: dict[str, float], base_height: float):
    qadr = base_qpos_addr(model)
    data.qpos[qadr : qadr + 7] = [0.0, 0.0, base_height, 1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    for joint_name in DDS_JOINT_ORDER:
        data.qpos[joint_qpos_addr(model, joint_name)] = pose[joint_name]
    mj.mj_forward(model, data)


def geom_vertical_extent(model, data, gid: int) -> float:
    geom_type = int(model.geom_type[gid])
    size = model.geom_size[gid]
    if geom_type == mj.mjtGeom.mjGEOM_SPHERE:
        return float(size[0])
    if geom_type == mj.mjtGeom.mjGEOM_BOX:
        # xmat is row-major world orientation. The third row gives each local
        # axis projection onto world z, so this is the box half-height in z.
        xmat = data.geom_xmat[gid]
        return float(abs(xmat[6]) * size[0] + abs(xmat[7]) * size[1] + abs(xmat[8]) * size[2])
    return float(model.geom_rbound[gid])


def geom_ground_clearance(model, data, geom_name: str) -> dict[str, float | tuple[float, float, float]]:
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise RuntimeError(f"geom not found in MuJoCo model: {geom_name}")
    center = tuple(float(v) for v in data.geom_xpos[gid])
    vertical_extent = geom_vertical_extent(model, data, gid)
    return {
        "center": center,
        "vertical_extent": vertical_extent,
        "ground_clearance": center[2] - vertical_extent,
    }


def body_ground_clearance(model, data, body_name: str, radius: float) -> dict[str, float | tuple[float, float, float]]:
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"body not found in MuJoCo model: {body_name}")
    center = tuple(float(v) for v in data.xpos[bid])
    return {
        "center": center,
        "vertical_extent": radius,
        "ground_clearance": center[2] - radius,
    }


def contact_clearances(
    model,
    data,
    include_low_bars: bool,
    include_knees: bool,
    knee_radius: float,
) -> dict[str, dict[str, float | tuple[float, float, float]]]:
    contacts = {}
    for name in FOOT_GEOMS:
        contacts[f"foot_{name}"] = geom_ground_clearance(model, data, name)
    if include_low_bars:
        for name in LOW_BAR_GEOMS:
            leg = name.split("_")[-2]
            contacts[f"low_bar_{leg}"] = geom_ground_clearance(model, data, name)
    if include_knees:
        for name in KNEE_BODIES:
            contacts[f"knee_{name[:2]}"] = body_ground_clearance(model, data, name, knee_radius)
    return contacts


def bounded_variables(model, x: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    hip, thigh, calf, base_height = x
    hip_lower, hip_upper = joint_range(model, "FR_hip_joint")
    front_thigh_lower, front_thigh_upper = joint_range(model, "FR_thigh_joint")
    rear_thigh_lower, rear_thigh_upper = joint_range(model, "RR_thigh_joint")
    calf_lower, calf_upper = joint_range(model, "FR_calf_joint")
    thigh_lower = max(front_thigh_lower, rear_thigh_lower)
    thigh_upper = min(front_thigh_upper, rear_thigh_upper)
    return (
        clamp(hip, hip_lower, hip_upper),
        clamp(thigh, thigh_lower, thigh_upper),
        clamp(calf, calf_lower, calf_upper),
        clamp(base_height, 0.02, 0.50),
    )


def evaluate(
    model,
    data,
    x: tuple[float, float, float, float],
    target_height: float,
    include_low_bars: bool,
    include_knees: bool,
    knee_radius: float,
) -> tuple[float, dict]:
    hip, thigh, calf, base_height = bounded_variables(model, x)
    pose = make_pose(hip, thigh, calf)
    set_model_state(model, data, pose, base_height)
    contacts = contact_clearances(model, data, include_low_bars, include_knees, knee_radius)
    clearances = [float(contact["ground_clearance"]) for contact in contacts.values()]

    contact_cost = sum(clearance * clearance for clearance in clearances) / len(clearances)
    height_cost = (base_height - target_height) ** 2
    hip_cost = hip * hip
    # Keep the solver near a compact down posture instead of using extreme
    # joint limits just to satisfy foot height.
    regularization = 0.01 * hip_cost + 0.002 * (thigh - 1.4) ** 2 + 0.002 * (calf + 2.5) ** 2
    cost = 1000.0 * contact_cost + 8.0 * height_cost + regularization
    info = {
        "x": (hip, thigh, calf, base_height),
        "pose": pose,
        "contacts": contacts,
        "contact_clearances": clearances,
        "cost": cost,
    }
    return cost, info


def coordinate_search(
    model,
    data,
    initial_x: tuple[float, float, float, float],
    target_height: float,
    include_low_bars: bool,
    include_knees: bool,
    knee_radius: float,
):
    x = bounded_variables(model, initial_x)
    best_cost, best_info = evaluate(model, data, x, target_height, include_low_bars, include_knees, knee_radius)
    steps = [0.05, 0.10, 0.10, 0.01]

    while max(steps) > 0.0005:
        improved = False
        for i, step in enumerate(steps):
            for sign in (1.0, -1.0):
                candidate = list(x)
                candidate[i] += sign * step
                candidate = bounded_variables(model, tuple(candidate))
                cost, info = evaluate(model, data, candidate, target_height, include_low_bars, include_knees, knee_radius)
                if cost < best_cost:
                    x = candidate
                    best_cost = cost
                    best_info = info
                    improved = True
        if not improved:
            steps = [step * 0.5 for step in steps]
    return best_info


def scipy_search(
    model,
    data,
    initial_x: tuple[float, float, float, float],
    target_height: float,
    include_low_bars: bool,
    include_knees: bool,
    knee_radius: float,
):
    try:
        from scipy.optimize import minimize
    except Exception:
        return None

    def objective(values):
        cost, _ = evaluate(
            model,
            data,
            tuple(float(v) for v in values),
            target_height,
            include_low_bars,
            include_knees,
            knee_radius,
        )
        return cost

    result = minimize(objective, bounded_variables(model, initial_x), method="Nelder-Mead", options={"maxiter": 800})
    clipped = bounded_variables(model, tuple(float(v) for v in result.x))
    # Refine once with bound-aware coordinate search; Nelder-Mead itself is not
    # bound-constrained and may finish slightly outside joint limits.
    return coordinate_search(model, data, clipped, target_height, include_low_bars, include_knees, knee_radius)


def print_solution(info: dict):
    hip, thigh, calf, base_height = info["x"]
    print("\nSolved symmetric down pose")
    print(f"  canonical_hip:   {hip:+.6f}")
    print(f"  canonical_thigh: {thigh:+.6f}")
    print(f"  canonical_calf:  {calf:+.6f}")
    print(f"  base_height:     {base_height:+.6f}")
    print(f"  cost:            {info['cost']:.8f}")
    print("\nContact clearances:")
    for name, contact in info["contacts"].items():
        center = contact["center"]
        print(
            f"  {name}: clearance={contact['ground_clearance']:+.6f}, "
            f"vertical_extent={contact['vertical_extent']:.6f}, "
            f"center=({center[0]:+.6f}, {center[1]:+.6f}, {center[2]:+.6f})"
        )
    print("\nModel joint pose:")
    for joint_name in DDS_JOINT_ORDER:
        print(f"{joint_name}: {info['pose'][joint_name]:+.6f}")


def view_solution(model, data, info: dict):
    set_model_state(model, data, info["pose"], info["x"][3])
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base")
        viewer.cam.distance = 1.8
        viewer.cam.elevation = -18
        viewer.cam.azimuth = 90
        while viewer.is_running():
            # Hold the solved kinematic pose fixed for visual inspection.
            set_model_state(model, data, info["pose"], info["x"][3])
            viewer.sync()
            time.sleep(1.0 / 60.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve VBot down pose from MuJoCo IK")
    parser.add_argument("--pose-file", default=str(DEFAULT_POSE_FILE))
    parser.add_argument("--pose-name", default="down")
    parser.add_argument("--target-height", type=float, default=0.10)
    parser.add_argument("--initial-hip", type=float, default=0.05)
    parser.add_argument("--initial-thigh", type=float, default=1.4)
    parser.add_argument("--initial-calf", type=float, default=-2.5)
    parser.add_argument("--initial-height", type=float, default=0.12)
    parser.add_argument(
        "--contact-profile",
        choices=("feet", "down"),
        default="down",
        help="feet: only foot contacts; down: feet + base low bars + knee points",
    )
    parser.add_argument(
        "--knee-radius",
        type=float,
        default=0.02,
        help="radius used when targeting calf-joint body origins near the ground",
    )
    parser.add_argument("--view", action="store_true", help="show the solved pose in MuJoCo viewer")
    parser.add_argument("--write", action="store_true", help="write solved pose into --pose-file")
    args = parser.parse_args()

    model = mj.MjModel.from_xml_path(str(XML_PATH))
    data = mj.MjData(model)
    initial_x = (args.initial_hip, args.initial_thigh, args.initial_calf, args.initial_height)
    include_low_bars = args.contact_profile == "down"
    include_knees = args.contact_profile == "down"

    info = scipy_search(
        model,
        data,
        initial_x,
        args.target_height,
        include_low_bars,
        include_knees,
        args.knee_radius,
    )
    if info is None:
        info = coordinate_search(
            model,
            data,
            initial_x,
            args.target_height,
            include_low_bars,
            include_knees,
            args.knee_radius,
        )
    print_solution(info)

    if args.write:
        pose_file = Path(args.pose_file)
        poses = load_yaml(pose_file)
        poses[args.pose_name] = {joint: round(value, 2) for joint, value in info["pose"].items()}
        save_yaml(pose_file, poses)
        print(f"\nWrote pose {args.pose_name!r} to {pose_file}")
    else:
        print("\nDry run only. Add --write to update the model pose YAML.")

    if args.view:
        view_solution(model, data, info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
