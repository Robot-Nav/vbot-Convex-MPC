"""View a named VBot joint pose in MuJoCo.

This is a pose-inspection helper only. It does not run MPC and does not publish
commands. Use it to tune configs/vbot_model_poses.yaml, especially the down pose.
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
XML_PATH = REPO / "models" / "fatu" / "fatu.xml"
#XML_PATH = REPO / "models" / "MJCF" / "vbot" / "vbot_mpc_scene.xml"
DEFAULT_POSE_FILE = REPO / "configs" / "vbot_model_poses.yaml"
FLOATING_JOINT = "joint_fixed_world"
mj = None
mj_viewer = None

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

EDIT_GROUPS = (
    ("single", ()),
    (
        "hip mirror",
        (
            ("FR_hip_joint", 1.0),
            ("FL_hip_joint", -1.0),
            ("RR_hip_joint", 1.0),
            ("RL_hip_joint", -1.0),
        ),
    ),
    (
        "thigh all",
        (
            ("FR_thigh_joint", 1.0),
            ("FL_thigh_joint", 1.0),
            ("RR_thigh_joint", 1.0),
            ("RL_thigh_joint", 1.0),
        ),
    ),
    (
        "calf all",
        (
            ("FR_calf_joint", 1.0),
            ("FL_calf_joint", 1.0),
            ("RR_calf_joint", 1.0),
            ("RL_calf_joint", 1.0),
        ),
    ),
)


def require_mujoco():
    global mj, mj_viewer
    if mj is not None and mj_viewer is not None:
        return
    try:
        import mujoco as mujoco_module
        import mujoco.viewer as viewer_module
    except Exception as exc:
        raise RuntimeError("MuJoCo is required to open the viewer; --print-only works without it") from exc
    mj = mujoco_module
    mj_viewer = viewer_module


@contextmanager
def raw_terminal_if_available(enabled: bool):
    if not enabled or not sys.stdin.isatty():
        yield False
        return

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        yield True
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def load_yaml(path: Path):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load pose YAML") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_override(text: str) -> tuple[str, float]:
    if "=" not in text:
        raise ValueError(f"override must be JOINT=VALUE, got {text!r}")
    name, value = text.split("=", 1)
    name = name.strip()
    if name not in DDS_JOINT_ORDER:
        raise ValueError(f"unknown joint {name}")
    return name, float(value)


def require_pose(poses, pose_name: str, pose_file: str) -> dict[str, float]:
    if pose_name not in poses:
        raise RuntimeError(f"pose {pose_name!r} not found in {pose_file}")
    return {name: float(value) for name, value in poses[pose_name].items()}


def validate_pose(pose_name: str, pose: dict[str, float]):
    missing = [name for name in DDS_JOINT_ORDER if name not in pose]
    if missing:
        raise RuntimeError(f"pose {pose_name!r} is missing joints: {', '.join(missing)}")


def blend_poses(from_pose: dict[str, float], to_pose: dict[str, float], alpha: float) -> dict[str, float]:
    return {
        name: from_pose[name] + alpha * (to_pose[name] - from_pose[name])
        for name in DDS_JOINT_ORDER
    }


def set_pose(model, data, pose: dict[str, float], base_height: float):
    base_jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, FLOATING_JOINT)
    base_qadr = int(model.jnt_qposadr[base_jid])
    data.qpos[base_qadr : base_qadr + 7] = [0.0, 0.0, base_height, 1.0, 0.0, 0.0, 0.0]

    for joint_name in DDS_JOINT_ORDER:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise RuntimeError(f"joint not found in MuJoCo model: {joint_name}")
        data.qpos[int(model.jnt_qposadr[jid])] = float(pose[joint_name])
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mj.mj_forward(model, data)


def base_qpos_addr(model) -> int:
    base_jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, FLOATING_JOINT)
    if base_jid < 0:
        raise RuntimeError(f"base joint not found in MuJoCo model: {FLOATING_JOINT}")
    return int(model.jnt_qposadr[base_jid])


def get_base_height(model, data) -> float:
    return float(data.qpos[base_qpos_addr(model) + 2])


def set_base_height(model, data, height: float):
    qadr = base_qpos_addr(model)
    data.qpos[qadr + 2] = max(float(height), 0.0)


def print_pose(pose: dict[str, float]):
    for joint_name in DDS_JOINT_ORDER:
        print(f"{joint_name}: {pose[joint_name]:+.6f}")


def read_pose_from_mujoco(model, data) -> dict[str, float]:
    pose = {}
    for joint_name in DDS_JOINT_ORDER:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        pose[joint_name] = float(data.qpos[int(model.jnt_qposadr[jid])])
    return pose


def joint_qpos_addr(model, joint_name: str) -> int:
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"joint not found in MuJoCo model: {joint_name}")
    return int(model.jnt_qposadr[jid])


def joint_limited_range(model, joint_name: str) -> tuple[bool, float, float]:
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"joint not found in MuJoCo model: {joint_name}")
    limited = bool(model.jnt_limited[jid])
    lower, upper = model.jnt_range[jid]
    return limited, float(lower), float(upper)


def clamp_joint_value(model, joint_name: str, value: float) -> float:
    limited, lower, upper = joint_limited_range(model, joint_name)
    if not limited:
        return value
    return min(max(value, lower), upper)


def signed_group_value(model, data, group: tuple[tuple[str, float], ...]) -> float:
    values = []
    for joint_name, sign in group:
        qadr = joint_qpos_addr(model, joint_name)
        values.append(float(data.qpos[qadr]) / sign)
    return sum(values) / len(values)


def apply_signed_group_value(model, data, group: tuple[tuple[str, float], ...], value: float):
    for joint_name, sign in group:
        qadr = joint_qpos_addr(model, joint_name)
        data.qpos[qadr] = clamp_joint_value(model, joint_name, sign * value)


def poll_key(enabled: bool) -> str | None:
    if not enabled:
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return None
    return sys.stdin.read(1)


def print_keyboard_help(selected_joint: str, step: float):
    print("\nKeyboard edit mode:")
    print("  j/k       select next/previous joint")
    print("  g         switch edit group: single, hip mirror, thigh all, calf all")
    print("  +/-       increase/decrease selected joint")
    print("  x/z       raise/lower floating base height")
    print("  ]/[       increase/decrease edit step")
    print("  p         print current pose")
    print("  q         quit viewer and print final pose")
    print(f"  selected  {selected_joint}, group=single, step={step:.4f} rad")


def print_group_values(model, data, group_name: str, group: tuple[tuple[str, float], ...]):
    value = signed_group_value(model, data, group)
    print(f"{group_name}: canonical={value:+.6f}")
    for joint_name, _ in group:
        qadr = joint_qpos_addr(model, joint_name)
        print(f"  {joint_name}: {data.qpos[qadr]:+.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="View a VBot model pose in MuJoCo")
    parser.add_argument("--pose-file", default=str(DEFAULT_POSE_FILE))
    parser.add_argument("--pose", default="down")
    parser.add_argument("--from-pose", default=None, help="blend start pose; --pose is the blend target")
    parser.add_argument("--pose-alpha", type=float, default=1.0, help="blend alpha from --from-pose to --pose")
    parser.add_argument("--base-height", type=float, default=0.28)
    parser.add_argument("--set", action="append", default=[], help="override a joint angle, e.g. FR_thigh_joint=1.3")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="initialize once, then let MuJoCo viewer edits persist; print final joint angles on exit",
    )
    parser.add_argument(
        "--keyboard-edit",
        action="store_true",
        help="edit qpos from the terminal instead of dragging MuJoCo joint sliders",
    )
    parser.add_argument("--step", type=float, default=0.05, help="keyboard edit step in radians")
    args = parser.parse_args()

    poses = load_yaml(Path(args.pose_file))
    target_pose = require_pose(poses, args.pose, args.pose_file)
    validate_pose(args.pose, target_pose)
    if args.from_pose is not None:
        if not 0.0 <= args.pose_alpha <= 1.0:
            raise RuntimeError("--pose-alpha must be in [0, 1]")
        start_pose = require_pose(poses, args.from_pose, args.pose_file)
        validate_pose(args.from_pose, start_pose)
        pose = blend_poses(start_pose, target_pose, args.pose_alpha)
        print(f"blended pose: from={args.from_pose}, to={args.pose}, alpha={args.pose_alpha:.3f}")
    else:
        if abs(args.pose_alpha - 1.0) > 1.0e-12:
            raise RuntimeError("--pose-alpha requires --from-pose")
        pose = target_pose
    for item in args.set:
        name, value = parse_override(item)
        pose[name] = value

    print_pose(pose)
    if args.print_only:
        return 0

    require_mujoco()
    model = mj.MjModel.from_xml_path(str(XML_PATH))
    data = mj.MjData(model)
    set_pose(model, data, pose, args.base_height)

    selected_index = 0
    group_index = 0
    step = abs(args.step)
    if args.keyboard_edit:
        print_keyboard_help(DDS_JOINT_ORDER[selected_index], step)

    effective_interactive = args.interactive or args.keyboard_edit
    with raw_terminal_if_available(args.keyboard_edit) as keyboard_enabled:
        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base")
            viewer.cam.distance = 1.8
            viewer.cam.elevation = -18
            viewer.cam.azimuth = 90
            while viewer.is_running():
                key = poll_key(keyboard_enabled)
                if key is not None:
                    joint_name = DDS_JOINT_ORDER[selected_index]
                    group_name, group = EDIT_GROUPS[group_index]
                    if key == "j":
                        selected_index = (selected_index + 1) % len(DDS_JOINT_ORDER)
                        print(f"selected {DDS_JOINT_ORDER[selected_index]}")
                    elif key == "k":
                        selected_index = (selected_index - 1) % len(DDS_JOINT_ORDER)
                        print(f"selected {DDS_JOINT_ORDER[selected_index]}")
                    elif key == "g":
                        group_index = (group_index + 1) % len(EDIT_GROUPS)
                        group_name, group = EDIT_GROUPS[group_index]
                        print(f"group={group_name}")
                    elif key in ("+", "="):
                        if group:
                            apply_signed_group_value(model, data, group, signed_group_value(model, data, group) + step)
                            print_group_values(model, data, group_name, group)
                        else:
                            qadr = joint_qpos_addr(model, joint_name)
                            data.qpos[qadr] = clamp_joint_value(model, joint_name, float(data.qpos[qadr]) + step)
                            print(f"{joint_name}: {data.qpos[qadr]:+.6f}")
                    elif key == "-":
                        if group:
                            apply_signed_group_value(model, data, group, signed_group_value(model, data, group) - step)
                            print_group_values(model, data, group_name, group)
                        else:
                            qadr = joint_qpos_addr(model, joint_name)
                            data.qpos[qadr] = clamp_joint_value(model, joint_name, float(data.qpos[qadr]) - step)
                            print(f"{joint_name}: {data.qpos[qadr]:+.6f}")
                    elif key == "x":
                        set_base_height(model, data, get_base_height(model, data) + step)
                        print(f"base_height: {get_base_height(model, data):+.6f}")
                    elif key == "z":
                        set_base_height(model, data, get_base_height(model, data) - step)
                        print(f"base_height: {get_base_height(model, data):+.6f}")
                    elif key == "]":
                        step *= 2.0
                        print(f"step={step:.4f} rad")
                    elif key == "[":
                        step = max(step * 0.5, 0.001)
                        print(f"step={step:.4f} rad")
                    elif key == "p":
                        print("\nCurrent MuJoCo joint pose:")
                        print(f"base_height: {get_base_height(model, data):+.6f}")
                        print_pose(read_pose_from_mujoco(model, data))
                    elif key == "q":
                        break

                if not effective_interactive:
                    set_pose(model, data, pose, args.base_height)
                else:
                    # Keep the viewer in sync after direct qpos edits; no control is sent.
                    mj.mj_forward(model, data)
                viewer.sync()
                time.sleep(1.0 / 60.0)

    if effective_interactive:
        print("\nFinal MuJoCo joint pose:")
        print(f"base_height: {get_base_height(model, data):+.6f}")
        print_pose(read_pose_from_mujoco(model, data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
