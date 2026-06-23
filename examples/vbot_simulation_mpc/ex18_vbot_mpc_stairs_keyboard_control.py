"""
VBot validation 18: keyboard-controlled MPC on a known stair terrain.

This example keeps the ex17 keyboard workflow, but loads a separate MuJoCo
scene with low stairs and gives the controller a simple known stair height
profile. The terrain profile is used only for touchdown z planning and body
height scheduling; the MPC force constraints are still the flat horizontal
contact model.

Controls:
- I/K: increase/decrease forward velocity
- J/L: increase/decrease lateral velocity
- U/O: increase/decrease yaw rate
- X: stop
- T: reset
- Esc: exit viewer
"""
import argparse
import os
import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# viewer scripts generally do not need matplotlib, but this avoids cache
# warnings if plotting helpers are imported indirectly.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import mujoco as mj
import mujoco.viewer
import numpy as np

from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.gait import FOOT_TOUCHDOWN_CLEARANCE, Gait
from convex_mpc.leg_controller import LegController
from convex_mpc.mujoco_vbot_model import MuJoCo_VBot_Model
from convex_mpc.vbot_robot_data import (
    DEFAULT_JOINT_ANGLES,
    JOINTS,
    MPC_LEG_ORDER,
    PinVBotModel,
)


REPO = Path(__file__).resolve().parents[2]
STAIRS_XML_PATH = REPO / "models" / "MJCF" / "vbot" / "vbot_mpc_scene_stairs.xml"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_PREFIX = "ex18_stairs"

# -----------------------------
# Simulation settings
# -----------------------------

SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ

CTRL_HZ = 200
CTRL_DT = 1.0 / CTRL_HZ
CTRL_DECIM = SIM_HZ // CTRL_HZ

VIEWER_HZ = 60.0
VIEWER_DT = 1.0 / VIEWER_HZ

# -----------------------------
# Conservative stair gait
# -----------------------------

GAIT_HZ = 0.60
GAIT_DUTY = 0.84
GAIT_T = 1.0 / GAIT_HZ
SWING_HEIGHT = 0.06
CRAWL_PHASE_OFFSET = np.array([0.00, 0.50, 0.75, 0.25])

MPC_DT = GAIT_T / 16
MPC_HZ = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))

INITIAL_X_POS = 0.0
INITIAL_Y_POS = 0.0

# Nominal body height above the local support surface. For the default VBot
# stand pose, the foot sphere bottoms touch the ground at about 0.277 m.
# On stairs the controller tracks only a fraction of the local terrain height;
# a 1:1 body-height target over-stretches this VBot model and saturates torque.
NOMINAL_BODY_HEIGHT = 0.2774
STAND_VEL_EPS = 1e-3

# Look ahead on the known stair profile so the body starts lifting before the
# front feet reach the first riser.
STAIR_HEIGHT_PREVIEW_X = 0.25
STAIR_BODY_HEIGHT_GAIN = 0.55
STAIR_BODY_SUPPORT_FRONT_X = 0.20
STAIR_BODY_FOOT_HEIGHT_GAIN = 1.0
STAIR_BODY_USE_FOOT_SUPPORT = True
Z_POS_RAMP = 0.04
USE_STAIR_AWARE_FOOTHOLD = True
STAIR_FOOTHOLD_MARGIN_X = 0.025
STAIR_SWING_TERRAIN_CLEARANCE = 0.055
USE_TERRAIN_AWARE_SWING = False
USE_ACTIVE_STAIR_SWING = False
STAIR_ACTIVE_SWING_CLEARANCE = 0.050
STAIR_ACTIVE_SWING_RISER_CLEARANCE = 0.055
STAIR_ACTIVE_SWING_RISER_PHASE_LEAD = 0.10
STAIR_ACTIVE_SWING_MIN_PEAK_PHASE = 0.24
STAIR_ACTIVE_SWING_MAX_PEAK_PHASE = 0.55
STAIR_ACTIVE_SWING_BETA_SHARPNESS = 8.0
STAIR_ACTIVE_SWING_SAMPLE_COUNT = 9
STAIR_ACTIVE_SWING_MAX_TERRAIN_CLEARANCE = 0.14
STAIR_ACTIVE_FRONT_FOOTHOLD_ADVANCE_X = 0.0
STAIR_ACTIVE_FRONT_ADVANCE_WINDOW_X = 0.08
STAIR_ACTIVE_FRONT_TREAD_ENTRY_EXTRA_X = 0.015
STAIR_FOOTHOLD_CENTERING_GAIN = 0.0
STAIR_FOOTHOLD_SOFT_PROJECTION = False
STAIR_FOOTHOLD_MAX_DELTA_X = 0.22
STAIR_REAR_FOOTHOLD_MAX_DELTA_X = 0.30
STAIR_SEQUENCE_FOOTHOLDS = True
STAIR_FRONT_STEP_LEAD_MAX = 1
STAIR_FRONT_STEP_LEAD_MAX_WHILE_REAR_GROUND = 1
STAIR_PREVENT_FRONT_STEP_DOWN = False
STAIR_FRONT_STEP_DOWN_ENTRY_EXTRA_X = 0.005
STAIR_FORCE_REAR_STEP_UP = True
STAIR_REAR_STEP_UP_BASE_MARGIN_X = 0.00
STAIR_REAR_STEP_UP_MAX_GAP = 999
STAIR_FORCE_REAR_TO_TREAD_ENTRY = False
STAIR_STEP_WAIT_X_VEL_LIMIT = 0.018
STAIR_STEP_WAIT_MAX_LEAD = 0
STAIR_END_SLOW_ZONE_X = 0.0
STAIR_END_STOP_MARGIN_X = 0.0
STAIR_END_X_VEL_LIMIT = 0.0
STAIR_BODY_FOOT_SUPPORT_MIN_COUNT = 1
STAIR_BODY_FOOT_NEAR_SURFACE_Z = 0.035
STAIR_POSTURE_REFERENCE = False
STAIR_SUPPORT_MARGIN_CONTROL = False
STAIR_PITCH_GAIN = 0.45
STAIR_PITCH_LIMIT = 0.14
STAIR_PITCH_RAMP = 0.20
STAIR_SUPPORT_MARGIN_X = 0.05
STAIR_SUPPORT_SLOW_MARGIN_X = 0.10
STAIR_SUPPORT_X_VEL_LIMIT = 0.012
STAIR_SUPPORT_POLYGON_MARGIN = 0.0
STAIR_SUPPORT_POLYGON_SLOW_MARGIN = 0.025
STAIR_SUPPORT_Y_MARGIN = 0.030
STAIR_SUPPORT_Y_KP = 0.8
STAIR_SUPPORT_Y_VEL_LIMIT = 0.05
STAIR_YAW_CENTER_KP = 0.0
STAIR_YAW_RATE_LIMIT = 0.08
STAIR_TERRAIN_AWARE_MPC_REFERENCE = False
STAIR_MPC_TERRAIN_PITCH_REFERENCE = True
STAIR_MPC_Z_TRAJ_RAMP = 0.08
STAIR_MPC_PITCH_TRAJ_RAMP = 0.22
STAIR_MPC_REFERENCE_FRONT_X = 0.18
STAIR_MPC_REFERENCE_REAR_X = -0.20
STAIR_TERRAIN_EPS_X = 1e-9

# Structural stair mode: once the body reaches the stair approach, slow the
# forward command, pull harder toward the centerline, and keep footholds inside
# the tread corridor instead of relying only on keyboard velocity limits.
STAIR_MODE_APPROACH_X = 0.25
STAIR_MODE_EXIT_X = 0.20
STAIR_MODE_HYSTERESIS_X = 0.08
STAIR_APPROACH_SPEED_ZONE_X = 0.25
STAIR_APPROACH_X_VEL_LIMIT = 0.04
STAIR_X_VEL_LIMIT = 0.025
STAIR_RECOVERY_X_VEL_LIMIT = 0.010
STAIR_RECOVERY_Y = 0.12
STAIR_RECOVERY_ROLL = 0.28
STAIR_Y_CENTER_KP = 1.4
STAIR_Y_CENTER_VEL_LIMIT = 0.16
STAIR_FOOTHOLD_Y_LIMIT = 0.30
RESET_GAIT_ON_STAIR_ENTRY = False
STAIR_MOVE_ONLY_ALL_STANCE = False
STAIR_SWING_X_VEL_LIMIT = 0.0

# Guard mode freezes forward progress and can temporarily use an all-stance
# gait when the body starts rolling or drifting sideways on the stair.
STAIR_GUARD_Y = 0.12
STAIR_GUARD_Y_RELEASE = 0.06
STAIR_GUARD_ROLL = 0.28
STAIR_GUARD_ROLL_RELEASE = 0.14
STAIR_GUARD_X_VEL_LIMIT = 0.0
STAIR_GUARD_Y_CENTER_KP = 1.0
STAIR_GUARD_Y_CENTER_VEL_LIMIT = 0.08
STAIR_GUARD_USE_STAND_GAIT = False
STAIR_GUARD_REQUIRE_IDLE_RELEASE = False

# Keep the robot near the stair centerline. The keyboard command remains small,
# but this feedback lets the gait correct lateral drift before it turns into roll.
Y_CENTER_KP = 0.8
Y_CENTER_VEL_LIMIT = 0.12

# LegController maps the MPC contact force to stance torque. Add the local joint
# bias term in EX18 for support legs so gravity/Coriolis are not left to the MPC
# force term alone.
STANCE_BIAS_COMP = True
COM_TRAJ_START_AT_CURRENT = True

# -----------------------------
# Keyboard command limits
# -----------------------------

X_VEL_STEP = 0.02
Y_VEL_STEP = 0.01
YAW_RATE_STEP = 0.06

X_VEL_LIMIT = 0.04
Y_VEL_LIMIT = 0.02
YAW_RATE_LIMIT = 0.12

X_VEL_RAMP = 0.08
Y_VEL_RAMP = 0.04
YAW_RATE_RAMP = 0.18

LOG_HZ = 20.0
LOG_CTRL_DECIM = max(1, int(CTRL_HZ // LOG_HZ))
STATUS_HZ = 1.0
STATUS_CTRL_DECIM = max(1, int(CTRL_HZ // STATUS_HZ))

# -----------------------------
# Torque limits
# -----------------------------

SAFETY = 0.9
TAU_LIM = SAFETY * np.array([
    17.0, 17.0, 34.0,  # FL
    17.0, 17.0, 34.0,  # FR
    17.0, 17.0, 34.0,  # RL
    17.0, 17.0, 34.0,  # RR
])

LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}
LEG_INDEX = {leg: i for i, leg in enumerate(MPC_LEG_ORDER)}

KP_STAND = np.array([
    40.0, 80.0, 80.0,  # FL
    40.0, 80.0, 80.0,  # FR
    40.0, 80.0, 80.0,  # RL
    40.0, 80.0, 80.0,  # RR
])
KD_STAND = np.array([
    2.0, 3.0, 3.0,  # FL
    2.0, 3.0, 3.0,  # FR
    2.0, 3.0, 3.0,  # RL
    2.0, 3.0, 3.0,  # RR
])


@dataclass(frozen=True)
class StairTerrain:
    """Analytic height profile matching vbot_mpc_scene_stairs.xml."""

    start_x: float = 1.20
    step_depth: float = 0.26
    step_height: float = 0.04
    step_count: int = 5
    step_heights: tuple[float, ...] = ()
    landing_depth: float = 0.52
    half_width_y: float = 0.55

    def __post_init__(self):
        if self.step_heights:
            heights = tuple(float(h) for h in self.step_heights)
        else:
            heights = tuple([float(self.step_height)] * int(self.step_count))
        object.__setattr__(self, "step_heights", heights)
        object.__setattr__(self, "step_count", len(heights))

    def cumulative_height(self, step_index: int) -> float:
        step_index = int(np.clip(step_index, 0, self.step_count))
        return float(np.sum(self.step_heights[:step_index]))

    def _stair_tread_index0(self, x: float) -> int | None:
        x_rel = float(x) - self.start_x
        if x_rel < -STAIR_TERRAIN_EPS_X:
            return None

        stairs_length = self.step_count * self.step_depth
        if x_rel < stairs_length - STAIR_TERRAIN_EPS_X:
            step_i = np.floor((x_rel + STAIR_TERRAIN_EPS_X) / self.step_depth)
            return int(np.clip(step_i, 0, self.step_count - 1))

        return None

    def height(self, x: float, y: float) -> float:
        if abs(y) > self.half_width_y:
            return 0.0

        x_rel = float(x) - self.start_x
        if x_rel < -STAIR_TERRAIN_EPS_X:
            return 0.0

        stairs_length = self.step_count * self.step_depth
        step_i = self._stair_tread_index0(x)
        if step_i is not None:
            return self.cumulative_height(step_i + 1)

        if x_rel < stairs_length + self.landing_depth + STAIR_TERRAIN_EPS_X:
            return self.cumulative_height(self.step_count)

        return 0.0

    def step_index_at(self, x: float, y: float) -> int:
        if abs(float(y)) > self.half_width_y:
            return 0

        x_rel = float(x) - self.start_x
        if x_rel < -STAIR_TERRAIN_EPS_X:
            return 0

        stairs_length = self.step_count * self.step_depth
        step_i = self._stair_tread_index0(x)
        if step_i is not None:
            return int(step_i + 1)

        if x_rel < stairs_length + self.landing_depth + STAIR_TERRAIN_EPS_X:
            return self.step_count

        return 0

    def tread_center_x(self, step_index: int) -> float:
        step_index = int(np.clip(step_index, 1, self.step_count))
        return self.start_x + (step_index - 0.5) * self.step_depth

    def end_x(self) -> float:
        return self.start_x + self.step_count * self.step_depth + self.landing_depth

    def in_stair_mode_x(self, x: float) -> bool:
        return self.start_x - STAIR_MODE_APPROACH_X <= float(x) <= self.end_x() + STAIR_MODE_EXIT_X

    def body_support_height_xy(self, x: float, y: float) -> float:
        front_x = STAIR_BODY_SUPPORT_FRONT_X
        samples = (
            (front_x, 0.08),
            (front_x, -0.08),
            (-0.20, 0.08),
            (-0.20, -0.08),
            (0.0, 0.0),
        )
        return max(self.height(float(x) + dx, float(y) + dy) for dx, dy in samples)

    def body_support_height(self, base_pos: np.ndarray) -> float:
        x = float(base_pos[0])
        y = float(base_pos[1])
        return self.body_support_height_xy(x, y)

    def feet_support_height(self, foot_positions: dict[str, np.ndarray]) -> float:
        heights = []
        for foot_pos in foot_positions.values():
            pos = np.asarray(foot_pos, dtype=float).reshape(3)
            terrain_h = self.height(pos[0], pos[1])
            foot_surface_h = float(pos[2] - FOOT_TOUCHDOWN_CLEARANCE)
            near_surface = abs(foot_surface_h - terrain_h) <= STAIR_BODY_FOOT_NEAR_SURFACE_Z
            if near_surface:
                heights.append(max(0.0, foot_surface_h))
        if not heights:
            return 0.0
        heights = sorted(heights)
        support_i = min(max(1, int(STAIR_BODY_FOOT_SUPPORT_MIN_COUNT)), len(heights))
        return heights[-support_i]


class StairAwareGait(Gait):
    """Gait wrapper that keeps planned footholds away from stair edges."""

    def __init__(self, frequency_hz, duty, terrain: StairTerrain, swing_height=SWING_HEIGHT, phase_offset=None):
        super().__init__(frequency_hz, duty, swing_height=swing_height, phase_offset=phase_offset)
        self.terrain = terrain
        self.last_swing_debug = {}

    def _planned_step_limits(self, go2: PinVBotModel | None, leg: str | None):
        if not STAIR_SEQUENCE_FOOTHOLDS or go2 is None or leg is None:
            return None, None

        foot_steps = {}
        for name in MPC_LEG_ORDER:
            foot_pos, _ = go2.get_single_foot_state_in_world(name)
            foot_steps[name] = self.terrain.step_index_at(foot_pos[0], foot_pos[1])

        front_min = min(foot_steps["FL"], foot_steps["FR"])
        rear_min = min(foot_steps["RL"], foot_steps["RR"])
        base_x = float(np.asarray(go2.current_config.base_pos).reshape(3)[0])

        if leg in ("FL", "FR"):
            lead_max = max(1, int(STAIR_FRONT_STEP_LEAD_MAX))
            if rear_min <= 0:
                lead_max = max(1, int(STAIR_FRONT_STEP_LEAD_MAX_WHILE_REAR_GROUND))
            return None, min(self.terrain.step_count, rear_min + lead_max)

        if leg in ("RL", "RR") and front_min >= 1:
            max_gap = int(STAIR_REAR_STEP_UP_MAX_GAP)
            if max_gap >= 0 and front_min - foot_steps[leg] > max_gap:
                forced_rear_step = max(1, front_min - max_gap)
                step_start_x = self.terrain.start_x + (forced_rear_step - 1) * self.terrain.step_depth
                if base_x >= step_start_x + STAIR_REAR_STEP_UP_BASE_MARGIN_X:
                    return forced_rear_step, forced_rear_step

            step_start_x = self.terrain.start_x + (front_min - 1) * self.terrain.step_depth
            if (
                STAIR_FORCE_REAR_STEP_UP
                and base_x >= step_start_x + STAIR_REAR_STEP_UP_BASE_MARGIN_X
                and foot_steps[leg] < front_min
            ):
                return front_min, front_min
            return None, front_min

        return None, None

    def _snap_touchdown_to_tread(
        self,
        pos_touchdown_world: np.ndarray,
        go2: PinVBotModel | None = None,
        leg: str | None = None,
    ) -> np.ndarray:
        pos = np.asarray(pos_touchdown_world, dtype=float).reshape(3).copy()
        if not self.terrain.in_stair_mode_x(pos[0]) and abs(pos[1]) > self.terrain.half_width_y:
            return pos

        x = float(pos[0])
        start = self.terrain.start_x
        depth = self.terrain.step_depth
        stairs_end = start + self.terrain.step_count * depth
        landing_end = stairs_end + self.terrain.landing_depth
        margin = min(STAIR_FOOTHOLD_MARGIN_X, 0.45 * depth)
        forced_step, max_step = self._planned_step_limits(go2, leg)

        if forced_step is not None:
            if STAIR_FORCE_REAR_TO_TREAD_ENTRY and leg in ("RL", "RR"):
                x = start + (forced_step - 1) * depth + margin
            else:
                x = self.terrain.tread_center_x(forced_step)

        if (
            STAIR_ACTIVE_FRONT_FOOTHOLD_ADVANCE_X > 0.0
            and leg in ("FL", "FR")
            and self.terrain.in_stair_mode_x(x)
        ):
            lead_limit = self.terrain.step_count if max_step is None else int(max_step)
            current_step = self.terrain.step_index_at(x, pos[1])
            candidate_step = None
            if current_step <= 0 and start - STAIR_ACTIVE_FRONT_ADVANCE_WINDOW_X <= x < start:
                candidate_step = 1
            elif 1 <= current_step < lead_limit:
                next_riser_x = start + current_step * depth
                if 0.0 <= next_riser_x - x <= STAIR_ACTIVE_FRONT_ADVANCE_WINDOW_X:
                    candidate_step = current_step + 1
            if candidate_step is not None and candidate_step <= lead_limit:
                tread_entry_x = (
                    start
                    + (candidate_step - 1) * depth
                    + margin
                    + STAIR_ACTIVE_FRONT_TREAD_ENTRY_EXTRA_X
                )
                x = min(max(x, tread_entry_x), x + STAIR_ACTIVE_FRONT_FOOTHOLD_ADVANCE_X)

        if STAIR_FOOTHOLD_SOFT_PROJECTION and go2 is not None and leg is not None:
            foot_pos, _ = go2.get_single_foot_state_in_world(leg)
            max_delta_x = (
                STAIR_REAR_FOOTHOLD_MAX_DELTA_X
                if leg in ("RL", "RR")
                else STAIR_FOOTHOLD_MAX_DELTA_X
            )
            x = min(x, float(foot_pos[0]) + max_delta_x)

        if start - margin <= x < start:
            x = start + margin
            if STAIR_FOOTHOLD_CENTERING_GAIN > 0.0 and not STAIR_FOOTHOLD_SOFT_PROJECTION:
                center = start + 0.5 * depth
                x = x + STAIR_FOOTHOLD_CENTERING_GAIN * (center - x)
        elif start <= x < stairs_end:
            step_i = int(np.clip(np.floor((x - start) / depth), 0, self.terrain.step_count - 1))
            x_min = start + step_i * depth + margin
            x_max = start + (step_i + 1) * depth - margin
            if STAIR_FOOTHOLD_CENTERING_GAIN > 0.0 and not STAIR_FOOTHOLD_SOFT_PROJECTION:
                center = start + (step_i + 0.5) * depth
                x = x + STAIR_FOOTHOLD_CENTERING_GAIN * (center - x)
            x = float(np.clip(x, x_min, x_max))
        elif stairs_end <= x < landing_end:
            x_min = stairs_end + margin
            x_max = landing_end - margin
            if STAIR_FOOTHOLD_CENTERING_GAIN > 0.0 and not STAIR_FOOTHOLD_SOFT_PROJECTION:
                center = 0.5 * (x_min + x_max)
                x = x + STAIR_FOOTHOLD_CENTERING_GAIN * (center - x)
            if x_min < x_max:
                x = float(np.clip(x, x_min, x_max))

        if max_step is not None:
            step_index = self.terrain.step_index_at(x, pos[1])
            if step_index > max_step:
                if STAIR_FOOTHOLD_SOFT_PROJECTION:
                    x = start + max_step * depth - margin
                else:
                    x = self.terrain.tread_center_x(max_step)

        if (
            STAIR_PREVENT_FRONT_STEP_DOWN
            and go2 is not None
            and leg in ("FL", "FR")
            and self.terrain.in_stair_mode_x(x)
        ):
            foot_pos, _ = go2.get_single_foot_state_in_world(leg)
            foot_step = self.terrain.step_index_at(foot_pos[0], foot_pos[1])
            proposed_step = self.terrain.step_index_at(x, pos[1])
            if foot_step > 0 and proposed_step < foot_step:
                tread_entry_x = (
                    start
                    + (foot_step - 1) * depth
                    + margin
                    + STAIR_FRONT_STEP_DOWN_ENTRY_EXTRA_X
                )
                x = max(x, tread_entry_x)

        pos[0] = x
        if self.terrain.in_stair_mode_x(pos[0]):
            pos[1] = float(np.clip(pos[1], -STAIR_FOOTHOLD_Y_LIMIT, STAIR_FOOTHOLD_Y_LIMIT))
        pos[2] = self.terrain.height(pos[0], pos[1]) + FOOT_TOUCHDOWN_CLEARANCE
        return pos

    def _terrain_aware_swing_trajectory(self, p0, pf, t_swing):
        p0 = np.asarray(p0, dtype=float).reshape(3)
        pf = np.asarray(pf, dtype=float).reshape(3)
        T = float(t_swing)
        dp = pf - p0

        sample_s = np.linspace(0.0, 1.0, 7)
        terrain_peak = 0.0
        for s in sample_s:
            p_xy = p0[:2] + (pf[:2] - p0[:2]) * s
            terrain_peak = max(terrain_peak, self.terrain.height(p_xy[0], p_xy[1]))
        apex_z = max(
            p0[2],
            pf[2],
            terrain_peak + FOOT_TOUCHDOWN_CLEARANCE + STAIR_SWING_TERRAIN_CLEARANCE,
        )
        mid_z = 0.5 * (p0[2] + pf[2])
        z_bump = max(0.0, apex_z - mid_z)

        def eval_at(t):
            s = np.clip(t / T, 0.0, 1.0)
            mj = 10 * s**3 - 15 * s**4 + 6 * s**5
            dmj = 30 * s**2 - 60 * s**3 + 30 * s**4
            d2mj = 60 * s - 180 * s**2 + 120 * s**3

            p = p0 + dp * mj
            v = dp * dmj / T
            a = dp * d2mj / (T**2)

            b = 64 * s**3 * (1 - s) ** 3
            db = 192 * s**2 * (1 - s) ** 2 * (1 - 2 * s)
            d2b = 192 * (
                2 * s * (1 - s) ** 2 * (1 - 2 * s)
                - 2 * s**2 * (1 - s) * (1 - 2 * s)
                - 2 * s**2 * (1 - s) ** 2
            )
            p[2] += z_bump * b
            v[2] += z_bump * db / T
            a[2] += z_bump * d2b / (T**2)
            return p, v, a

        return eval_at

    @staticmethod
    def _min_jerk_terms(s: float):
        s = float(np.clip(s, 0.0, 1.0))
        mj = 10 * s**3 - 15 * s**4 + 6 * s**5
        dmj = 30 * s**2 - 60 * s**3 + 30 * s**4
        d2mj = 60 * s - 180 * s**2 + 120 * s**3
        return mj, dmj, d2mj

    @staticmethod
    def _beta_bump_terms(s: float, peak_phase: float):
        s = float(np.clip(s, 0.0, 1.0))
        peak_phase = float(np.clip(peak_phase, 0.05, 0.95))
        sharpness = max(4.5, float(STAIR_ACTIVE_SWING_BETA_SHARPNESS))
        a = max(2.2, sharpness * peak_phase)
        b = max(2.2, sharpness * (1.0 - peak_phase))
        peak = a / (a + b)
        norm = max(1e-12, (peak**a) * ((1.0 - peak) ** b))
        if s <= 1e-9 or s >= 1.0 - 1e-9:
            return 0.0, 0.0, 0.0
        bump = (s**a) * ((1.0 - s) ** b) / norm
        log_d1 = a / s - b / (1.0 - s)
        log_d2 = -a / (s**2) - b / ((1.0 - s) ** 2)
        dbump = bump * log_d1
        d2bump = bump * (log_d1**2 + log_d2)
        return float(bump), float(dbump), float(d2bump)

    def _active_stair_swing_trajectory(
        self,
        p0,
        pf,
        t_swing,
        takeoff_step: int,
        touchdown_step: int,
    ):
        p0 = np.asarray(p0, dtype=float).reshape(3)
        pf = np.asarray(pf, dtype=float).reshape(3)
        T = float(t_swing)
        dp = pf - p0

        dx = float(pf[0] - p0[0])
        riser_x = self.terrain.start_x + (int(touchdown_step) - 1) * self.terrain.step_depth
        if abs(dx) > 1e-6:
            riser_phase = float(np.clip((riser_x - float(p0[0])) / dx, 0.05, 0.95))
        else:
            riser_phase = 0.50
        peak_phase = float(
            np.clip(
                riser_phase - STAIR_ACTIVE_SWING_RISER_PHASE_LEAD,
                STAIR_ACTIVE_SWING_MIN_PEAK_PHASE,
                STAIR_ACTIVE_SWING_MAX_PEAK_PHASE,
            )
        )

        sample_s = np.linspace(0.0, 1.0, int(max(3, STAIR_ACTIVE_SWING_SAMPLE_COUNT)))
        terrain_peak = 0.0
        for s_sample in sample_s:
            p_xy = p0[:2] + (pf[:2] - p0[:2]) * s_sample
            terrain_peak = max(terrain_peak, self.terrain.height(p_xy[0], p_xy[1]))

        takeoff_h = self.terrain.cumulative_height(takeoff_step)
        touchdown_h = self.terrain.cumulative_height(touchdown_step)
        step_delta = max(0.0, touchdown_h - takeoff_h)
        swing_clearance = max(
            STAIR_ACTIVE_SWING_CLEARANCE,
            STAIR_ACTIVE_SWING_RISER_CLEARANCE + 0.25 * step_delta,
        )
        riser_clearance = max(
            STAIR_ACTIVE_SWING_RISER_CLEARANCE,
            0.55 * step_delta + 0.025,
        )
        required_peak_z = max(
            p0[2],
            pf[2],
            terrain_peak + FOOT_TOUCHDOWN_CLEARANCE + swing_clearance,
            touchdown_h + FOOT_TOUCHDOWN_CLEARANCE + riser_clearance,
        )

        mj_peak, _, _ = self._min_jerk_terms(peak_phase)
        base_peak_z = float(p0[2] + dp[2] * mj_peak)
        bump_at_peak, _, _ = self._beta_bump_terms(peak_phase, peak_phase)
        bump_height = max(0.0, (required_peak_z - base_peak_z) / max(0.10, bump_at_peak))

        mj_riser, _, _ = self._min_jerk_terms(riser_phase)
        base_riser_z = float(p0[2] + dp[2] * mj_riser)
        bump_at_riser, _, _ = self._beta_bump_terms(riser_phase, peak_phase)
        required_riser_z = touchdown_h + FOOT_TOUCHDOWN_CLEARANCE + riser_clearance
        bump_height = max(
            bump_height,
            max(0.0, required_riser_z - base_riser_z) / max(0.10, bump_at_riser),
        )

        dense_s = np.linspace(0.0, 1.0, 51)
        base_z_values = np.asarray(
            [float(p0[2] + dp[2] * self._min_jerk_terms(s_sample)[0]) for s_sample in dense_s],
            dtype=float,
        )
        bump_values = np.asarray(
            [self._beta_bump_terms(s_sample, peak_phase)[0] for s_sample in dense_s],
            dtype=float,
        )
        max_apex_z = max(
            float(p0[2]),
            float(pf[2]),
            terrain_peak + FOOT_TOUCHDOWN_CLEARANCE + STAIR_ACTIVE_SWING_MAX_TERRAIN_CLEARANCE,
        )
        if bump_height > 0.0 and np.max(bump_values) > 1e-6:
            max_base_z = float(np.max(base_z_values))
            max_bump = float(np.max(bump_values))
            bump_height = min(
                bump_height,
                max(0.0, (max_apex_z - max_base_z) / max_bump),
            )

        apex_z = max(
            float(p0[2]),
            float(pf[2]),
            max(
                float(base_z) + bump_height * float(bump)
                for base_z, bump in zip(base_z_values, bump_values)
            ),
        )
        debug = {
            "takeoff_step": int(takeoff_step),
            "touchdown_step": int(touchdown_step),
            "swing_apex_z": float(apex_z),
            "swing_clearance": float(max(0.0, apex_z - terrain_peak - FOOT_TOUCHDOWN_CLEARANCE)),
        }

        def eval_at(t):
            s = np.clip(t / T, 0.0, 1.0)
            mj, dmj, d2mj = self._min_jerk_terms(s)

            p = p0 + dp * mj
            v = dp * dmj / T
            a = dp * d2mj / (T**2)

            bump, dbump, d2bump = self._beta_bump_terms(s, peak_phase)
            p[2] += bump_height * bump
            v[2] += bump_height * dbump / T
            a[2] += bump_height * d2bump / (T**2)
            return p, v, a

        return eval_at, debug

    def compute_touchdown_world_for_traj_purpose_only(
        self,
        go2: PinVBotModel,
        leg: str,
        terrain_height_fn=None,
    ):
        pos_touchdown_world = super().compute_touchdown_world_for_traj_purpose_only(
            go2,
            leg,
            terrain_height_fn=terrain_height_fn,
        )
        return self._snap_touchdown_to_tread(pos_touchdown_world, go2, leg)

    def compute_swing_traj_and_touchdown(self, go2: PinVBotModel, leg: str):
        _, pos_touchdown_world = super().compute_swing_traj_and_touchdown(go2, leg)
        nominal_touchdown_world = np.asarray(pos_touchdown_world, dtype=float).reshape(3).copy()
        pos_touchdown_world = self._snap_touchdown_to_tread(pos_touchdown_world, go2, leg)
        foot_pos, _ = go2.get_single_foot_state_in_world(leg)
        takeoff_step = self.terrain.step_index_at(foot_pos[0], foot_pos[1])
        touchdown_step = self.terrain.step_index_at(pos_touchdown_world[0], pos_touchdown_world[1])
        foothold_projected = bool(
            np.linalg.norm(pos_touchdown_world[:2] - nominal_touchdown_world[:2]) > 1e-5
        )
        debug = {
            "takeoff_step": int(takeoff_step),
            "touchdown_step": int(touchdown_step),
            "swing_apex_z": float("nan"),
            "swing_clearance": float("nan"),
            "foothold_projected": foothold_projected,
        }
        if USE_ACTIVE_STAIR_SWING and touchdown_step > takeoff_step:
            foot_traj, active_debug = self._active_stair_swing_trajectory(
                foot_pos,
                pos_touchdown_world,
                self.swing_time,
                takeoff_step,
                touchdown_step,
            )
            debug.update(active_debug)
        elif USE_TERRAIN_AWARE_SWING:
            foot_traj = self._terrain_aware_swing_trajectory(
                foot_pos,
                pos_touchdown_world,
                self.swing_time,
            )
            sample_s = np.linspace(0.0, self.swing_time, 31)
            apex_z = max(float(foot_traj(t_sample)[0][2]) for t_sample in sample_s)
            terrain_peak = max(
                self.terrain.height(*(foot_pos[:2] + (pos_touchdown_world[:2] - foot_pos[:2]) * s))
                for s in np.linspace(0.0, 1.0, 9)
            )
            debug["swing_apex_z"] = float(apex_z)
            debug["swing_clearance"] = float(max(0.0, apex_z - terrain_peak - FOOT_TOUCHDOWN_CLEARANCE))
        else:
            foot_traj = self.make_swing_trajectory(
                foot_pos,
                pos_touchdown_world,
                self.swing_time,
                h_sw=self.swing_height,
            )
            debug["swing_apex_z"] = float(max(foot_pos[2], pos_touchdown_world[2]) + self.swing_height)
            debug["swing_clearance"] = float(self.swing_height)
        debug["foothold_projected"] = foothold_projected
        self.last_swing_debug[leg] = debug
        return foot_traj, pos_touchdown_world


@dataclass
class KeyboardCommand:
    target_x_vel: float = 0.0
    target_y_vel: float = 0.0
    target_yaw_rate: float = 0.0
    x_vel: float = 0.0
    y_vel: float = 0.0
    yaw_rate: float = 0.0
    z_pos: float = NOMINAL_BODY_HEIGHT
    pitch: float = 0.0
    reset_requested: bool = False
    exit_requested: bool = False

    def clamp(self):
        self.target_x_vel = float(np.clip(self.target_x_vel, -X_VEL_LIMIT, X_VEL_LIMIT))
        self.target_y_vel = float(np.clip(self.target_y_vel, -Y_VEL_LIMIT, Y_VEL_LIMIT))
        self.target_yaw_rate = float(np.clip(self.target_yaw_rate, -YAW_RATE_LIMIT, YAW_RATE_LIMIT))

    def stop(self):
        self.target_x_vel = 0.0
        self.target_y_vel = 0.0
        self.target_yaw_rate = 0.0
        self.x_vel = 0.0
        self.y_vel = 0.0
        self.yaw_rate = 0.0
        self.pitch = 0.0

    def summary(self):
        return (
            f"target x={self.target_x_vel:+.2f} m/s, y={self.target_y_vel:+.2f} m/s, "
            f"yaw={self.target_yaw_rate:+.2f} rad/s | "
            f"filtered x={self.x_vel:+.2f} m/s, y={self.y_vel:+.2f} m/s, "
            f"yaw={self.yaw_rate:+.2f} rad/s | z={self.z_pos:.3f} m, "
            f"pitch_ref={self.pitch:+.3f} rad"
        )


@dataclass
class ControlRuntime:
    walking: bool = False
    stair_mode: bool = False
    stair_guard: bool = False
    gait_start_time_s: float = 0.0
    gait_ctrl_i: int = 0
    gait_time_s: float = 0.0
    x_eff: float = 0.0
    yaw_rate_eff: float = 0.0
    pitch_ref: float = 0.0
    support_margin_xy: float = np.nan
    support_margin_x: float = np.nan
    support_forward_margin_x: float = np.nan
    support_center_y: float = np.nan
    support_y_feedback: float = 0.0
    support_count: int = 0
    leg_outputs: dict = field(default_factory=dict)
    swing_debug: dict = field(default_factory=dict)

    def reset(self):
        self.walking = False
        self.stair_mode = False
        self.stair_guard = False
        self.gait_start_time_s = 0.0
        self.gait_ctrl_i = 0
        self.gait_time_s = 0.0
        self.x_eff = 0.0
        self.yaw_rate_eff = 0.0
        self.pitch_ref = 0.0
        self.support_margin_xy = np.nan
        self.support_margin_x = np.nan
        self.support_forward_margin_x = np.nan
        self.support_center_y = np.nan
        self.support_y_feedback = 0.0
        self.support_count = 0
        self.leg_outputs.clear()
        self.swing_debug.clear()


def ramp_value(value: float, target: float, rate: float, dt: float) -> float:
    max_delta = rate * dt
    return float(value + np.clip(target - value, -max_delta, max_delta))


def wrap_to_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def update_command_filter(cmd: KeyboardCommand, dt: float):
    cmd.x_vel = ramp_value(cmd.x_vel, cmd.target_x_vel, X_VEL_RAMP, dt)
    cmd.y_vel = ramp_value(cmd.y_vel, cmd.target_y_vel, Y_VEL_RAMP, dt)
    cmd.yaw_rate = ramp_value(cmd.yaw_rate, cmd.target_yaw_rate, YAW_RATE_RAMP, dt)


def command_is_idle(cmd: KeyboardCommand) -> bool:
    return (
        abs(cmd.target_x_vel) < STAND_VEL_EPS
        and abs(cmd.target_y_vel) < STAND_VEL_EPS
        and abs(cmd.target_yaw_rate) < STAND_VEL_EPS
        and abs(cmd.x_vel) < STAND_VEL_EPS
        and abs(cmd.y_vel) < STAND_VEL_EPS
        and abs(cmd.yaw_rate) < STAND_VEL_EPS
    )


class RunLogger:
    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = LOG_DIR / f"{LOG_PREFIX}_{stamp}.csv"
        self.file = self.path.open("w", newline="")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "time_s",
                "gait_time_s",
                "target_x_vel",
                "target_y_vel",
                "target_yaw_rate",
                "x_vel",
                "x_vel_eff",
                "y_vel",
                "y_vel_eff",
                "y_center_feedback",
                "yaw_rate",
                "yaw_rate_eff",
                "z_cmd",
                "pitch_cmd",
                "support_h",
                "foot_support_h",
                "support_count",
                "support_margin_xy",
                "support_margin_x",
                "support_forward_margin_x",
                "support_center_y",
                "support_y_feedback",
                "preview_h",
                "base_x",
                "base_y",
                "base_z",
                "roll",
                "pitch",
                "yaw",
                "base_vx",
                "base_vy",
                "base_vz",
                "terrain_h",
                "tau_max",
                "tau_sat_frac",
                "fl_fz",
                "fr_fz",
                "rl_fz",
                "rr_fz",
                "fl_tau_max",
                "fr_tau_max",
                "rl_tau_max",
                "rr_tau_max",
                "fl_foot_err",
                "fr_foot_err",
                "rl_foot_err",
                "rr_foot_err",
                "fl_foot_z_err",
                "fr_foot_z_err",
                "rl_foot_z_err",
                "rr_foot_z_err",
                "fl_foot_des_z",
                "fr_foot_des_z",
                "rl_foot_des_z",
                "rr_foot_des_z",
                "fl_foot_now_z",
                "fr_foot_now_z",
                "rl_foot_now_z",
                "rr_foot_now_z",
                "fl_foot_des_x",
                "fr_foot_des_x",
                "rl_foot_des_x",
                "rr_foot_des_x",
                "fl_foot_now_x",
                "fr_foot_now_x",
                "rl_foot_now_x",
                "rr_foot_now_x",
                "fl_takeoff_step",
                "fr_takeoff_step",
                "rl_takeoff_step",
                "rr_takeoff_step",
                "fl_touchdown_step",
                "fr_touchdown_step",
                "rl_touchdown_step",
                "rr_touchdown_step",
                "fl_swing_apex_z",
                "fr_swing_apex_z",
                "rl_swing_apex_z",
                "rr_swing_apex_z",
                "fl_swing_clearance",
                "fr_swing_clearance",
                "rl_swing_clearance",
                "rr_swing_clearance",
                "fl_foothold_projected",
                "fr_foothold_projected",
                "rl_foothold_projected",
                "rr_foothold_projected",
                "mpc_update_ms",
                "mpc_solve_ms",
                "contact_mask",
                "stair_mode",
                "stair_guard",
                "mode",
            ],
        )
        self.writer.writeheader()

    def log(
        self,
        time_s,
        vbot,
        cmd,
        terrain,
        tau_cmd,
        mpc,
        mode,
        mpc_force_now=None,
        y_eff=0.0,
        runtime=None,
        contact_mask=None,
    ):
        base_pos = np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)
        base_vel = np.asarray(vbot.current_config.base_vel, dtype=float).reshape(3)
        roll, pitch, yaw = vbot.current_config.compute_euler_angle_world()
        tau_abs = np.abs(np.asarray(tau_cmd, dtype=float).reshape(12))
        mpc_force = np.zeros(12, dtype=float)
        if mpc_force_now is not None:
            mpc_force = np.asarray(mpc_force_now, dtype=float).reshape(12)
        support_h = terrain.body_support_height(base_pos)
        mask = None
        if contact_mask is not None:
            mask = np.asarray(contact_mask, dtype=int).reshape(-1)
        foot_positions = {}
        for leg_i, leg in enumerate(MPC_LEG_ORDER):
            if mask is not None and leg_i < mask.size and int(mask[leg_i]) != 1:
                continue
            foot_positions[leg] = vbot.get_single_foot_state_in_world(leg)[0]
        foot_support_h = terrain.feet_support_height(foot_positions)
        preview_h = terrain.height(base_pos[0] + STAIR_HEIGHT_PREVIEW_X, base_pos[1])
        row = {
            "time_s": f"{time_s:.4f}",
            "gait_time_s": f"{float(getattr(runtime, 'gait_time_s', 0.0)):.4f}",
            "target_x_vel": f"{cmd.target_x_vel:.4f}",
            "target_y_vel": f"{cmd.target_y_vel:.4f}",
            "target_yaw_rate": f"{cmd.target_yaw_rate:.4f}",
            "x_vel": f"{cmd.x_vel:.4f}",
            "x_vel_eff": f"{float(getattr(runtime, 'x_eff', cmd.x_vel)):.4f}",
            "y_vel": f"{cmd.y_vel:.4f}",
            "y_vel_eff": f"{float(y_eff):.4f}",
            "y_center_feedback": f"{float(y_eff - cmd.y_vel):.4f}",
            "yaw_rate": f"{cmd.yaw_rate:.4f}",
            "yaw_rate_eff": f"{float(getattr(runtime, 'yaw_rate_eff', cmd.yaw_rate)):.4f}",
            "z_cmd": f"{cmd.z_pos:.4f}",
            "pitch_cmd": f"{cmd.pitch:.4f}",
            "support_h": f"{support_h:.4f}",
            "foot_support_h": f"{foot_support_h:.4f}",
            "support_count": f"{int(getattr(runtime, 'support_count', 0))}",
            "support_margin_xy": f"{float(getattr(runtime, 'support_margin_xy', np.nan)):.4f}",
            "support_margin_x": f"{float(getattr(runtime, 'support_margin_x', np.nan)):.4f}",
            "support_forward_margin_x": f"{float(getattr(runtime, 'support_forward_margin_x', np.nan)):.4f}",
            "support_center_y": f"{float(getattr(runtime, 'support_center_y', np.nan)):.4f}",
            "support_y_feedback": f"{float(getattr(runtime, 'support_y_feedback', 0.0)):.4f}",
            "preview_h": f"{preview_h:.4f}",
            "base_x": f"{base_pos[0]:.4f}",
            "base_y": f"{base_pos[1]:.4f}",
            "base_z": f"{base_pos[2]:.4f}",
            "roll": f"{roll:.4f}",
            "pitch": f"{pitch:.4f}",
            "yaw": f"{yaw:.4f}",
            "base_vx": f"{base_vel[0]:.4f}",
            "base_vy": f"{base_vel[1]:.4f}",
            "base_vz": f"{base_vel[2]:.4f}",
            "terrain_h": f"{support_h:.4f}",
            "tau_max": f"{float(np.max(tau_abs)):.4f}",
            "tau_sat_frac": f"{float(np.mean(tau_abs >= 0.98 * TAU_LIM)):.4f}",
            "mpc_update_ms": f"{float(getattr(mpc, 'update_time', 0.0)):.4f}",
            "mpc_solve_ms": f"{float(getattr(mpc, 'solve_time', 0.0)):.4f}",
            "contact_mask": "".join(str(int(v)) for v in np.asarray(contact_mask if contact_mask is not None else [0, 0, 0, 0]).reshape(4)),
            "stair_mode": int(bool(getattr(runtime, "stair_mode", False))),
            "stair_guard": int(bool(getattr(runtime, "stair_guard", False))),
            "mode": mode,
        }
        for leg, leg_slice in LEG_SLICE.items():
            prefix = leg.lower()
            row[f"{prefix}_fz"] = f"{float(mpc_force[leg_slice][2]):.4f}"
            row[f"{prefix}_tau_max"] = f"{float(np.max(tau_abs[leg_slice])):.4f}"
            out = getattr(runtime, "leg_outputs", {}).get(leg) if runtime is not None else None
            if out is None:
                row[f"{prefix}_foot_err"] = "0.0000"
                row[f"{prefix}_foot_z_err"] = "0.0000"
                row[f"{prefix}_foot_des_z"] = "0.0000"
                row[f"{prefix}_foot_now_z"] = "0.0000"
                row[f"{prefix}_foot_des_x"] = "0.0000"
                row[f"{prefix}_foot_now_x"] = "0.0000"
            else:
                foot_des = np.asarray(out.pos_des, dtype=float).reshape(3)
                foot_now = np.asarray(out.pos_now, dtype=float).reshape(3)
                foot_err = foot_des - foot_now
                row[f"{prefix}_foot_err"] = f"{float(np.linalg.norm(foot_err)):.4f}"
                row[f"{prefix}_foot_z_err"] = f"{float(foot_err[2]):.4f}"
                row[f"{prefix}_foot_des_z"] = f"{float(foot_des[2]):.4f}"
                row[f"{prefix}_foot_now_z"] = f"{float(foot_now[2]):.4f}"
                row[f"{prefix}_foot_des_x"] = f"{float(foot_des[0]):.4f}"
                row[f"{prefix}_foot_now_x"] = f"{float(foot_now[0]):.4f}"
            swing_debug = getattr(runtime, "swing_debug", {}).get(leg, {}) if runtime is not None else {}
            row[f"{prefix}_takeoff_step"] = str(int(swing_debug.get("takeoff_step", -1)))
            row[f"{prefix}_touchdown_step"] = str(int(swing_debug.get("touchdown_step", -1)))
            row[f"{prefix}_swing_apex_z"] = f"{float(swing_debug.get('swing_apex_z', np.nan)):.4f}"
            row[f"{prefix}_swing_clearance"] = f"{float(swing_debug.get('swing_clearance', np.nan)):.4f}"
            row[f"{prefix}_foothold_projected"] = int(bool(swing_debug.get("foothold_projected", False)))
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def make_key_callback(cmd: KeyboardCommand):
    def key_callback(keycode: int):
        try:
            key = chr(keycode).lower()
        except ValueError:
            key = ""

        if key == "i":
            cmd.target_x_vel += X_VEL_STEP
        elif key == "k":
            cmd.target_x_vel -= X_VEL_STEP
        elif key == "j":
            cmd.target_y_vel += Y_VEL_STEP
        elif key == "l":
            cmd.target_y_vel -= Y_VEL_STEP
        elif key == "u":
            cmd.target_yaw_rate += YAW_RATE_STEP
        elif key == "o":
            cmd.target_yaw_rate -= YAW_RATE_STEP
        elif key == "t":
            cmd.reset_requested = True
            print("\n[Keyboard] reset requested")
            return
        elif key == "x":
            cmd.stop()
            print(f"\n[Keyboard] stop -> {cmd.summary()}")
            return
        elif keycode == 256:
            cmd.exit_requested = True
            print("\n[Keyboard] exit requested")
            return
        else:
            return

        cmd.clamp()
        print(f"\n[Keyboard] command -> {cmd.summary()}")

    return key_callback


def initialize_robot(vbot: PinVBotModel, mujoco_vbot: MuJoCo_VBot_Model):
    q_init = vbot.q_init.copy()
    q_init[0], q_init[1], q_init[2] = INITIAL_X_POS, INITIAL_Y_POS, NOMINAL_BODY_HEIGHT
    q_init[3:7] = [0.0, 0.0, 0.0, 1.0]
    mujoco_vbot.update_with_q_pin(q_init)
    mujoco_vbot.data.qvel[:] = 0.0
    mujoco_vbot.data.ctrl[:] = 0.0
    mj.mj_forward(mujoco_vbot.model, mujoco_vbot.data)
    mujoco_vbot.update_pin_with_mujoco(vbot)


def compute_stand_pd_torque(vbot: PinVBotModel, mujoco_vbot: MuJoCo_VBot_Model):
    tau = np.zeros(12, dtype=float)
    g, C, _ = vbot.compute_dynamcis_terms()
    bias = C @ vbot.current_config.get_dq() + g
    for leg_i, leg in enumerate(MPC_LEG_ORDER):
        bias_leg = np.asarray(bias[vbot.get_leg_joint_vcols(leg)], dtype=float).reshape(3)
        for joint_i, joint_name in enumerate(JOINTS[leg]):
            jid = mujoco_vbot.joint_ids[joint_name]
            qadr = int(mujoco_vbot.model.jnt_qposadr[jid])
            vadr = int(mujoco_vbot.model.jnt_dofadr[jid])
            idx = 3 * leg_i + joint_i
            q = float(mujoco_vbot.data.qpos[qadr])
            dq = float(mujoco_vbot.data.qvel[vadr])
            q_des = float(DEFAULT_JOINT_ANGLES[leg][joint_i])
            tau[idx] = bias_leg[joint_i] + KP_STAND[idx] * (q_des - q) - KD_STAND[idx] * dq
    return np.clip(tau, -TAU_LIM, TAU_LIM)


def update_body_height_command(
    vbot: PinVBotModel,
    terrain: StairTerrain,
    cmd: KeyboardCommand,
    dt: float | None = None,
    contact_mask: np.ndarray | None = None,
):
    base_pos = np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)
    support_height = terrain.body_support_height(base_pos)
    preview_height = terrain.height(base_pos[0] + STAIR_HEIGHT_PREVIEW_X, base_pos[1])
    height_target = STAIR_BODY_HEIGHT_GAIN * max(support_height, preview_height)
    if STAIR_BODY_USE_FOOT_SUPPORT:
        mask = None
        if contact_mask is not None:
            mask = np.asarray(contact_mask, dtype=int).reshape(-1)
        foot_positions = {}
        for leg_i, leg in enumerate(MPC_LEG_ORDER):
            if mask is not None and leg_i < mask.size and int(mask[leg_i]) != 1:
                continue
            foot_positions[leg] = vbot.get_single_foot_state_in_world(leg)[0]
        foot_height = terrain.feet_support_height(foot_positions)
        height_target = max(height_target, STAIR_BODY_FOOT_HEIGHT_GAIN * foot_height)
    z_target = NOMINAL_BODY_HEIGHT + height_target
    if dt is None:
        cmd.z_pos = z_target
    else:
        cmd.z_pos = ramp_value(cmd.z_pos, z_target, Z_POS_RAMP, dt)
    return support_height, preview_height


def get_support_foot_samples(
    vbot: PinVBotModel,
    terrain: StairTerrain,
    contact_mask: np.ndarray | None = None,
):
    mask = None
    if contact_mask is not None:
        mask = np.asarray(contact_mask, dtype=int).reshape(-1)

    near_samples = []
    fallback_samples = []
    for leg_i, leg in enumerate(MPC_LEG_ORDER):
        if mask is not None and leg_i < mask.size and int(mask[leg_i]) != 1:
            continue
        foot_pos, _ = vbot.get_single_foot_state_in_world(leg)
        pos = np.asarray(foot_pos, dtype=float).reshape(3)
        terrain_h = terrain.height(pos[0], pos[1])
        foot_surface_h = float(pos[2] - FOOT_TOUCHDOWN_CLEARANCE)
        near_surface = abs(foot_surface_h - terrain_h) <= STAIR_BODY_FOOT_NEAR_SURFACE_Z
        sample_h = max(0.0, foot_surface_h) if near_surface else terrain_h
        sample = (leg, pos, sample_h, near_surface)
        fallback_samples.append(sample)
        if near_surface:
            near_samples.append(sample)

    if len(near_samples) >= 2:
        return near_samples
    return fallback_samples


def _support_com_xy(vbot: PinVBotModel) -> np.ndarray:
    try:
        return np.asarray(vbot.compute_com_x_vec(), dtype=float).reshape(-1)[0:2]
    except Exception:
        return np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)[0:2]


def _convex_hull_xy(points: np.ndarray) -> np.ndarray:
    pts = sorted({(float(p[0]), float(p[1])) for p in np.asarray(points, dtype=float).reshape(-1, 2)})
    if len(pts) <= 1:
        return np.asarray(pts, dtype=float).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float).reshape(-1, 2)


def _point_segment_distance_xy(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def _support_polygon_margin_xy(com_xy: np.ndarray, points_xy: np.ndarray) -> float:
    points_xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if len(points_xy) < 3:
        return np.nan

    hull = _convex_hull_xy(points_xy)
    if len(hull) < 3:
        return np.nan

    edge_distances = []
    inside = True
    for i in range(len(hull)):
        a = hull[i]
        b = hull[(i + 1) % len(hull)]
        edge = b - a
        rel = com_xy - a
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        if cross < -1e-6:
            inside = False
        edge_distances.append(_point_segment_distance_xy(com_xy, a, b))
    margin = min(edge_distances)
    return float(margin if inside else -margin)


def estimate_support_pitch(samples) -> float:
    front = []
    rear = []
    for leg, pos, support_h, _ in samples:
        entry = (float(pos[0]), float(support_h))
        if leg in ("FL", "FR"):
            front.append(entry)
        elif leg in ("RL", "RR"):
            rear.append(entry)

    if not front or not rear:
        return 0.0

    front_x = float(np.mean([v[0] for v in front]))
    rear_x = float(np.mean([v[0] for v in rear]))
    front_h = float(np.mean([v[1] for v in front]))
    rear_h = float(np.mean([v[1] for v in rear]))
    dx = max(0.08, front_x - rear_x)
    pitch = STAIR_PITCH_GAIN * np.arctan2(front_h - rear_h, dx)
    return float(np.clip(pitch, -STAIR_PITCH_LIMIT, STAIR_PITCH_LIMIT))


def estimate_terrain_pitch_at_xy(terrain: StairTerrain, x: float, y: float) -> float:
    front_x = max(STAIR_MPC_REFERENCE_FRONT_X, STAIR_BODY_SUPPORT_FRONT_X)
    rear_x = STAIR_MPC_REFERENCE_REAR_X
    front_h = 0.5 * (
        terrain.height(float(x) + front_x, float(y) + 0.08)
        + terrain.height(float(x) + front_x, float(y) - 0.08)
    )
    rear_h = 0.5 * (
        terrain.height(float(x) + rear_x, float(y) + 0.08)
        + terrain.height(float(x) + rear_x, float(y) - 0.08)
    )
    dx = max(0.08, front_x - rear_x)
    pitch = STAIR_PITCH_GAIN * np.arctan2(front_h - rear_h, dx)
    return float(np.clip(pitch, -STAIR_PITCH_LIMIT, STAIR_PITCH_LIMIT))


def make_terrain_mpc_reference_functions(terrain: StairTerrain, cmd: KeyboardCommand):
    if not STAIR_TERRAIN_AWARE_MPC_REFERENCE:
        return None, None

    z_start = float(cmd.z_pos)
    pitch_start = float(cmd.pitch)

    def z_ref(xs, ys):
        refs = []
        z_prev = z_start
        for x, y in zip(np.asarray(xs, dtype=float).reshape(-1), np.asarray(ys, dtype=float).reshape(-1)):
            support_h = terrain.body_support_height_xy(x, y)
            preview_h = terrain.height(x + STAIR_HEIGHT_PREVIEW_X, y)
            target_h = STAIR_BODY_HEIGHT_GAIN * max(support_h, preview_h)
            z_target = NOMINAL_BODY_HEIGHT + target_h
            z_prev = ramp_value(z_prev, z_target, STAIR_MPC_Z_TRAJ_RAMP, MPC_DT)
            refs.append(z_prev)
        return np.asarray(refs, dtype=float)

    def pitch_ref(xs, ys):
        refs = []
        pitch_prev = pitch_start
        for x, y in zip(np.asarray(xs, dtype=float).reshape(-1), np.asarray(ys, dtype=float).reshape(-1)):
            pitch_target = estimate_terrain_pitch_at_xy(terrain, x, y) if STAIR_POSTURE_REFERENCE else 0.0
            pitch_prev = ramp_value(pitch_prev, pitch_target, STAIR_MPC_PITCH_TRAJ_RAMP, MPC_DT)
            refs.append(pitch_prev)
        return np.asarray(refs, dtype=float)

    return z_ref, pitch_ref if STAIR_MPC_TERRAIN_PITCH_REFERENCE else None


def update_stair_reference_command(
    vbot: PinVBotModel,
    terrain: StairTerrain,
    cmd: KeyboardCommand,
    runtime: ControlRuntime,
    contact_mask: np.ndarray,
    stair_mode: bool,
    x_eff: float,
    y_eff: float,
    dt: float,
) -> tuple[float, float]:
    update_body_height_command(vbot, terrain, cmd, dt, contact_mask=contact_mask)

    samples = get_support_foot_samples(vbot, terrain, contact_mask)
    points_xy = np.asarray([sample[1][0:2] for sample in samples], dtype=float).reshape(-1, 2)
    com_xy = _support_com_xy(vbot)
    runtime.support_count = len(samples)
    if len(samples) >= 2:
        xs = points_xy[:, 0]
        runtime.support_margin_x = float(min(com_xy[0] - np.min(xs), np.max(xs) - com_xy[0]))
        runtime.support_forward_margin_x = float(np.max(xs) - com_xy[0])
        runtime.support_margin_xy = _support_polygon_margin_xy(com_xy, points_xy)
        runtime.support_center_y = float(np.mean(points_xy[:, 1]))
    else:
        runtime.support_margin_x = np.nan
        runtime.support_forward_margin_x = np.nan
        runtime.support_margin_xy = np.nan
        runtime.support_center_y = np.nan
    runtime.support_y_feedback = 0.0

    pitch_target = 0.0
    if STAIR_POSTURE_REFERENCE and stair_mode:
        pitch_target = estimate_support_pitch(samples)
    cmd.pitch = ramp_value(cmd.pitch, pitch_target, STAIR_PITCH_RAMP, dt)
    runtime.pitch_ref = cmd.pitch

    if STAIR_SUPPORT_MARGIN_CONTROL and stair_mode and x_eff > 0.0:
        forward_margin = runtime.support_forward_margin_x
        if np.isfinite(forward_margin) and forward_margin < STAIR_SUPPORT_SLOW_MARGIN_X:
            if STAIR_SUPPORT_SLOW_MARGIN_X > STAIR_SUPPORT_MARGIN_X:
                alpha = (forward_margin - STAIR_SUPPORT_MARGIN_X) / (
                    STAIR_SUPPORT_SLOW_MARGIN_X - STAIR_SUPPORT_MARGIN_X
                )
                alpha = float(np.clip(alpha, 0.0, 1.0))
            else:
                alpha = 0.0
            allowed_x = STAIR_SUPPORT_X_VEL_LIMIT + alpha * (x_eff - STAIR_SUPPORT_X_VEL_LIMIT)
            x_eff = min(x_eff, allowed_x)

        polygon_margin = runtime.support_margin_xy
        if np.isfinite(polygon_margin) and polygon_margin < STAIR_SUPPORT_POLYGON_SLOW_MARGIN:
            if STAIR_SUPPORT_POLYGON_SLOW_MARGIN > STAIR_SUPPORT_POLYGON_MARGIN:
                alpha = (polygon_margin - STAIR_SUPPORT_POLYGON_MARGIN) / (
                    STAIR_SUPPORT_POLYGON_SLOW_MARGIN - STAIR_SUPPORT_POLYGON_MARGIN
                )
                alpha = float(np.clip(alpha, 0.0, 1.0))
            else:
                alpha = 0.0
            allowed_x = STAIR_SUPPORT_X_VEL_LIMIT + alpha * (x_eff - STAIR_SUPPORT_X_VEL_LIMIT)
            x_eff = min(x_eff, allowed_x)

    if (
        STAIR_SUPPORT_MARGIN_CONTROL
        and stair_mode
        and np.isfinite(runtime.support_margin_xy)
        and runtime.support_margin_xy < STAIR_SUPPORT_Y_MARGIN
        and np.isfinite(runtime.support_center_y)
    ):
        y_feedback = STAIR_SUPPORT_Y_KP * (runtime.support_center_y - com_xy[1])
        y_feedback = float(np.clip(y_feedback, -STAIR_SUPPORT_Y_VEL_LIMIT, STAIR_SUPPORT_Y_VEL_LIMIT))
        y_eff = float(np.clip(y_eff + y_feedback, -STAIR_Y_CENTER_VEL_LIMIT, STAIR_Y_CENTER_VEL_LIMIT))
        runtime.support_y_feedback = y_feedback

    runtime.x_eff = x_eff
    return x_eff, y_eff


def compute_centered_y_velocity(vbot: PinVBotModel, cmd: KeyboardCommand) -> float:
    base_y = float(np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)[1])
    y_feedback = -Y_CENTER_KP * base_y
    return float(np.clip(cmd.y_vel + y_feedback, -Y_CENTER_VEL_LIMIT, Y_CENTER_VEL_LIMIT))


def compute_stair_mode_with_hysteresis(
    terrain: StairTerrain,
    base_x: float,
    previous_stair_mode: bool,
) -> bool:
    enter_min = terrain.start_x - STAIR_MODE_APPROACH_X
    enter_max = terrain.end_x() + STAIR_MODE_EXIT_X
    if previous_stair_mode:
        return enter_min - STAIR_MODE_HYSTERESIS_X <= float(base_x) <= enter_max + STAIR_MODE_HYSTERESIS_X
    return enter_min <= float(base_x) <= enter_max


def update_stair_guard(
    runtime: ControlRuntime,
    base_y: float,
    roll: float,
    stair_mode: bool,
    cmd: KeyboardCommand | None = None,
) -> bool:
    if runtime.stair_guard:
        posture_recovered = (
            abs(float(base_y)) <= STAIR_GUARD_Y_RELEASE
            and abs(float(roll)) <= STAIR_GUARD_ROLL_RELEASE
        )
        command_released = True
        if STAIR_GUARD_REQUIRE_IDLE_RELEASE and cmd is not None:
            command_released = command_is_idle(cmd)
        runtime.stair_guard = not (posture_recovered and command_released)
    else:
        runtime.stair_guard = (
            abs(float(base_y)) > STAIR_GUARD_Y
            or abs(float(roll)) > STAIR_GUARD_ROLL
        )
    return runtime.stair_guard


def compute_effective_velocity_command(
    vbot: PinVBotModel,
    terrain: StairTerrain,
    cmd: KeyboardCommand,
    stair_mode: bool,
    stair_guard: bool = False,
):
    base_pos = np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)
    roll, _, _ = vbot.current_config.compute_euler_angle_world()

    x_eff = float(cmd.x_vel)
    if x_eff > 0.0:
        if stair_guard:
            x_eff = min(x_eff, STAIR_GUARD_X_VEL_LIMIT)
        elif float(base_pos[0]) >= terrain.start_x - STAIR_APPROACH_SPEED_ZONE_X:
            if stair_mode:
                x_limit = STAIR_X_VEL_LIMIT
                if abs(float(base_pos[1])) > STAIR_RECOVERY_Y or abs(float(roll)) > STAIR_RECOVERY_ROLL:
                    x_limit = STAIR_RECOVERY_X_VEL_LIMIT
            else:
                x_limit = STAIR_APPROACH_X_VEL_LIMIT
            x_eff = min(x_eff, x_limit)
        elif stair_mode:
            x_limit = STAIR_X_VEL_LIMIT
            if abs(float(base_pos[1])) > STAIR_RECOVERY_Y or abs(float(roll)) > STAIR_RECOVERY_ROLL:
                x_limit = STAIR_RECOVERY_X_VEL_LIMIT
            x_eff = min(x_eff, x_limit)

    if stair_mode and STAIR_SEQUENCE_FOOTHOLDS and x_eff > STAIR_STEP_WAIT_X_VEL_LIMIT:
        foot_steps = {}
        for leg in MPC_LEG_ORDER:
            foot_pos, _ = vbot.get_single_foot_state_in_world(leg)
            foot_steps[leg] = terrain.step_index_at(foot_pos[0], foot_pos[1])
        front_step = max(foot_steps["FL"], foot_steps["FR"])
        rear_step = min(foot_steps["RL"], foot_steps["RR"])
        if front_step - rear_step > STAIR_STEP_WAIT_MAX_LEAD:
            x_eff = min(x_eff, STAIR_STEP_WAIT_X_VEL_LIMIT)

    if stair_mode and x_eff > 0.0 and STAIR_END_SLOW_ZONE_X > 0.0:
        distance_to_end = terrain.end_x() - float(base_pos[0])
        stop_margin = max(0.0, float(STAIR_END_STOP_MARGIN_X))
        if distance_to_end <= stop_margin:
            x_eff = min(x_eff, STAIR_END_X_VEL_LIMIT)
        elif distance_to_end < STAIR_END_SLOW_ZONE_X:
            denom = max(1e-6, STAIR_END_SLOW_ZONE_X - stop_margin)
            alpha = float(np.clip((distance_to_end - stop_margin) / denom, 0.0, 1.0))
            allowed_x = STAIR_END_X_VEL_LIMIT + alpha * (x_eff - STAIR_END_X_VEL_LIMIT)
            x_eff = min(x_eff, allowed_x)

    if stair_guard:
        y_feedback = -STAIR_GUARD_Y_CENTER_KP * float(base_pos[1])
        y_limit = STAIR_GUARD_Y_CENTER_VEL_LIMIT
    elif stair_mode:
        y_feedback = -STAIR_Y_CENTER_KP * float(base_pos[1])
        y_limit = STAIR_Y_CENTER_VEL_LIMIT
    else:
        y_feedback = -Y_CENTER_KP * float(base_pos[1])
        y_limit = Y_CENTER_VEL_LIMIT
    y_eff = float(np.clip(cmd.y_vel + y_feedback, -y_limit, y_limit))
    return x_eff, y_eff


def compute_effective_yaw_rate_command(
    vbot: PinVBotModel,
    cmd: KeyboardCommand,
    stair_mode: bool,
) -> float:
    yaw_eff = float(cmd.yaw_rate)
    if stair_mode and STAIR_YAW_CENTER_KP > 0.0:
        _, _, yaw = vbot.current_config.compute_euler_angle_world()
        yaw_feedback = -STAIR_YAW_CENTER_KP * wrap_to_pi(yaw)
        yaw_eff += float(np.clip(yaw_feedback, -STAIR_YAW_RATE_LIMIT, STAIR_YAW_RATE_LIMIT))
        yaw_eff = float(np.clip(yaw_eff, -YAW_RATE_LIMIT, YAW_RATE_LIMIT))
    return yaw_eff


def compute_joint_bias(vbot: PinVBotModel) -> np.ndarray:
    g, C, _ = vbot.compute_dynamcis_terms()
    return np.asarray(C @ vbot.current_config.get_dq() + g, dtype=float).reshape(-1)


def reset_com_traj_start_to_current(vbot: PinVBotModel, traj: ComTraj):
    if COM_TRAJ_START_AT_CURRENT:
        traj.pos_des_world[0:2] = vbot.compute_com_x_vec().reshape(-1)[0:2]


def start_walking_phase(
    runtime: ControlRuntime,
    time_now_s: float,
    vbot: PinVBotModel,
    leg_controller: LegController,
    traj: ComTraj,
    gait: Gait,
):
    runtime.walking = True
    runtime.gait_start_time_s = time_now_s
    runtime.gait_ctrl_i = 0
    runtime.gait_time_s = 0.0

    # Start crawl from the gait's initial contact state instead of inheriting
    # an arbitrary MuJoCo wall-clock phase.
    leg_controller.last_mask = gait.compute_current_mask(0.0).reshape(4).copy()
    traj.pos_des_world = vbot.compute_com_x_vec().reshape(-1)[0:3].copy()
    traj._last_generate_time = 0.0


def compute_control_tick(
    vbot: PinVBotModel,
    mujoco_vbot: MuJoCo_VBot_Model,
    leg_controller: LegController,
    traj: ComTraj,
    gait: Gait,
    stand_gait: Gait,
    mpc: CentroidalMPC,
    cmd: KeyboardCommand,
    terrain: StairTerrain,
    ctrl_i: int,
    u_opt: np.ndarray,
    runtime: ControlRuntime,
):
    time_now_s = float(mujoco_vbot.data.time)

    mujoco_vbot.update_pin_with_mujoco(vbot)
    update_command_filter(cmd, CTRL_DT)
    base_pos = np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)
    roll, _, _ = vbot.current_config.compute_euler_angle_world()
    prev_stair_mode = runtime.stair_mode
    prev_stair_guard = runtime.stair_guard
    stair_now = compute_stair_mode_with_hysteresis(terrain, base_pos[0], prev_stair_mode)
    stair_guard = update_stair_guard(runtime, base_pos[1], roll, stair_now, cmd)
    mode_changed = (stair_now != prev_stair_mode) or (stair_guard != prev_stair_guard)
    x_eff, y_eff = compute_effective_velocity_command(vbot, terrain, cmd, stair_now, stair_guard)
    yaw_rate_eff = compute_effective_yaw_rate_command(vbot, cmd, stair_now)
    runtime.x_eff = x_eff
    runtime.yaw_rate_eff = yaw_rate_eff
    if command_is_idle(cmd):
        cmd.pitch = ramp_value(cmd.pitch, 0.0, STAIR_PITCH_RAMP, CTRL_DT)
        runtime.pitch_ref = cmd.pitch
        runtime.yaw_rate_eff = 0.0
        runtime.reset()
        tau_cmd = compute_stand_pd_torque(vbot, mujoco_vbot)
        return tau_cmd, u_opt, "stand_pd", np.zeros(12, dtype=float), y_eff, np.zeros(4, dtype=int)

    if stair_guard and STAIR_GUARD_USE_STAND_GAIT and not stair_now:
        runtime.walking = False
        runtime.stair_mode = stair_now
        runtime.x_eff = 0.0
        runtime.leg_outputs.clear()
        tau_cmd = compute_stand_pd_torque(vbot, mujoco_vbot)
        return tau_cmd, u_opt, "stair_guard", np.zeros(12, dtype=float), y_eff, np.ones(4, dtype=int)

    active_gait = stand_gait if stair_guard and STAIR_GUARD_USE_STAND_GAIT else gait
    if stair_guard:
        mode = "stair_guard"
    else:
        mode = "stair_crawl" if stair_now else "crawl"
    if not runtime.walking:
        start_walking_phase(runtime, time_now_s, vbot, leg_controller, traj, active_gait)
    elif RESET_GAIT_ON_STAIR_ENTRY and stair_now and not runtime.stair_mode:
        start_walking_phase(runtime, time_now_s, vbot, leg_controller, traj, active_gait)
        u_opt = np.zeros_like(u_opt)
    elif mode_changed:
        u_opt = np.zeros_like(u_opt)
    runtime.stair_mode = stair_now

    gait_time_s = max(0.0, time_now_s - runtime.gait_start_time_s)
    runtime.gait_time_s = gait_time_s
    contact_mask_for_command = active_gait.compute_current_mask(gait_time_s).reshape(-1)
    if (
        stair_now
        and STAIR_MOVE_ONLY_ALL_STANCE
        and not np.all(contact_mask_for_command == 1)
        and x_eff > STAIR_SWING_X_VEL_LIMIT
    ):
        x_eff = STAIR_SWING_X_VEL_LIMIT
        runtime.x_eff = x_eff

    x_eff, y_eff = update_stair_reference_command(
        vbot,
        terrain,
        cmd,
        runtime,
        contact_mask_for_command,
        stair_now,
        x_eff,
        y_eff,
        CTRL_DT,
    )

    if (runtime.gait_ctrl_i % STEPS_PER_MPC) == 0 or not np.any(u_opt):
        reset_com_traj_start_to_current(vbot, traj)
        z_ref_fn, pitch_ref_fn = make_terrain_mpc_reference_functions(terrain, cmd)
        traj.generate_traj(
            vbot,
            active_gait,
            gait_time_s,
            x_eff,
            y_eff,
            cmd.z_pos,
            yaw_rate_eff,
            time_step=MPC_DT,
            terrain_height_fn=terrain.height,
            pitch_des_body=cmd.pitch if STAIR_POSTURE_REFERENCE else 0.0,
            z_traj_world_fn=z_ref_fn,
            pitch_traj_world_fn=pitch_ref_fn,
        )
        sol = mpc.solve_QP(vbot, traj, False)
        n_horizon = traj.N
        w_opt = sol["x"].full().flatten()
        u_opt = w_opt[12 * n_horizon :].reshape((12, n_horizon), order="F")

    mpc_force_now = u_opt[:, 0]

    tau_raw = np.zeros(12, dtype=float)
    contact_mask = active_gait.compute_current_mask(gait_time_s).reshape(-1)
    joint_bias = compute_joint_bias(vbot) if STANCE_BIAS_COMP else None
    leg_outputs = {}
    for leg, leg_slice in LEG_SLICE.items():
        out = leg_controller.compute_leg_torque(
            leg,
            vbot,
            active_gait,
            mpc_force_now[leg_slice],
            gait_time_s,
        )
        leg_outputs[leg] = out
        tau_leg = np.asarray(out.tau, dtype=float).reshape(3)
        if STANCE_BIAS_COMP and contact_mask[LEG_INDEX[leg]] == 1:
            tau_leg = tau_leg + joint_bias[vbot.get_leg_joint_vcols(leg)]
        tau_raw[leg_slice] = tau_leg

    tau_cmd = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
    runtime.leg_outputs = leg_outputs
    runtime.swing_debug = dict(getattr(active_gait, "last_swing_debug", {}))
    runtime.gait_ctrl_i += 1
    return tau_cmd, u_opt, mode, mpc_force_now, y_eff, contact_mask


def build_initial_mpc(vbot: PinVBotModel, gait: Gait, cmd: KeyboardCommand, terrain: StairTerrain):
    traj = ComTraj(vbot)
    update_body_height_command(vbot, terrain, cmd)
    y_eff = compute_centered_y_velocity(vbot, cmd)
    reset_com_traj_start_to_current(vbot, traj)
    z_ref_fn, pitch_ref_fn = make_terrain_mpc_reference_functions(terrain, cmd)
    traj.generate_traj(
        vbot,
        gait,
        0.0,
        cmd.x_vel,
        y_eff,
        cmd.z_pos,
        cmd.yaw_rate,
        time_step=MPC_DT,
        terrain_height_fn=terrain.height,
        pitch_des_body=cmd.pitch if STAIR_POSTURE_REFERENCE else 0.0,
        z_traj_world_fn=z_ref_fn,
        pitch_traj_world_fn=pitch_ref_fn,
    )
    mpc = CentroidalMPC(vbot, traj)
    u_opt = np.zeros((12, traj.N), dtype=float)
    return traj, mpc, u_opt


def run_headless(duration_s: float, target_x_vel: float | None = None, stop_on_fall: bool = True):
    cmd = KeyboardCommand()
    cmd.target_x_vel = X_VEL_STEP if target_x_vel is None else float(target_x_vel)
    cmd.clamp()
    terrain = StairTerrain()

    vbot = PinVBotModel()
    vbot.terrain_height_fn = terrain.height

    mujoco_vbot = MuJoCo_VBot_Model(xml_path=STAIRS_XML_PATH)
    mujoco_vbot.model.opt.timestep = SIM_DT
    initialize_robot(vbot, mujoco_vbot)

    leg_controller = LegController()
    if USE_STAIR_AWARE_FOOTHOLD:
        gait = StairAwareGait(
            GAIT_HZ,
            GAIT_DUTY,
            terrain,
            swing_height=SWING_HEIGHT,
            phase_offset=CRAWL_PHASE_OFFSET,
        )
    else:
        gait = Gait(
            GAIT_HZ,
            GAIT_DUTY,
            swing_height=SWING_HEIGHT,
            phase_offset=CRAWL_PHASE_OFFSET,
        )
    stand_gait = Gait(
        GAIT_HZ,
        1.0,
        swing_height=0.0,
        phase_offset=np.zeros(4),
    )
    traj, mpc, u_opt = build_initial_mpc(vbot, stand_gait, cmd, terrain)
    logger = RunLogger()
    runtime = ControlRuntime()
    tau_hold = np.zeros(12, dtype=float)
    mpc_force_now = np.zeros(12, dtype=float)
    contact_mask = np.zeros(4, dtype=int)
    y_eff = 0.0
    active_mode = "stand"
    fell_reason = None

    print("VBot stair MPC headless run")
    print(f"Scene: {STAIRS_XML_PATH}")
    print(f"Target x velocity: {cmd.target_x_vel:+.3f} m/s")
    print(f"Log: {logger.path}")

    ctrl_i = 0
    sim_steps = int(float(duration_s) * SIM_HZ)
    try:
        for sim_i in range(sim_steps):
            if (sim_i % CTRL_DECIM) == 0:
                tau_hold, u_opt, active_mode, mpc_force_now, y_eff, contact_mask = compute_control_tick(
                    vbot,
                    mujoco_vbot,
                    leg_controller,
                    traj,
                    gait,
                    stand_gait,
                    mpc,
                    cmd,
                    terrain,
                    ctrl_i,
                    u_opt,
                    runtime,
                )
                if (ctrl_i % LOG_CTRL_DECIM) == 0:
                    logger.log(
                        float(mujoco_vbot.data.time),
                        vbot,
                        cmd,
                        terrain,
                        tau_hold,
                        mpc,
                        active_mode,
                        mpc_force_now,
                        y_eff,
                        runtime,
                        contact_mask,
                    )
                if (ctrl_i % STATUS_CTRL_DECIM) == 0:
                    base_pos = vbot.current_config.base_pos
                    roll, pitch, _ = vbot.current_config.compute_euler_angle_world()
                    tau_abs = np.abs(tau_hold)
                    print(
                        f"[Headless] t={float(mujoco_vbot.data.time):.2f}s "
                        f"x={base_pos[0]:+.3f} y={base_pos[1]:+.3f} z={base_pos[2]:.3f} "
                        f"roll={roll:+.2f} pitch={pitch:+.2f} "
                        f"mode={active_mode} mask={''.join(str(int(v)) for v in contact_mask)} "
                        f"x_eff={runtime.x_eff:+.3f} stair={int(runtime.stair_mode)} "
                        f"guard={int(runtime.stair_guard)} yaw_eff={runtime.yaw_rate_eff:+.3f} "
                        f"pitch_ref={runtime.pitch_ref:+.3f} "
                        f"mx={runtime.support_margin_x:+.3f} mxy={runtime.support_margin_xy:+.3f} "
                        f"tau_max={float(np.max(tau_abs)):.1f}"
                    )

                base_pos = vbot.current_config.base_pos
                roll, pitch, _ = vbot.current_config.compute_euler_angle_world()
                if stop_on_fall:
                    if float(base_pos[2]) < 0.12:
                        fell_reason = f"base_z_low:{float(base_pos[2]):.3f}"
                        break
                    if abs(float(roll)) > 1.20:
                        fell_reason = f"roll_large:{float(roll):.3f}"
                        break
                    if abs(float(pitch)) > 1.20:
                        fell_reason = f"pitch_large:{float(pitch):.3f}"
                        break
                ctrl_i += 1

            mj.mj_step1(mujoco_vbot.model, mujoco_vbot.data)
            mujoco_vbot.set_joint_torque(tau_hold)
            mj.mj_step2(mujoco_vbot.model, mujoco_vbot.data)

        mujoco_vbot.update_pin_with_mujoco(vbot)
        base_pos = vbot.current_config.base_pos
        roll, pitch, _ = vbot.current_config.compute_euler_angle_world()
        print(
            f"[Headless] done t={float(mujoco_vbot.data.time):.2f}s "
            f"x={base_pos[0]:+.3f} y={base_pos[1]:+.3f} z={base_pos[2]:.3f} "
            f"roll={roll:+.3f} pitch={pitch:+.3f} fell={fell_reason}"
        )
        return logger.path, fell_reason
    finally:
        logger.close()
        print(f"Log saved: {logger.path}")


def run_viewer():
    cmd = KeyboardCommand()
    terrain = StairTerrain()

    vbot = PinVBotModel()
    vbot.terrain_height_fn = terrain.height

    mujoco_vbot = MuJoCo_VBot_Model(xml_path=STAIRS_XML_PATH)
    mujoco_vbot.model.opt.timestep = SIM_DT
    initialize_robot(vbot, mujoco_vbot)

    leg_controller = LegController()
    if USE_STAIR_AWARE_FOOTHOLD:
        gait = StairAwareGait(
            GAIT_HZ,
            GAIT_DUTY,
            terrain,
            swing_height=SWING_HEIGHT,
            phase_offset=CRAWL_PHASE_OFFSET,
        )
    else:
        gait = Gait(
            GAIT_HZ,
            GAIT_DUTY,
            swing_height=SWING_HEIGHT,
            phase_offset=CRAWL_PHASE_OFFSET,
        )
    stand_gait = Gait(
        GAIT_HZ,
        1.0,
        swing_height=0.0,
        phase_offset=np.zeros(4),
    )
    traj, mpc, u_opt = build_initial_mpc(vbot, stand_gait, cmd, terrain)
    logger = RunLogger()

    print("VBot stair MPC keyboard control")
    print(f"Scene: {STAIRS_XML_PATH}")
    print(
        "Stairs: "
        f"x_start={terrain.start_x:.2f} m, "
        f"step={terrain.step_depth:.2f} x {terrain.step_height:.2f} m, "
        f"count={terrain.step_count}"
    )
    print(
        "Stair control: "
        f"preview={STAIR_HEIGHT_PREVIEW_X:.2f} m, "
        f"height_gain={STAIR_BODY_HEIGHT_GAIN:.2f}, "
        f"z_ramp={Z_POS_RAMP:.2f} m/s, "
        f"stair_foothold={USE_STAIR_AWARE_FOOTHOLD}, "
        f"terrain_swing={USE_TERRAIN_AWARE_SWING}"
    )
    print("Keys: I/K forward, J/L lateral, U/O yaw, X stop, T reset, Esc exit")
    print(f"Initial command -> {cmd.summary()}")
    print(f"Log: {logger.path}")

    ctrl_i = 0
    sim_i = 0
    runtime = ControlRuntime()
    tau_hold = np.zeros(12, dtype=float)
    mpc_force_now = np.zeros(12, dtype=float)
    contact_mask = np.zeros(4, dtype=int)
    y_eff = 0.0
    active_mode = "stand"
    next_viewer_sync_t = 0.0

    try:
        with mujoco.viewer.launch_passive(
            mujoco_vbot.model,
            mujoco_vbot.data,
            key_callback=make_key_callback(cmd),
        ) as viewer:
            viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco_vbot.base_bid
            viewer.cam.distance = 2.8
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 90
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True

            last_wall_time = time.perf_counter()

            while viewer.is_running() and not cmd.exit_requested:
                loop_start = time.perf_counter()

                if cmd.reset_requested:
                    initialize_robot(vbot, mujoco_vbot)
                    leg_controller = LegController()
                    traj, mpc, u_opt = build_initial_mpc(vbot, stand_gait, cmd, terrain)
                    runtime.reset()
                    tau_hold[:] = 0.0
                    mpc_force_now[:] = 0.0
                    contact_mask[:] = 0
                    y_eff = 0.0
                    active_mode = "stand"
                    ctrl_i = 0
                    sim_i = 0
                    cmd.reset_requested = False

                if (sim_i % CTRL_DECIM) == 0:
                    tau_hold, u_opt, active_mode, mpc_force_now, y_eff, contact_mask = compute_control_tick(
                        vbot,
                        mujoco_vbot,
                        leg_controller,
                        traj,
                        gait,
                        stand_gait,
                        mpc,
                        cmd,
                        terrain,
                        ctrl_i,
                        u_opt,
                        runtime,
                    )
                    if (ctrl_i % LOG_CTRL_DECIM) == 0:
                        logger.log(
                            float(mujoco_vbot.data.time),
                            vbot,
                            cmd,
                            terrain,
                            tau_hold,
                            mpc,
                            active_mode,
                            mpc_force_now,
                            y_eff,
                            runtime,
                            contact_mask,
                        )
                    if (ctrl_i % STATUS_CTRL_DECIM) == 0:
                        base_pos = vbot.current_config.base_pos
                        roll, pitch, _ = vbot.current_config.compute_euler_angle_world()
                        tau_abs = np.abs(tau_hold)
                        sat_frac = float(np.mean(tau_abs >= 0.98 * TAU_LIM))
                        terrain_h = terrain.body_support_height(base_pos)
                        print(
                            f"[Status] t={float(mujoco_vbot.data.time):.2f}s "
                            f"x={base_pos[0]:+.3f} y={base_pos[1]:+.3f} z={base_pos[2]:.3f} "
                            f"roll={roll:+.2f} pitch={pitch:+.2f} "
                            f"mode={active_mode} "
                            f"gait_t={runtime.gait_time_s:.2f} mask={''.join(str(int(v)) for v in contact_mask)} "
                            f"cmd_x={cmd.x_vel:+.3f}/{cmd.target_x_vel:+.3f} x_eff={runtime.x_eff:+.3f} "
                            f"stair={int(runtime.stair_mode)} guard={int(runtime.stair_guard)} "
                            f"y_eff={y_eff:+.3f} terr={terrain_h:.2f} zcmd={cmd.z_pos:.3f} "
                            f"yaw_eff={runtime.yaw_rate_eff:+.3f} "
                            f"pitch_ref={runtime.pitch_ref:+.3f} mx={runtime.support_margin_x:+.3f} "
                            f"mxy={runtime.support_margin_xy:+.3f} "
                            f"tau_max={float(np.max(tau_abs)):.1f} sat={sat_frac:.2f}"
                        )
                    ctrl_i += 1

                mj.mj_step1(mujoco_vbot.model, mujoco_vbot.data)
                mujoco_vbot.set_joint_torque(tau_hold)
                mj.mj_step2(mujoco_vbot.model, mujoco_vbot.data)
                sim_i += 1

                sim_time = float(mujoco_vbot.data.time)
                if sim_time + 1e-12 >= next_viewer_sync_t:
                    viewer.sync()
                    next_viewer_sync_t += VIEWER_DT

                elapsed_wall = loop_start - last_wall_time
                sleep_time = SIM_DT - elapsed_wall
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_wall_time = loop_start
    finally:
        logger.close()
        print(f"Log saved: {logger.path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without MuJoCo viewer")
    parser.add_argument("--duration", type=float, default=20.0, help="headless run duration in seconds")
    parser.add_argument("--target-x-vel", type=float, default=None, help="headless target forward velocity")
    parser.add_argument("--no-stop-on-fall", action="store_true", help="continue headless run after fall thresholds")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.headless:
        run_headless(
            args.duration,
            target_x_vel=args.target_x_vel,
            stop_on_fall=not args.no_stop_on_fall,
        )
    else:
        run_viewer()


if __name__ == "__main__":
    main()
