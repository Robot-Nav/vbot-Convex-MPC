"""
VBot validation 19: keyboard-controlled MPC on a higher stair terrain.

This experiment reuses the EX18 controller and swaps in a 6 cm/step stair
profile. It is intentionally conservative so the taller scene can separate
"the stair is too low" from genuine foot-placement and body-height issues.
"""
from dataclasses import dataclass
from pathlib import Path

import ex18_vbot_mpc_stairs_keyboard_control as ex18


REPO = Path(__file__).resolve().parents[2]
HIGH_STAIRS_XML_PATH = REPO / "models" / "MJCF" / "vbot" / "vbot_mpc_scene_high_stairs.xml"


@dataclass(frozen=True)
class HighStairTerrain(ex18.StairTerrain):
    """Analytic profile matching vbot_mpc_scene_high_stairs.xml."""

    start_x: float = 0.55
    step_depth: float = 0.30
    step_height: float = 0.06
    step_heights: tuple[float, ...] = (0.06, 0.06, 0.06, 0.06, 0.06)
    landing_depth: float = 0.90
    half_width_y: float = 0.36


def configure_high_stairs():
    ex18.STAIRS_XML_PATH = HIGH_STAIRS_XML_PATH
    ex18.LOG_PREFIX = "ex19_high_stairs"
    ex18.StairTerrain = HighStairTerrain

    # Higher stairs need more swing clearance and a slower approach. Use a
    # soft tread projection so Raibert-style nominal footholds are kept, and
    # only unsafe edge/riser positions are clipped into the tread.
    ex18.SWING_HEIGHT = 0.09
    ex18.GAIT_DUTY = 0.84
    ex18.STAIR_HEIGHT_PREVIEW_X = 0.02
    ex18.STAIR_BODY_HEIGHT_GAIN = 0.20
    ex18.STAIR_BODY_FOOT_HEIGHT_GAIN = 0.90
    ex18.STAIR_BODY_USE_FOOT_SUPPORT = True
    ex18.STAIR_BODY_FOOT_SUPPORT_MIN_COUNT = 4
    ex18.STAIR_BODY_FOOT_NEAR_SURFACE_Z = 0.035
    ex18.STAIR_BODY_SUPPORT_FRONT_X = 0.05
    ex18.Z_POS_RAMP = 0.04
    ex18.STAIR_POSTURE_REFERENCE = True
    ex18.STAIR_SUPPORT_MARGIN_CONTROL = True
    ex18.STAIR_PITCH_GAIN = 0.45
    ex18.STAIR_PITCH_LIMIT = 0.12
    ex18.STAIR_PITCH_RAMP = 0.18
    ex18.STAIR_SUPPORT_MARGIN_X = 0.045
    ex18.STAIR_SUPPORT_SLOW_MARGIN_X = 0.090
    ex18.STAIR_SUPPORT_X_VEL_LIMIT = 0.012
    ex18.STAIR_SUPPORT_POLYGON_MARGIN = 0.000
    ex18.STAIR_SUPPORT_POLYGON_SLOW_MARGIN = 0.025
    ex18.STAIR_SUPPORT_Y_MARGIN = 0.030
    ex18.STAIR_SUPPORT_Y_KP = 0.0
    ex18.STAIR_SUPPORT_Y_VEL_LIMIT = 0.0
    ex18.STAIR_YAW_CENTER_KP = 0.35
    ex18.STAIR_YAW_RATE_LIMIT = 0.08
    ex18.STAIR_APPROACH_SPEED_ZONE_X = 0.25
    ex18.STAIR_APPROACH_X_VEL_LIMIT = 0.055
    ex18.STAIR_X_VEL_LIMIT = 0.045
    ex18.STAIR_RECOVERY_X_VEL_LIMIT = 0.015
    ex18.STAIR_MODE_APPROACH_X = 0.25
    ex18.STAIR_MODE_HYSTERESIS_X = 0.10
    ex18.STAIR_FOOTHOLD_MARGIN_X = 0.035
    ex18.STAIR_FOOTHOLD_CENTERING_GAIN = 0.00
    ex18.STAIR_FOOTHOLD_SOFT_PROJECTION = True
    ex18.STAIR_FOOTHOLD_MAX_DELTA_X = 0.18
    ex18.STAIR_REAR_FOOTHOLD_MAX_DELTA_X = 0.34
    ex18.STAIR_FOOTHOLD_Y_LIMIT = 0.22
    ex18.STAIR_SWING_TERRAIN_CLEARANCE = 0.055
    ex18.STAIR_MOVE_ONLY_ALL_STANCE = False
    ex18.STAIR_SWING_X_VEL_LIMIT = 0.0
    ex18.STAIR_Y_CENTER_KP = 1.5
    ex18.STAIR_Y_CENTER_VEL_LIMIT = 0.14
    ex18.STAIR_FORCE_REAR_STEP_UP = False
    ex18.STAIR_REAR_STEP_UP_BASE_MARGIN_X = 0.10
    ex18.STAIR_FORCE_REAR_TO_TREAD_ENTRY = True
    ex18.STAIR_FRONT_STEP_LEAD_MAX = 1
    ex18.STAIR_STEP_WAIT_MAX_LEAD = 1
    ex18.STAIR_STEP_WAIT_X_VEL_LIMIT = 0.020
    ex18.STAIR_END_SLOW_ZONE_X = 0.35
    ex18.STAIR_END_STOP_MARGIN_X = 0.18
    ex18.STAIR_END_X_VEL_LIMIT = 0.0
    ex18.USE_STAIR_AWARE_FOOTHOLD = True
    ex18.USE_TERRAIN_AWARE_SWING = True

    # If the body starts rolling sideways on the stair, pause forward motion
    # and briefly solve the MPC with all four feet in support.
    ex18.STAIR_GUARD_Y = 0.07
    ex18.STAIR_GUARD_Y_RELEASE = 0.035
    ex18.STAIR_GUARD_ROLL = 0.26
    ex18.STAIR_GUARD_ROLL_RELEASE = 0.18
    ex18.STAIR_GUARD_X_VEL_LIMIT = 0.0
    ex18.STAIR_GUARD_Y_CENTER_KP = 0.9
    ex18.STAIR_GUARD_Y_CENTER_VEL_LIMIT = 0.10
    ex18.STAIR_GUARD_USE_STAND_GAIT = False
    ex18.STAIR_GUARD_REQUIRE_IDLE_RELEASE = False

    # Use a quicker flat-ground approach; the stair-mode limiter above still
    # clamps the actual stair crawl speed once the body reaches the first step.
    ex18.X_VEL_LIMIT = 0.06
    ex18.X_VEL_RAMP = 0.12


def main():
    configure_high_stairs()
    ex18.main()


if __name__ == "__main__":
    main()
