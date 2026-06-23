"""
VBot validation 20: active high-stair climb with terrain-aware swing.

This experiment reuses EX18/EX19 and only opens the new active stair-climb
strategy through parameter overrides. The default terrain matches
vbot_mpc_scene_active_high_stairs.xml: five 6 cm risers, 30 cm treads,
and a 90 cm landing.
"""

from dataclasses import dataclass
from pathlib import Path

import ex18_vbot_mpc_stairs_keyboard_control as ex18
import ex19_vbot_mpc_high_stairs_keyboard_control as ex19


REPO = Path(__file__).resolve().parents[2]
ACTIVE_HIGH_STAIRS_XML_PATH = (
    REPO / "models" / "MJCF" / "vbot" / "vbot_mpc_scene_active_high_stairs.xml"
)


@dataclass(frozen=True)
class ActiveHighStairTerrain(ex18.StairTerrain):
    """Analytic profile matching vbot_mpc_scene_active_high_stairs.xml."""

    start_x: float = 0.55
    step_depth: float = 0.30
    step_height: float = 0.06
    step_heights: tuple[float, ...] = (0.06, 0.06, 0.06, 0.06, 0.06)
    landing_depth: float = 0.90
    half_width_y: float = 0.36


def configure_active_high_stairs():
    ex19.configure_high_stairs()

    ex18.LOG_PREFIX = "ex20_active_high_stairs"
    ex18.STAIRS_XML_PATH = ACTIVE_HIGH_STAIRS_XML_PATH
    ex18.StairTerrain = ActiveHighStairTerrain

    # Keep the crawl phase ordering, but give each swing a little more time to
    # clear the riser before the foot settles on the next tread.
    ex18.GAIT_DUTY = 0.84
    ex18.SWING_HEIGHT = 0.10

    # Use the high-stair analytic map for both touchdown z and the swing shape.
    ex18.USE_STAIR_AWARE_FOOTHOLD = True
    ex18.USE_TERRAIN_AWARE_SWING = True
    ex18.USE_ACTIVE_STAIR_SWING = True
    ex18.STAIR_SWING_TERRAIN_CLEARANCE = 0.060
    ex18.STAIR_ACTIVE_SWING_CLEARANCE = 0.055
    ex18.STAIR_ACTIVE_SWING_RISER_CLEARANCE = 0.060
    ex18.STAIR_ACTIVE_SWING_RISER_PHASE_LEAD = 0.08
    ex18.STAIR_ACTIVE_SWING_MIN_PEAK_PHASE = 0.22
    ex18.STAIR_ACTIVE_SWING_MAX_PEAK_PHASE = 0.72
    ex18.STAIR_ACTIVE_SWING_BETA_SHARPNESS = 8.5
    ex18.STAIR_ACTIVE_SWING_MAX_TERRAIN_CLEARANCE = 0.125

    # Preserve the Raibert nominal foothold, then only nudge unsafe edge/riser
    # positions. This allows a two-step front lead without pinning all feet to
    # stair centers.
    ex18.STAIR_SEQUENCE_FOOTHOLDS = True
    ex18.STAIR_FRONT_STEP_LEAD_MAX = 2
    ex18.STAIR_FRONT_STEP_LEAD_MAX_WHILE_REAR_GROUND = 1
    # Available as an EX20-only hook, but disabled here: forcing no front
    # step-down made the first high-riser support too stretched in headless
    # tests.
    ex18.STAIR_PREVENT_FRONT_STEP_DOWN = False
    ex18.STAIR_FRONT_STEP_DOWN_ENTRY_EXTRA_X = 0.010
    ex18.STAIR_STEP_WAIT_MAX_LEAD = 1
    ex18.STAIR_STEP_WAIT_X_VEL_LIMIT = 0.030
    ex18.STAIR_FORCE_REAR_STEP_UP = False
    ex18.STAIR_REAR_STEP_UP_MAX_GAP = 1
    ex18.STAIR_FORCE_REAR_TO_TREAD_ENTRY = True
    ex18.STAIR_REAR_STEP_UP_BASE_MARGIN_X = 0.08
    ex18.STAIR_FOOTHOLD_MARGIN_X = 0.040
    ex18.STAIR_FOOTHOLD_CENTERING_GAIN = 0.0
    ex18.STAIR_FOOTHOLD_SOFT_PROJECTION = True
    ex18.STAIR_FOOTHOLD_MAX_DELTA_X = 0.18
    ex18.STAIR_REAR_FOOTHOLD_MAX_DELTA_X = 0.32
    ex18.STAIR_FOOTHOLD_Y_LIMIT = 0.20
    ex18.STAIR_ACTIVE_FRONT_FOOTHOLD_ADVANCE_X = 0.0
    ex18.STAIR_ACTIVE_FRONT_ADVANCE_WINDOW_X = 0.080
    ex18.STAIR_ACTIVE_FRONT_TREAD_ENTRY_EXTRA_X = 0.020

    # Let the body command move with the stair profile instead of creeping on a
    # flat height until the feet drag it upward. The horizon z/pitch hook is
    # kept available in EX18, but this EX20 setting leaves it off for stability.
    ex18.STAIR_BODY_HEIGHT_GAIN = 0.20
    ex18.STAIR_BODY_FOOT_HEIGHT_GAIN = 0.90
    ex18.STAIR_BODY_USE_FOOT_SUPPORT = True
    ex18.STAIR_BODY_FOOT_SUPPORT_MIN_COUNT = 4
    ex18.STAIR_BODY_SUPPORT_FRONT_X = 0.05
    ex18.STAIR_HEIGHT_PREVIEW_X = 0.02
    ex18.Z_POS_RAMP = 0.040
    ex18.STAIR_TERRAIN_AWARE_MPC_REFERENCE = False
    ex18.STAIR_MPC_TERRAIN_PITCH_REFERENCE = False
    ex18.STAIR_MPC_Z_TRAJ_RAMP = 0.08
    ex18.STAIR_MPC_PITCH_TRAJ_RAMP = 0.20
    ex18.STAIR_MPC_REFERENCE_FRONT_X = 0.20
    ex18.STAIR_MPC_REFERENCE_REAR_X = -0.20
    ex18.STAIR_POSTURE_REFERENCE = True
    ex18.STAIR_PITCH_GAIN = 0.45
    ex18.STAIR_PITCH_LIMIT = 0.12
    ex18.STAIR_PITCH_RAMP = 0.18

    # More active phase-dependent progress: nominally use the requested 0.06
    # m/s, but still slow when the support polygon is thin or the front margin
    # gets small during a single-leg swing.
    ex18.STAIR_SUPPORT_MARGIN_CONTROL = True
    ex18.STAIR_SUPPORT_MARGIN_X = 0.045
    ex18.STAIR_SUPPORT_SLOW_MARGIN_X = 0.090
    ex18.STAIR_SUPPORT_X_VEL_LIMIT = 0.012
    ex18.STAIR_SUPPORT_POLYGON_MARGIN = 0.000
    ex18.STAIR_SUPPORT_POLYGON_SLOW_MARGIN = 0.025
    ex18.STAIR_SUPPORT_Y_MARGIN = 0.028
    ex18.STAIR_SUPPORT_Y_KP = 0.0
    ex18.STAIR_SUPPORT_Y_VEL_LIMIT = 0.0
    ex18.STAIR_APPROACH_SPEED_ZONE_X = 0.25
    ex18.STAIR_APPROACH_X_VEL_LIMIT = 0.065
    ex18.STAIR_X_VEL_LIMIT = 0.045
    ex18.STAIR_RECOVERY_X_VEL_LIMIT = 0.020
    ex18.STAIR_MOVE_ONLY_ALL_STANCE = False
    ex18.STAIR_SWING_X_VEL_LIMIT = 0.0

    # Keep the robot straight without using guard as the primary strategy.
    ex18.STAIR_Y_CENTER_KP = 1.50
    ex18.STAIR_Y_CENTER_VEL_LIMIT = 0.14
    ex18.STAIR_YAW_CENTER_KP = 0.45
    ex18.STAIR_YAW_RATE_LIMIT = 0.08
    ex18.STAIR_GUARD_Y = 0.09
    ex18.STAIR_GUARD_Y_RELEASE = 0.040
    ex18.STAIR_GUARD_ROLL = 0.30
    ex18.STAIR_GUARD_ROLL_RELEASE = 0.18
    ex18.STAIR_GUARD_USE_STAND_GAIT = False
    ex18.STAIR_GUARD_REQUIRE_IDLE_RELEASE = False

    # Slow only near the far end of the landing, after the robot has already
    # climbed the staircase.
    ex18.STAIR_END_SLOW_ZONE_X = 0.65
    ex18.STAIR_END_STOP_MARGIN_X = 0.35
    ex18.STAIR_END_X_VEL_LIMIT = 0.0

    ex18.X_VEL_LIMIT = 0.07
    ex18.X_VEL_RAMP = 0.12


def main():
    configure_active_high_stairs()
    ex18.main()


if __name__ == "__main__":
    main()
