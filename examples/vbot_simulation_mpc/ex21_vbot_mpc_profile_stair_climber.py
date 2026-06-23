"""
VBot validation 21: profile-driven stair climbing MPC.

EX21 keeps the EX18/EX19/EX20 experiments untouched and installs a more
general stair-climb stack at runtime:

  StairProfile + StairPhasePolicy + FootholdPlanner + SwingPlanner
  + BodyReferencePlanner

The profile is analytic and must match the selected MJCF scene exactly.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

import ex18_vbot_mpc_stairs_keyboard_control as ex18
import ex20_vbot_mpc_active_stair_climb_keyboard_control as ex20
from convex_mpc.gait import Gait, FOOT_TOUCHDOWN_CLEARANCE
from convex_mpc.mujoco_vbot_model import PinVBotModel


REPO = Path(__file__).resolve().parents[2]
PROFILE_XML = {
    "uniform06": REPO / "models" / "MJCF" / "vbot" / "vbot_mpc_scene_profile_stairs_06.xml",
    "variable": REPO / "models" / "MJCF" / "vbot" / "vbot_mpc_scene_profile_stairs_variable.xml",
}
PROFILE_STEP_HEIGHTS = {
    "uniform06": (0.06, 0.06, 0.06, 0.06, 0.06),
    "variable": (0.05, 0.065, 0.08, 0.065, 0.05),
}
JOINT_SHORT_NAMES = ("hip", "thigh", "calf")
PROFILE_DEBUG_JOINTS = True
PROFILE_TOUCHDOWN_SETTLE_S = 0.18
PROFILE_TOUCHDOWN_MIN_FORCE_BLEND = 0.35
PROFILE_FORCE_RATE_LIMIT = np.array([420.0, 360.0, 1250.0] * 4, dtype=float)
PROFILE_TAU_RATE_LIMIT = np.array([1250.0, 1650.0, 2450.0] * 4, dtype=float)
PROFILE_TOUCHDOWN_X_LIMIT = 0.032
PROFILE_TOUCHDOWN_YAW_SCALE = 0.45
PROFILE_TOUCHDOWN_Y_SCALE = 0.55
PROFILE_BODY_Z_RAMP = 0.026
PROFILE_BODY_PITCH_RAMP = 0.085
PROFILE_MAX_TOUCHDOWN_VZ = -0.32
PROFILE_START_PHASE_HOLD_S = 0.20
PROFILE_START_SOFT_WINDOW_S = 3.00
PROFILE_START_X_LIMIT_MIN = 0.018
PROFILE_START_X_LIMIT_MAX = 0.052
PROFILE_ENTRY_DAMPING_BEFORE_X = 0.48
PROFILE_ENTRY_DAMPING_AFTER_X = 0.18
PROFILE_ENTRY_Y_KP = 0.55
PROFILE_ENTRY_Y_LIMIT = 0.070
PROFILE_ENTRY_ROLL_SLOW = 0.15
PROFILE_ENTRY_Y_SLOW = 0.025
PROFILE_APPROACH_X_LIMIT = 0.055
PROFILE_APPROACH_RISK_X_LIMIT = 0.026
PROFILE_GUARD_RECOVERY_X_MIN = 0.004
PROFILE_GUARD_RECOVERY_X_MAX = 0.012
PROFILE_GUARD_RECOVERY_Y_KP = 1.35
PROFILE_GUARD_RECOVERY_Y_LIMIT = 0.075
PROFILE_GUARD_YAW_SCALE = 0.28
PROFILE_GUARD_STANCE_ROLL = 0.32
PROFILE_GUARD_STANCE_Y = 0.100
PROFILE_GUARD_STANCE_SUPPORT_MARGIN = -0.020
PROFILE_X_EFF_RAMP = 0.12
PROFILE_Y_EFF_RAMP = 0.22
PROFILE_YAW_EFF_RAMP = 0.16
PROFILE_SAFE_TREAD_MARGIN_X = 0.035
PROFILE_SAFE_TREAD_MARGIN_Y = 0.045
PROFILE_TOUCHDOWN_SAFE_MARGIN_X = 0.006
PROFILE_EDGE_NEAR_X = 0.010
PROFILE_TOUCHDOWN_CONFIRM_S = 0.055
PROFILE_TOUCHDOWN_CONFIRM_CLEARANCE = 0.026
PROFILE_TOUCHDOWN_CONFIRM_Z = 0.040
PROFILE_TOUCHDOWN_CONFIRM_VZ = -0.28
PROFILE_FSM_SETTLE_S = 0.12
PROFILE_FSM_SUPPORT_MARGIN_MIN = 0.000
PROFILE_FSM_HOLD_X_LIMIT = 0.012
PROFILE_FSM_REPLAN_X_LIMIT = 0.008
PROFILE_REAR_APPROACH_CATCHUP_X = 0.125
PROFILE_REAR_STEPUP_DIRECT_X = 0.225
PROFILE_REAR_STEPUP_ENTRY_X = 0.050
PROFILE_REAR_BODY_CATCHUP_MAX_DELTA_X = 0.080
PROFILE_REAR_FORCED_SWING_TIME = 0.52
PROFILE_REAR_FORCED_SWING_SETTLE_S = 0.18
PROFILE_REAR_FORCED_RETRY_S = 0.10
PROFILE_REAR_FORCED_MAX_DELTA_X = 0.30
PROFILE_REAR_FORCED_MIN_SUPPORT_MARGIN = 0.015
PROFILE_REAR_BODY_CATCHUP_MIN_SUPPORT_MARGIN = 0.045
PROFILE_REAR_FORCED_MAX_SUPPORT_CENTER_Y = 0.080
PROFILE_CANDIDATE_X_OFFSETS = (-0.075, -0.045, -0.020, 0.0, 0.020, 0.045, 0.075)
PROFILE_CANDIDATE_Y_OFFSETS = (-0.035, 0.0, 0.035)


@dataclass(frozen=True)
class ProjectionResult:
    pos: np.ndarray
    projected: bool
    reason: str
    target_step: int


@dataclass(frozen=True)
class StairProfile(ex18.StairTerrain):
    """Analytic stair profile matching the EX21 MJCF box geometry."""

    start_x: float = 0.55
    step_depth: float = 0.30
    step_height: float = 0.06
    step_heights: tuple[float, ...] = (0.06, 0.06, 0.06, 0.06, 0.06)
    landing_depth: float = 0.90
    half_width_y: float = 0.36

    def cumulative_height(self, step_index: int) -> float:
        step_index = int(np.clip(step_index, 0, self.step_count))
        return float(np.sum(self.step_heights[:step_index]))

    def _stairs_end_x(self) -> float:
        return self.start_x + self.step_count * self.step_depth

    def end_x(self) -> float:
        return self._stairs_end_x() + self.landing_depth

    def height(self, x: float, y: float) -> float:
        return super().height(x, y)

    def step_index_at(self, x: float, y: float) -> int:
        return super().step_index_at(x, y)

    def tread_interval(self, step_index: int) -> tuple[float, float]:
        """Return the x interval for a support surface.

        step_index=0 is the approach ground; 1..N are treads. The landing has
        the same height as step N, so callers that already know x is on the
        landing should use step_index=N+1.
        """
        i = int(step_index)
        if i <= 0:
            return -np.inf, self.start_x
        if 1 <= i <= self.step_count:
            return self.start_x + (i - 1) * self.step_depth, self.start_x + i * self.step_depth
        return self._stairs_end_x(), self.end_x()

    def safe_tread_interval(self, step_index: int, margin: float) -> tuple[float, float]:
        lo, hi = self.tread_interval(step_index)
        if not np.isfinite(lo):
            lo = self.start_x - 2.0 * self.step_depth
        width = hi - lo
        m = min(max(0.0, float(margin)), max(0.0, 0.45 * width))
        return lo + m, hi - m

    def max_height_between(self, p0, p1) -> float:
        p0 = np.asarray(p0, dtype=float).reshape(3)
        p1 = np.asarray(p1, dtype=float).reshape(3)
        peak = 0.0
        for s in np.linspace(0.0, 1.0, 17):
            xy = p0[:2] + (p1[:2] - p0[:2]) * s
            peak = max(peak, self.height(xy[0], xy[1]))
        return float(peak)

    def _step_for_projection_x(self, x: float, y: float, leg: str | None) -> int:
        if abs(float(y)) > self.half_width_y:
            return 0
        x = float(x)
        margin = min(0.04, 0.45 * self.step_depth)
        if x < self.start_x - margin:
            return 0
        if x < self.start_x:
            return 1
        if x < self._stairs_end_x():
            step = int(np.clip(np.floor((x - self.start_x) / self.step_depth), 0, self.step_count - 1)) + 1
            riser_x = self.start_x + step * self.step_depth
            if leg in ("FL", "FR") and step < self.step_count and 0.0 <= riser_x - x <= 0.06:
                return step + 1
            return step
        if x <= self.end_x():
            return self.step_count + 1
        return 0

    def project_to_safe_tread_result(
        self,
        pos,
        leg: str | None = None,
        nominal_pos=None,
        max_delta_x: float = 0.18,
        soft: bool = True,
        margin: float | None = None,
    ) -> ProjectionResult:
        pos = np.asarray(pos, dtype=float).reshape(3).copy()
        nominal = pos if nominal_pos is None else np.asarray(nominal_pos, dtype=float).reshape(3)
        reason = "none"
        target_step = self._step_for_projection_x(pos[0], pos[1], leg)

        if abs(float(pos[1])) > self.half_width_y:
            pos[1] = float(np.clip(pos[1], -0.20, 0.20))
            reason = "y_outside"
            target_step = max(1, target_step)

        if target_step <= 0:
            pos[2] = self.height(pos[0], pos[1]) + FOOT_TOUCHDOWN_CLEARANCE
            return ProjectionResult(pos, False, reason, target_step)

        m = min(0.040 if margin is None else float(margin), 0.45 * self.step_depth)
        lo, hi = self.safe_tread_interval(target_step, m)
        x_before = float(pos[0])
        if x_before < lo:
            pos[0] = lo
            reason = "before_safe_tread" if reason == "none" else reason
        elif x_before > hi:
            pos[0] = hi
            reason = "after_safe_tread" if reason == "none" else reason

        # Explicitly protect risers and edges even when the interval selection
        # did not move the nominal point very far.
        if 1 <= target_step <= self.step_count:
            tread_lo, tread_hi = self.tread_interval(target_step)
            edge_dist = min(abs(pos[0] - tread_lo), abs(tread_hi - pos[0]))
            if edge_dist < m - 1e-6:
                pos[0] = float(np.clip(pos[0], tread_lo + m, tread_hi - m))
                reason = "edge_or_riser" if reason == "none" else reason

        if soft and nominal_pos is not None and np.isfinite(max_delta_x):
            dx = float(pos[0] - nominal[0])
            if abs(dx) > max_delta_x:
                pos[0] = float(nominal[0] + np.sign(dx) * max_delta_x)
                reason = "soft_delta_limit" if reason == "none" else reason
                # A soft limit is useful for ordinary Raibert nudges, but it
                # must not leave a stair target on the wrong side of a riser.
                if 1 <= target_step <= self.step_count:
                    pos[0] = float(np.clip(pos[0], lo, hi))
                elif target_step > self.step_count:
                    pos[0] = max(pos[0], lo)

        if self.in_stair_mode_x(pos[0]):
            pos[1] = float(np.clip(pos[1], -0.20, 0.20))
        pos[2] = self.height(pos[0], pos[1]) + FOOT_TOUCHDOWN_CLEARANCE
        projected = bool(np.linalg.norm(pos[:2] - nominal[:2]) > 1e-5)
        return ProjectionResult(pos, projected, reason if projected else "none", target_step)

    def project_to_safe_tread(
        self,
        pos,
        leg: str | None = None,
        nominal_pos=None,
        max_delta_x: float = 0.18,
        soft: bool = True,
    ) -> np.ndarray:
        return self.project_to_safe_tread_result(pos, leg, nominal_pos, max_delta_x, soft).pos


@dataclass(frozen=True)
class Uniform06StairProfile(StairProfile):
    """Analytic profile matching vbot_mpc_scene_profile_stairs_06.xml."""


@dataclass(frozen=True)
class VariableStairProfile(StairProfile):
    step_height: float = 0.062
    step_heights: tuple[float, ...] = PROFILE_STEP_HEIGHTS["variable"]


@dataclass(frozen=True)
class FootholdCandidate:
    pos: np.ndarray
    step_index: int
    edge_distance: float
    reason: str


@dataclass
class LocalStairMap:
    profile: StairProfile
    margin_x: float = PROFILE_SAFE_TREAD_MARGIN_X
    margin_y: float = PROFILE_SAFE_TREAD_MARGIN_Y

    def height_at(self, x: float, y: float) -> float:
        return float(self.profile.height(float(x), float(y)))

    def step_index_at(self, x: float, y: float) -> int:
        if abs(float(y)) > self.profile.half_width_y:
            return 0
        x = float(x)
        stairs_end = self.profile._stairs_end_x()
        if stairs_end <= x <= self.profile.end_x():
            return self.profile.step_count + 1
        return int(self.profile.step_index_at(x, y))

    def step_height(self, step_index: int) -> float:
        step = int(step_index)
        if step > self.profile.step_count:
            step = self.profile.step_count
        return float(self.profile.cumulative_height(max(0, step)))

    def safe_tread_interval(self, step_index: int, margin: float | None = None) -> tuple[float, float]:
        return self.profile.safe_tread_interval(
            int(step_index),
            self.margin_x if margin is None else float(margin),
        )

    def edge_distance(self, x: float, y: float, step_index: int | None = None) -> float:
        step = self.step_index_at(x, y) if step_index is None else int(step_index)
        if step <= 0:
            if float(x) < self.profile.start_x:
                x_dist = max(0.0, self.profile.start_x - float(x))
            else:
                x_dist = -1.0
        else:
            lo, hi = self.safe_tread_interval(step, 0.0)
            x_dist = min(float(x) - lo, hi - float(x))
        y_dist = self.profile.half_width_y - abs(float(y))
        return float(min(x_dist, y_dist))

    def is_safe_foothold(self, pos, step_index: int | None = None, margin: float | None = None) -> bool:
        pos = np.asarray(pos, dtype=float).reshape(3)
        step = self.step_index_at(pos[0], pos[1]) if step_index is None else int(step_index)
        if step <= 0:
            return bool(
                pos[0] < self.profile.start_x - 0.5 * self.margin_x
                and abs(float(pos[1])) <= self.profile.half_width_y - self.margin_y
            )
        lo, hi = self.safe_tread_interval(step, self.margin_x if margin is None else float(margin))
        return bool(
            lo <= float(pos[0]) <= hi
            and abs(float(pos[1])) <= self.profile.half_width_y - self.margin_y
        )

    def candidate_footholds(self, nominal, target_step: int, leg: str | None = None) -> list[FootholdCandidate]:
        nominal = np.asarray(nominal, dtype=float).reshape(3)
        target = int(target_step)
        candidates: list[FootholdCandidate] = []
        y_base = float(np.clip(nominal[1], -ex18.STAIR_FOOTHOLD_Y_LIMIT, ex18.STAIR_FOOTHOLD_Y_LIMIT))

        if target <= 0:
            for dx in PROFILE_CANDIDATE_X_OFFSETS:
                p = nominal.copy()
                p[0] = min(float(nominal[0] + dx), self.profile.start_x - self.margin_x)
                p[1] = float(np.clip(y_base, -0.20, 0.20))
                p[2] = self.height_at(p[0], p[1]) + FOOT_TOUCHDOWN_CLEARANCE
                candidates.append(FootholdCandidate(p, 0, self.edge_distance(p[0], p[1], 0), "approach"))
            return candidates

        lo, hi = self.safe_tread_interval(target)
        center_x = 0.5 * (lo + hi)
        x_seeds = [float(np.clip(nominal[0] + dx, lo, hi)) for dx in PROFILE_CANDIDATE_X_OFFSETS]
        x_seeds.extend([center_x, lo, hi])
        if target == 1 and leg in ("FL", "FR"):
            x_seeds.append(float(np.clip(lo + 0.030, lo, hi)))
        if target > 0 and leg in ("RL", "RR"):
            x_seeds.extend(
                [
                    float(np.clip(lo + 0.060, lo, hi)),
                    float(np.clip(lo + 0.090, lo, hi)),
                ]
            )
        for x in sorted(set(round(float(v), 5) for v in x_seeds)):
            for dy in PROFILE_CANDIDATE_Y_OFFSETS:
                p = nominal.copy()
                p[0] = float(np.clip(x, lo, hi))
                p[1] = float(np.clip(y_base + dy, -0.20, 0.20))
                p[2] = self.step_height(target) + FOOT_TOUCHDOWN_CLEARANCE
                step = self.step_index_at(p[0], p[1])
                candidates.append(
                    FootholdCandidate(
                        p,
                        step,
                        self.edge_distance(p[0], p[1], target),
                        "safe_tread_candidate",
                    )
                )
        return candidates


@dataclass
class LegTerrainState:
    observed_step_index: int = 0
    confirmed_step: int = 0
    rear_prepare_step: int = 0
    foot_x: float = float("nan")
    foot_y: float = float("nan")
    foot_z: float = float("nan")
    contact_confidence: float = 0.0
    on_safe_tread: bool = False
    near_edge: bool = False
    edge_distance: float = float("nan")
    touchdown_age: float = float("nan")
    last_confirmed_touchdown: float = float("nan")
    touchdown_confirmed: bool = False
    stable_age: float = 0.0
    terrain_clearance: float = float("nan")


@dataclass
class LegTerrainStateEstimator:
    stair_map: LocalStairMap
    states: dict[str, LegTerrainState] = field(
        default_factory=lambda: {leg: LegTerrainState() for leg in ex18.MPC_LEG_ORDER}
    )

    def update(self, vbot: PinVBotModel, contact_mask, runtime, dt: float, time_s: float = 0.0):
        mask = np.asarray(contact_mask, dtype=int).reshape(4)
        for leg_i, leg in enumerate(ex18.MPC_LEG_ORDER):
            state = self.states[leg]
            foot_pos, foot_vel = vbot.get_single_foot_state_in_world(leg)
            foot_pos = np.asarray(foot_pos, dtype=float).reshape(3)
            foot_vel = np.asarray(foot_vel, dtype=float).reshape(3)
            observed_step = self.stair_map.step_index_at(foot_pos[0], foot_pos[1])
            terrain_h = self.stair_map.height_at(foot_pos[0], foot_pos[1])
            terrain_clearance = float(foot_pos[2] - FOOT_TOUCHDOWN_CLEARANCE - terrain_h)
            edge_dist = self.stair_map.edge_distance(foot_pos[0], foot_pos[1], observed_step)
            on_safe = self.stair_map.is_safe_foothold(foot_pos, observed_step, margin=PROFILE_TOUCHDOWN_SAFE_MARGIN_X)
            height_error = abs(float(foot_pos[2] - FOOT_TOUCHDOWN_CLEARANCE) - self.stair_map.step_height(observed_step))
            stance_like = int(mask[leg_i]) == 1
            close_to_surface = (
                abs(terrain_clearance) <= PROFILE_TOUCHDOWN_CONFIRM_CLEARANCE
                and height_error <= PROFILE_TOUCHDOWN_CONFIRM_Z
            )
            vz_ok = float(foot_vel[2]) >= PROFILE_TOUCHDOWN_CONFIRM_VZ
            stable_now = bool(stance_like and close_to_surface and vz_ok and on_safe)
            if stable_now:
                state.stable_age = min(PROFILE_TOUCHDOWN_CONFIRM_S, state.stable_age + float(dt))
            else:
                state.stable_age = 0.0
            confirmed_now = state.stable_age >= PROFILE_TOUCHDOWN_CONFIRM_S - 1e-9
            if confirmed_now:
                state.confirmed_step = max(int(state.confirmed_step), int(observed_step))
                state.last_confirmed_touchdown = float(time_s)
            if leg in ("RL", "RR"):
                prepare_step = int(state.rear_prepare_step)
                if confirmed_now and observed_step <= 0:
                    max_step = int(self.stair_map.profile.step_count)
                    for step in range(max(1, prepare_step + 1), max_step + 1):
                        safe_lo, _ = self.stair_map.safe_tread_interval(step)
                        prep_x = float(safe_lo) - PROFILE_REAR_STEPUP_DIRECT_X - 0.010
                        if float(foot_pos[0]) >= prep_x:
                            prepare_step = step
                if int(observed_step) > 0 and confirmed_now:
                    prepare_step = max(prepare_step, int(observed_step))
                state.rear_prepare_step = max(int(state.confirmed_step), int(prepare_step))

            td_age = getattr(runtime, "profile_touchdown_age", {}).get(leg, np.nan) if runtime is not None else np.nan
            state.observed_step_index = int(observed_step)
            state.foot_x = float(foot_pos[0])
            state.foot_y = float(foot_pos[1])
            state.foot_z = float(foot_pos[2])
            state.contact_confidence = float(np.clip(state.stable_age / max(1e-6, PROFILE_TOUCHDOWN_CONFIRM_S), 0.0, 1.0))
            state.on_safe_tread = bool(on_safe)
            state.near_edge = bool(edge_dist < PROFILE_EDGE_NEAR_X)
            state.edge_distance = float(edge_dist)
            state.touchdown_age = float(td_age) if np.isfinite(td_age) else np.nan
            state.touchdown_confirmed = bool(confirmed_now)
            state.terrain_clearance = float(terrain_clearance)
        if runtime is not None:
            runtime.leg_terrain_states = self.states
        return self.states


@dataclass
class EventStairFSM:
    terrain: StairProfile
    stair_map: LocalStairMap
    state: str = "approach"
    target_step: int = 1
    hold_reason: str = "approach"
    last_transition_time_s: float = 0.0

    def _front_confirmed(self, states, step: int) -> bool:
        return all(states[leg].confirmed_step >= step for leg in ("FL", "FR"))

    def _rear_confirmed(self, states, step: int) -> bool:
        return all(states[leg].confirmed_step >= step for leg in ("RL", "RR"))

    def _all_confirmed(self, states, step: int) -> bool:
        return all(states[leg].confirmed_step >= step for leg in ex18.MPC_LEG_ORDER)

    def _rear_stepup_reachable(self, states, step: int) -> bool:
        for leg in ("RL", "RR"):
            state = states[leg]
            if int(state.confirmed_step) >= int(step) or int(getattr(state, "rear_prepare_step", 0)) >= int(step):
                continue
            return False
        return True

    def update(self, states, runtime, base_pos, time_s: float = 0.0):
        support_margin = float(getattr(runtime, "support_margin_xy", np.nan))
        settle_ok = all(
            np.isfinite(states[leg].touchdown_age) and states[leg].touchdown_age >= PROFILE_FSM_SETTLE_S
            for leg in ("FL", "FR")
        )
        support_ok = np.isfinite(support_margin) and support_margin >= PROFILE_FSM_SUPPORT_MARGIN_MIN
        rear_settle_ok = all(
            np.isfinite(states[leg].touchdown_age) and states[leg].touchdown_age >= PROFILE_FSM_SETTLE_S
            for leg in ("RL", "RR")
        )
        centered_ok = abs(float(base_pos[1])) <= 0.055
        unsafe_legs = [
            leg
            for leg, state in states.items()
            if state.observed_step_index > 0
            and np.isfinite(state.touchdown_age)
            and state.touchdown_age >= 0.030
            and (not state.on_safe_tread or state.near_edge)
        ]
        max_step = int(self.terrain.step_count)
        prev = self.state
        reason = "waiting"

        if self._all_confirmed(states, max_step):
            self.state = "landing_confirmed"
            self.target_step = max_step
            reason = "all_legs_confirmed_last_step"
        elif self.state == "approach":
            self.target_step = max(1, min(self.target_step, max_step))
            if self._front_confirmed(states, self.target_step):
                self.state = f"front_confirmed_step_{self.target_step}"
                reason = "front_touchdown_confirmed"
            elif bool(getattr(runtime, "stair_mode", False)) or max(
                states["FL"].observed_step_index,
                states["FR"].observed_step_index,
            ) > 0 or float(base_pos[0]) >= float(self.terrain.start_x) - 0.35:
                self.state = f"front_swing_to_step_{self.target_step}"
                reason = "front_target_reachable"
            else:
                reason = "approach_nominal"
        elif self.state.startswith("front_swing_to_step_"):
            if unsafe_legs:
                reason = "unsafe_foothold:" + ",".join(unsafe_legs)
            elif self._front_confirmed(states, self.target_step):
                self.state = f"front_confirmed_step_{self.target_step}"
                reason = "front_touchdown_confirmed"
            else:
                reason = "waiting_front_confirmed"
        elif self.state.startswith("front_confirmed_step_"):
            if not settle_ok:
                reason = "front_settle_window"
            elif not support_ok:
                reason = "support_margin_not_positive"
            else:
                self.state = f"body_commit_step_{self.target_step}"
                reason = "body_commit_guard_ok"
        elif self.state.startswith("body_commit_step_"):
            if unsafe_legs:
                reason = "unsafe_foothold:" + ",".join(unsafe_legs)
            elif self._rear_confirmed(states, self.target_step):
                self.state = f"rear_confirmed_step_{self.target_step}"
                reason = "rear_touchdown_confirmed"
            elif not settle_ok:
                reason = "front_settle_window"
            elif not support_ok:
                reason = "support_margin_not_positive"
            elif (
                str(getattr(runtime, "forced_rear_phase", "")) in ("swing", "settle", "retry_wait")
                and not self._rear_stepup_reachable(states, self.target_step)
            ):
                reason = "rear_prepare_cycle_settle"
            elif self._rear_stepup_reachable(states, self.target_step):
                self.state = f"rear_swing_to_step_{self.target_step}"
                reason = "rear_target_reachable"
            else:
                reason = "rear_stepup_not_reachable"
        elif self.state.startswith("rear_swing_to_step_"):
            if unsafe_legs:
                reason = "unsafe_foothold:" + ",".join(unsafe_legs)
            elif self._rear_confirmed(states, self.target_step):
                self.state = f"rear_confirmed_step_{self.target_step}"
                reason = "rear_touchdown_confirmed"
            else:
                reason = "waiting_rear_confirmed"
        elif self.state.startswith("rear_confirmed_step_"):
            if self.target_step >= max_step:
                if self._all_confirmed(states, max_step):
                    self.state = "landing_confirmed"
                    reason = "all_legs_confirmed_last_step"
                else:
                    reason = "waiting_all_legs_last_step"
            elif not rear_settle_ok:
                reason = "rear_settle_window"
            elif not support_ok:
                reason = "support_margin_not_positive"
            elif not centered_ok:
                reason = "centerline_recovery"
            else:
                self.target_step += 1
                self.state = f"front_swing_to_step_{self.target_step}"
                reason = "advance_next_step"
        elif self.state == "landing_confirmed":
            reason = "landing_confirmed"
        else:
            self.state = "approach"
            reason = "fsm_reset_unknown_state"

        if self.state != prev:
            self.last_transition_time_s = float(time_s)
        self.hold_reason = reason
        if runtime is not None:
            runtime.fsm_state = self.state
            runtime.fsm_target_step = int(self.target_step)
            runtime.fsm_hold_reason = reason
        return self.state, reason

    def target_for_leg(self, leg: str, states) -> tuple[int, str]:
        max_step = int(self.terrain.step_count)
        current = int(states.get(leg, LegTerrainState()).confirmed_step)
        k = int(np.clip(self.target_step, 1, max_step))
        if self.state == "landing_confirmed":
            return max(current, max_step), "fsm_landing_confirmed"
        if leg in ("FL", "FR"):
            if self.state.startswith(("front_swing_to_step_", "approach")):
                return k, "fsm_front_swing"
            return max(current, k), "fsm_front_hold"
        if self.state.startswith(("body_commit_step_", "rear_swing_to_step_")):
            return k, "fsm_rear_catch_up"
        return min(max(current, 0), k), "fsm_rear_hold"


@dataclass
class PhaseCommand:
    phase_name: str = "approach"
    x_vel_limit: float = 0.065
    rear_step_priority: float = 0.0
    front_step_hint: int = 0
    yaw_gain_scale: float = 1.0
    y_gain_scale: float = 1.0
    z_bias: float = 0.0
    pitch_bias: float = 0.0
    fsm_state: str = "approach"
    fsm_target_step: int = 1
    fsm_hold_reason: str = "approach"


@dataclass
class StairPhasePolicy:
    terrain: StairProfile
    last_command: PhaseCommand = field(default_factory=PhaseCommand)
    stair_map: LocalStairMap = field(init=False)
    estimator: LegTerrainStateEstimator = field(init=False)
    fsm: EventStairFSM = field(init=False)

    def __post_init__(self):
        self.stair_map = LocalStairMap(self.terrain)
        self.estimator = LegTerrainStateEstimator(self.stair_map)
        self.fsm = EventStairFSM(self.terrain, self.stair_map)
        self.forced_rear_leg = None
        self.forced_rear_active = False
        self.forced_rear_phase = "idle"

    def update(self, vbot: PinVBotModel, contact_mask, runtime) -> PhaseCommand:
        base = np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)
        time_s = float(getattr(runtime, "gait_time_s", 0.0))
        states = self.estimator.update(vbot, contact_mask, runtime, ex18.CTRL_DT, time_s)
        self.fsm.update(states, runtime, base, time_s)
        foot_steps = {leg: int(state.confirmed_step) for leg, state in states.items()}
        front_min = min(foot_steps["FL"], foot_steps["FR"])
        rear_min = min(foot_steps["RL"], foot_steps["RR"])
        support_margin = float(getattr(runtime, "support_forward_margin_x", np.nan))
        mask = np.asarray(contact_mask, dtype=int).reshape(-1)
        single_swing = np.sum(mask == 0) == 1

        fsm_state = self.fsm.state
        if fsm_state == "landing_confirmed":
            phase = "landing"
        elif fsm_state.startswith("front_swing_to_step_"):
            phase = "front_step_up"
        elif fsm_state.startswith("front_confirmed_step_"):
            phase = "front_stable"
        elif fsm_state.startswith("rear_swing_to_step_"):
            phase = "rear_catch_up"
        elif fsm_state.startswith("body_commit_step_"):
            phase = "body_commit"
        elif fsm_state.startswith("rear_confirmed_step_"):
            phase = "front_stable"
        else:
            phase = "approach"

        cmd = PhaseCommand(
            phase_name=phase,
            fsm_state=fsm_state,
            fsm_target_step=int(self.fsm.target_step),
            fsm_hold_reason=str(self.fsm.hold_reason),
        )
        if phase == "approach":
            cmd.x_vel_limit = 0.065
        elif phase == "front_step_up":
            cmd.x_vel_limit = 0.045
            cmd.front_step_hint = min(self.terrain.step_count, max(1, int(self.fsm.target_step)))
            cmd.pitch_bias = 0.010
        elif phase == "front_stable":
            cmd.x_vel_limit = 0.045
            cmd.z_bias = 0.004
        elif phase == "rear_catch_up":
            cmd.x_vel_limit = 0.045
            cmd.rear_step_priority = 1.0
            cmd.z_bias = 0.008
            cmd.yaw_gain_scale = 0.35
            cmd.y_gain_scale = 0.55
        elif phase == "body_commit":
            cmd.x_vel_limit = 0.045
            cmd.rear_step_priority = 0.4
        elif phase == "landing":
            cmd.x_vel_limit = 0.055
            cmd.yaw_gain_scale = 0.7
            cmd.y_gain_scale = 0.7

        if single_swing or (np.isfinite(support_margin) and support_margin < 0.095):
            cmd.yaw_gain_scale *= 0.45
            cmd.y_gain_scale *= 0.55
            cmd.x_vel_limit = min(cmd.x_vel_limit, 0.032)
        rear_confirmed_count = sum(states[leg].confirmed_step >= int(self.fsm.target_step) for leg in ("RL", "RR"))
        if fsm_state.startswith("rear_swing_to_step_") and rear_confirmed_count == 1:
            cmd.x_vel_limit = min(cmd.x_vel_limit, 0.008)
            cmd.yaw_gain_scale *= 0.25
            cmd.y_gain_scale *= 1.35
        if fsm_state.startswith("rear_swing_to_step_") and not bool(getattr(self, "forced_rear_active", False)):
            cmd.x_vel_limit = 0.0
            cmd.yaw_gain_scale *= 0.25
            cmd.y_gain_scale *= 1.65
        hold_reason = str(self.fsm.hold_reason)
        if hold_reason.startswith("waiting_rear_confirmed"):
            cmd.x_vel_limit = min(cmd.x_vel_limit, 0.026)
        elif hold_reason.startswith(("waiting_", "front_settle", "rear_settle", "support_margin", "centerline", "unsafe_foothold")):
            cmd.x_vel_limit = min(cmd.x_vel_limit, PROFILE_FSM_HOLD_X_LIMIT)
        if hold_reason.startswith("unsafe_foothold"):
            cmd.x_vel_limit = min(cmd.x_vel_limit, PROFILE_FSM_REPLAN_X_LIMIT)
        settling_legs = _runtime_settling_legs(runtime)
        if settling_legs:
            cmd.x_vel_limit = min(cmd.x_vel_limit, PROFILE_TOUCHDOWN_X_LIMIT)
            cmd.yaw_gain_scale *= PROFILE_TOUCHDOWN_YAW_SCALE
            cmd.y_gain_scale *= PROFILE_TOUCHDOWN_Y_SCALE
            if any(leg in settling_legs for leg in ("FL", "FR")):
                cmd.x_vel_limit = min(cmd.x_vel_limit, 0.026)

        self.last_command = cmd
        runtime.phase_name = cmd.phase_name
        runtime.phase_foot_steps = foot_steps
        runtime.fsm_state = cmd.fsm_state
        runtime.fsm_target_step = cmd.fsm_target_step
        runtime.fsm_hold_reason = cmd.fsm_hold_reason
        return cmd


@dataclass
class FootholdPlanner:
    terrain: StairProfile
    phase_policy: StairPhasePolicy

    def _support_margin_score(self, go2: PinVBotModel, leg: str, candidate_xy: np.ndarray) -> float:
        points = []
        for name in ex18.MPC_LEG_ORDER:
            foot_pos, _ = go2.get_single_foot_state_in_world(name)
            foot_pos = np.asarray(foot_pos, dtype=float).reshape(3)
            if name == leg:
                points.append(np.asarray(candidate_xy, dtype=float).reshape(2))
            else:
                points.append(foot_pos[:2])
        try:
            return float(ex18._support_polygon_margin_xy(ex18._support_com_xy(go2), np.asarray(points, dtype=float)))
        except Exception:
            pts = np.asarray(points, dtype=float)
            com = ex18._support_com_xy(go2)
            return float(min(com[0] - np.min(pts[:, 0]), np.max(pts[:, 0]) - com[0]))

    def _choose_target_step(self, go2: PinVBotModel, leg: str, nominal: np.ndarray) -> tuple[int, str]:
        states = getattr(self.phase_policy.estimator, "states", {})
        target, source = self.phase_policy.fsm.target_for_leg(leg, states)
        target = int(target)
        nominal_step = self.phase_policy.stair_map.step_index_at(nominal[0], nominal[1])
        current_pos, _ = go2.get_single_foot_state_in_world(leg)
        current_step = self.phase_policy.stair_map.step_index_at(current_pos[0], current_pos[1])
        current_pos = np.asarray(current_pos, dtype=float).reshape(3)
        forced_rear_leg = getattr(self.phase_policy, "forced_rear_leg", None)
        forced_rear_active = bool(getattr(self.phase_policy, "forced_rear_active", False))
        forced_rear_stepup_active = bool(getattr(self.phase_policy, "forced_rear_stepup_active", False))

        if (
            forced_rear_active
            and forced_rear_stepup_active
            and forced_rear_leg == leg
            and leg in ("RL", "RR")
            and self.phase_policy.fsm.state.startswith("rear_swing_to_step_")
        ):
            return min(max(1, target), self.terrain.step_count), "fsm_rear_forced"

        if (
            leg in ("RL", "RR")
            and target > 0
            and self.phase_policy.fsm.state.startswith("body_commit_step_")
        ):
            return 0, "fsm_rear_body_commit_catchup"

        if (
            leg in ("RL", "RR")
            and target > 0
            and self.phase_policy.fsm.state.startswith("rear_swing_to_step_")
        ):
            lo, _ = self.phase_policy.stair_map.safe_tread_interval(target)
            reachable_x = float(current_pos[0]) + PROFILE_REAR_STEPUP_DIRECT_X
            if reachable_x < lo - 0.010:
                if forced_rear_active and forced_rear_leg == leg:
                    return 0, "fsm_rear_forced_approach_catchup"
                return 0, "fsm_rear_approach_catchup"

        if (
            target == 1
            and nominal_step <= 0
            and float(nominal[0]) < float(self.terrain.start_x) - ex18.STAIR_ACTIVE_FRONT_ADVANCE_WINDOW_X
            and not (leg in ("RL", "RR") and self.phase_policy.fsm.state.startswith(("body_commit_step_", "rear_swing_to_step_")))
        ):
            return 0, "nominal_approach"
        if nominal_step <= 0 and target <= 0:
            return 0, "nominal_approach"
        if target <= 0:
            return int(nominal_step), "nominal_map"
        if source.endswith("hold") and current_step > 0:
            return min(max(current_step, target), self.terrain.step_count), source
        if forced_rear_active and forced_rear_leg == leg and self.phase_policy.fsm.state.startswith("rear_swing_to_step_"):
            return min(max(1, target), self.terrain.step_count), "fsm_rear_forced"
        return min(max(1, target), self.terrain.step_count), source

    def plan(self, go2: PinVBotModel, leg: str, nominal_touchdown_world) -> tuple[np.ndarray, dict]:
        nominal = np.asarray(nominal_touchdown_world, dtype=float).reshape(3).copy()
        current_pos, _ = go2.get_single_foot_state_in_world(leg)
        current_pos = np.asarray(current_pos, dtype=float).reshape(3)
        stair_map = self.phase_policy.stair_map
        target_step, target_source = self._choose_target_step(go2, leg, nominal)
        nominal_step = stair_map.step_index_at(nominal[0], nominal[1])
        current_step = stair_map.step_index_at(current_pos[0], current_pos[1])
        rear_approach_catchup = target_source in (
            "fsm_rear_approach_catchup",
            "fsm_rear_forced_approach_catchup",
            "fsm_rear_body_commit_catchup",
        )
        forced_rear_leg = getattr(self.phase_policy, "forced_rear_leg", None)
        forced_rear_active = bool(getattr(self.phase_policy, "forced_rear_active", False))
        forced_rear_stepup = bool(
            forced_rear_active
            and forced_rear_leg == leg
            and leg in ("RL", "RR")
            and target_step > 0
        )
        forced_body_commit_catchup = bool(
            forced_rear_active
            and forced_rear_leg == leg
            and leg in ("RL", "RR")
            and target_source == "fsm_rear_body_commit_catchup"
        )
        if (
            target_step <= 0
            and not rear_approach_catchup
            and not self.terrain.in_stair_mode_x(nominal[0])
        ):
            pos = nominal.copy()
            pos[2] = self.terrain.height(pos[0], pos[1]) + FOOT_TOUCHDOWN_CLEARANCE
            debug = {
                "foot_nom_x": float(nominal[0]),
                "foot_nom_z": float(nominal[2]),
                "target_step": int(target_step),
                "foothold_projected": False,
                "projection_reason": "none",
                "projection_delta_x": 0.0,
                "projection_delta_y": 0.0,
                "selected_candidate_score": 0.0,
                "candidate_score": 0.0,
                "edge_distance": float(stair_map.edge_distance(pos[0], pos[1], target_step)),
                "target_step_source": target_source,
                "replan_reason": "none",
            }
            return pos, debug

        candidate_nominal = nominal.copy()
        if rear_approach_catchup:
            if forced_body_commit_catchup:
                safe_lo, _ = stair_map.safe_tread_interval(max(1, int(self.phase_policy.fsm.target_step)))
                prep_x = min(
                    float(self.terrain.start_x - PROFILE_SAFE_TREAD_MARGIN_X),
                    float(safe_lo) - PROFILE_REAR_STEPUP_DIRECT_X + 0.020,
                )
                incremental_x = float(current_pos[0]) + PROFILE_REAR_BODY_CATCHUP_MAX_DELTA_X
                candidate_nominal[0] = max(float(candidate_nominal[0]), min(float(prep_x), incremental_x))
            else:
                candidate_nominal[0] = max(
                    float(candidate_nominal[0]),
                    min(
                        float(self.terrain.start_x - PROFILE_SAFE_TREAD_MARGIN_X),
                        float(current_pos[0]) + PROFILE_REAR_APPROACH_CATCHUP_X,
                    ),
                )
        candidates = stair_map.candidate_footholds(candidate_nominal, target_step, leg)
        if not rear_approach_catchup and not forced_rear_stepup:
            legacy = self.terrain.project_to_safe_tread_result(
                nominal,
                leg=leg,
                nominal_pos=nominal,
                max_delta_x=ex18.STAIR_REAR_FOOTHOLD_MAX_DELTA_X if leg in ("RL", "RR") else ex18.STAIR_FOOTHOLD_MAX_DELTA_X,
                soft=True,
                margin=PROFILE_SAFE_TREAD_MARGIN_X,
            )
            legacy_pos = np.asarray(legacy.pos, dtype=float).reshape(3)
            legacy_step = stair_map.step_index_at(legacy_pos[0], legacy_pos[1])
            candidates.append(
                FootholdCandidate(
                    legacy_pos,
                    legacy_step,
                    stair_map.edge_distance(legacy_pos[0], legacy_pos[1], legacy_step),
                    "legacy_safe_projection_candidate",
                )
            )
        states = getattr(self.phase_policy.estimator, "states", {})
        confirmed_steps = {name: int(state.confirmed_step) for name, state in states.items()}
        best: FootholdCandidate | None = None
        best_score = -float("inf")
        best_reason = "no_candidate"
        max_dx_from_current = (
            PROFILE_REAR_FORCED_MAX_DELTA_X
            if forced_rear_stepup
            else PROFILE_REAR_BODY_CATCHUP_MAX_DELTA_X
            if forced_body_commit_catchup
            else (ex18.STAIR_REAR_FOOTHOLD_MAX_DELTA_X if leg in ("RL", "RR") else ex18.STAIR_FOOTHOLD_MAX_DELTA_X)
        )

        for cand in candidates:
            p = np.asarray(cand.pos, dtype=float).reshape(3)
            safe = stair_map.is_safe_foothold(p, target_step)
            if forced_rear_stepup and (not safe or int(cand.step_index) != int(target_step)):
                continue
            dist_ref = candidate_nominal if rear_approach_catchup else nominal
            dist_nom = float(np.linalg.norm((p - dist_ref)[:2]))
            projection_delta = float(np.linalg.norm((p - nominal)[:2]))
            height_error = abs(stair_map.step_height(target_step) - stair_map.height_at(p[0], p[1]))
            workspace_dx = max(0.0, abs(float(p[0] - current_pos[0])) - max_dx_from_current)
            workspace_y = max(0.0, abs(float(p[1] - current_pos[1])) - 0.26)
            support_margin = self._support_margin_score(go2, leg, p[:2])
            if leg in ("FL", "FR"):
                rear_ref = min(confirmed_steps.get("RL", 0), confirmed_steps.get("RR", 0))
                step_gap = max(0, int(target_step) - int(rear_ref) - 1)
            else:
                front_ref = min(confirmed_steps.get("FL", 0), confirmed_steps.get("FR", 0))
                step_gap = max(0, int(front_ref) - int(target_step))

            score = 0.0
            score -= 10.0 * dist_nom
            score += 4.0 * min(0.08, max(0.0, float(cand.edge_distance)))
            score -= 30.0 * height_error
            score -= 12.0 * workspace_dx + 8.0 * workspace_y
            edge = max(0.0, float(cand.edge_distance))
            score += 12.0 * min(0.09, edge)
            if target_step > 0 and edge < 0.030:
                score -= 45.0 * (0.030 - edge)
            score += 2.5 * np.clip(support_margin, -0.04, 0.08)
            score -= 1.5 * step_gap
            score -= 3.0 * projection_delta
            if rear_approach_catchup:
                forward_gain = float(p[0] - current_pos[0])
                gain_cap = (
                    PROFILE_REAR_BODY_CATCHUP_MAX_DELTA_X
                    if forced_body_commit_catchup
                    else PROFILE_REAR_APPROACH_CATCHUP_X
                )
                if forward_gain > gain_cap + 0.010:
                    score -= 120.0 + 500.0 * (forward_gain - gain_cap)
                score += 18.0 * max(0.0, min(gain_cap, forward_gain))
                score -= (26.0 if forced_body_commit_catchup else 16.0) * abs(float(p[0] - candidate_nominal[0]))
            elif leg in ("RL", "RR") and target_step > 0 and current_step <= 0:
                safe_lo, _ = stair_map.safe_tread_interval(target_step)
                tread_lo, _ = self.terrain.tread_interval(target_step)
                entry_ref = float(safe_lo + PROFILE_REAR_STEPUP_ENTRY_X) if forced_rear_stepup else float(tread_lo + 0.045)
                rear_entry_x = float(np.clip(entry_ref, safe_lo, safe_lo + 0.055))
                score -= (18.0 if forced_rear_stepup else 8.0) * abs(float(p[0] - rear_entry_x))
                if float(p[0]) > rear_entry_x + 0.045:
                    score -= 18.0 * (float(p[0]) - rear_entry_x - 0.045)
                if forced_rear_stepup and float(p[0]) < safe_lo + 0.006:
                    score -= 20.0 * (safe_lo + 0.006 - float(p[0]))
            if workspace_dx > 0.035 or workspace_y > 0.035:
                score -= 250.0 + 600.0 * (workspace_dx + workspace_y)
            if cand.step_index != target_step and target_step > 0:
                score -= 5.0
            if not safe:
                score -= 20.0
            if score > best_score:
                best_score = float(score)
                best = cand
                if not safe:
                    best_reason = "unsafe_candidate_fallback"
                elif cand.step_index != target_step and target_step > 0:
                    best_reason = "step_mismatch_fallback"
                elif projection_delta > 1e-5:
                    best_reason = cand.reason
                else:
                    best_reason = "nominal_safe"

        if best is None and forced_rear_stepup:
            safe_lo, _ = stair_map.safe_tread_interval(target_step)
            pos = nominal.copy()
            pos[0] = float(safe_lo + PROFILE_REAR_STEPUP_ENTRY_X)
            pos[1] = float(np.clip(pos[1], -0.18, 0.18))
            pos[2] = stair_map.step_height(target_step) + FOOT_TOUCHDOWN_CLEARANCE
            target_step = int(stair_map.step_index_at(pos[0], pos[1]))
            best_score = -999.0
            best_reason = "forced_rear_safe_entry_fallback"
        elif best is None:
            result = self.terrain.project_to_safe_tread_result(nominal, leg, nominal, margin=PROFILE_SAFE_TREAD_MARGIN_X)
            pos = result.pos
            target_step = result.target_step
            best_score = -999.0
            best_reason = "fallback_projection"
        else:
            pos = np.asarray(best.pos, dtype=float).reshape(3).copy()
            target_step = int(best.step_index)

        projected = bool(np.linalg.norm(pos[:2] - nominal[:2]) > 1e-5)
        edge_distance = float(stair_map.edge_distance(pos[0], pos[1], target_step))
        unsafe = not stair_map.is_safe_foothold(pos, target_step) if target_step > 0 else False
        replan_reason = "unsafe_foothold" if unsafe else ("edge_candidate" if edge_distance < PROFILE_EDGE_NEAR_X else "none")
        debug = {
            "foot_nom_x": float(nominal[0]),
            "foot_nom_z": float(nominal[2]),
            "target_step": int(target_step),
            "foothold_projected": projected,
            "projection_reason": best_reason if projected else "none",
            "projection_delta_x": float(pos[0] - nominal[0]),
            "projection_delta_y": float(pos[1] - nominal[1]),
            "selected_candidate_score": float(best_score),
            "candidate_score": float(best_score),
            "edge_distance": float(edge_distance),
            "target_step_source": target_source if target_step != nominal_step else f"{target_source}:nominal_step",
            "replan_reason": replan_reason,
        }
        return pos, debug


@dataclass
class SwingPlanner:
    terrain: StairProfile

    @staticmethod
    def _min_jerk_terms(s: float):
        s = float(np.clip(s, 0.0, 1.0))
        return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4, 60 * s - 180 * s**2 + 120 * s**3

    @staticmethod
    def _beta_bump_terms(s: float, peak_phase: float):
        s = float(np.clip(s, 0.0, 1.0))
        peak_phase = float(np.clip(peak_phase, 0.08, 0.90))
        sharpness = 8.5
        a = max(2.2, sharpness * peak_phase)
        b = max(2.2, sharpness * (1.0 - peak_phase))
        peak = a / (a + b)
        norm = max(1e-12, (peak**a) * ((1.0 - peak) ** b))
        if s <= 1e-9 or s >= 1.0 - 1e-9:
            return 0.0, 0.0, 0.0
        bump = (s**a) * ((1.0 - s) ** b) / norm
        log_d1 = a / s - b / (1.0 - s)
        log_d2 = -a / (s**2) - b / ((1.0 - s) ** 2)
        return float(bump), float(bump * log_d1), float(bump * (log_d1**2 + log_d2))

    def make(self, p0, pf, t_swing: float, takeoff_step: int, touchdown_step: int):
        p0 = np.asarray(p0, dtype=float).reshape(3)
        pf = np.asarray(pf, dtype=float).reshape(3)
        dp = pf - p0
        T = max(1e-6, float(t_swing))
        terrain_peak = self.terrain.max_height_between(p0, pf)
        takeoff_h = self.terrain.cumulative_height(takeoff_step)
        touchdown_h = self.terrain.cumulative_height(touchdown_step)
        step_delta = max(0.0, touchdown_h - takeoff_h)
        active = touchdown_step > takeoff_step

        dx = float(pf[0] - p0[0])
        if active and abs(dx) > 1e-6:
            riser_x = self.terrain.start_x + (touchdown_step - 1) * self.terrain.step_depth
            riser_phase = float(np.clip((riser_x - p0[0]) / dx, 0.05, 0.95))
            peak_phase = float(
                np.clip(
                    riser_phase - ex18.STAIR_ACTIVE_SWING_RISER_PHASE_LEAD,
                    ex18.STAIR_ACTIVE_SWING_MIN_PEAK_PHASE,
                    ex18.STAIR_ACTIVE_SWING_MAX_PEAK_PHASE,
                )
            )
        else:
            riser_phase = 0.50
            peak_phase = 0.50

        if active:
            swing_clearance = max(
                ex18.STAIR_ACTIVE_SWING_CLEARANCE,
                ex18.STAIR_ACTIVE_SWING_RISER_CLEARANCE + 0.25 * step_delta,
            )
            riser_clearance = max(
                ex18.STAIR_ACTIVE_SWING_RISER_CLEARANCE,
                0.55 * step_delta + 0.025,
            )
            required_peak_z = max(
                float(p0[2]),
                float(pf[2]),
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
        else:
            required_peak_z = max(
                float(p0[2]),
                float(pf[2]),
                terrain_peak + FOOT_TOUCHDOWN_CLEARANCE + ex18.STAIR_SWING_TERRAIN_CLEARANCE,
            )
            mid_z = 0.5 * (float(p0[2]) + float(pf[2]))
            bump_height = max(0.0, required_peak_z - mid_z)

        dense_s = np.linspace(0.0, 1.0, 51)
        base_z_values = np.asarray(
            [float(p0[2] + dp[2] * self._min_jerk_terms(s)[0]) for s in dense_s],
            dtype=float,
        )
        if active:
            bump_values = np.asarray(
                [self._beta_bump_terms(s, peak_phase)[0] for s in dense_s],
                dtype=float,
            )
            max_apex_z = max(
                float(p0[2]),
                float(pf[2]),
                terrain_peak + FOOT_TOUCHDOWN_CLEARANCE + ex18.STAIR_ACTIVE_SWING_MAX_TERRAIN_CLEARANCE,
            )
            if bump_height > 0.0 and np.max(bump_values) > 1e-6:
                max_base_z = float(np.max(base_z_values))
                max_bump = float(np.max(bump_values))
                bump_height = min(bump_height, max(0.0, (max_apex_z - max_base_z) / max_bump))
            apex_z = max(float(base_z) + bump_height * float(bump) for base_z, bump in zip(base_z_values, bump_values))
        else:
            bump_values = np.asarray([64 * s**3 * (1 - s) ** 3 for s in dense_s], dtype=float)
            apex_z = max(float(base_z) + bump_height * float(bump) for base_z, bump in zip(base_z_values, bump_values))
        debug = {
            "terrain_peak": float(terrain_peak),
            "swing_apex_z": float(apex_z),
            "swing_clearance": float(max(0.0, apex_z - terrain_peak - FOOT_TOUCHDOWN_CLEARANCE)),
            "riser_phase": float(riser_phase),
            "peak_phase": float(peak_phase),
            "terminal_vz_limit": float(PROFILE_MAX_TOUCHDOWN_VZ),
        }

        def eval_at(t):
            s = float(np.clip(t / T, 0.0, 1.0))
            mj, dmj, d2mj = self._min_jerk_terms(s)
            p = p0 + dp * mj
            v = dp * dmj / T
            a = dp * d2mj / (T**2)
            if active:
                bump, dbump, d2bump = self._beta_bump_terms(s, peak_phase)
            else:
                bump = 64 * s**3 * (1 - s) ** 3
                dbump = 192 * s**2 * (1 - s) ** 2 * (1 - 2 * s)
                d2bump = 192 * (
                    2 * s * (1 - s) ** 2 * (1 - 2 * s)
                    - 2 * s**2 * (1 - s) * (1 - 2 * s)
                    - 2 * s**2 * (1 - s) ** 2
                )
            p[2] += bump_height * bump
            v[2] += bump_height * dbump / T
            a[2] += bump_height * d2bump / (T**2)
            if s >= 0.62 and v[2] < 0.0:
                q = float(np.clip((s - 0.78) / 0.22, 0.0, 1.0))
                q = q * q * (3.0 - 2.0 * q)
                max_down_vz = PROFILE_MAX_TOUCHDOWN_VZ * (1.0 - q)
                if v[2] < max_down_vz:
                    v[2] = max_down_vz
                    a[2] = max(a[2], -1.5)
            return p, v, a

        return eval_at, debug


@dataclass
class BodyReferencePlanner:
    terrain: StairProfile
    z_cmd: float = ex18.NOMINAL_BODY_HEIGHT
    pitch_cmd: float = 0.0

    def update(self, vbot: PinVBotModel, cmd, runtime, contact_mask, phase: PhaseCommand, dt: float):
        base = np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)
        support_h = self.terrain.body_support_height(base)
        preview_h = self.terrain.height(base[0] + 0.08, base[1])
        samples = ex18.get_support_foot_samples(vbot, self.terrain, contact_mask)
        foot_heights = [sample[2] for sample in samples if sample[3]]
        foot_positions = {}
        mask = np.asarray(contact_mask, dtype=int).reshape(-1)
        for leg_i, leg in enumerate(ex18.MPC_LEG_ORDER):
            if leg_i < mask.size and int(mask[leg_i]) == 1:
                foot_positions[leg] = vbot.get_single_foot_state_in_world(leg)[0]
        foot_support_h = self.terrain.feet_support_height(foot_positions)
        confirmed_heights = []
        leg_states = getattr(runtime, "leg_terrain_states", {})
        for state in leg_states.values():
            if int(getattr(state, "confirmed_step", 0)) > 0:
                confirmed_heights.append(self.terrain.cumulative_height(int(state.confirmed_step)))
        confirmed_support_h = max(confirmed_heights) if confirmed_heights else foot_support_h
        if phase.fsm_state in ("approach", "front_swing_to_step_1") and not confirmed_heights:
            structural_h = 0.0
        else:
            structural_h = confirmed_support_h
        terrain_h = max(
            ex18.STAIR_BODY_HEIGHT_GAIN * structural_h,
            ex18.STAIR_BODY_FOOT_HEIGHT_GAIN * foot_support_h,
        ) + phase.z_bias
        z_target = ex18.NOMINAL_BODY_HEIGHT + terrain_h
        settling_legs = _runtime_settling_legs(runtime)
        front_settling = any(leg in settling_legs for leg in ("FL", "FR"))
        z_rate = PROFILE_BODY_Z_RAMP
        if front_settling and phase.phase_name in ("front_step_up", "body_commit", "front_stable"):
            z_rate *= 0.55
        self.z_cmd = ex18.ramp_value(self.z_cmd, z_target, z_rate, dt)

        pitch_target = 0.0
        if phase.phase_name != "landing":
            pitch_target = ex18.estimate_support_pitch(samples)
            pitch_target += phase.pitch_bias
            if phase.fsm_state.startswith(("body_commit_step_", "rear_swing_to_step_")):
                pitch_target = min(pitch_target, 0.055)
            elif phase.fsm_state.startswith("front_swing_to_step_"):
                pitch_target = min(pitch_target, 0.075)
            pitch_target = float(np.clip(pitch_target, -ex18.STAIR_PITCH_LIMIT, ex18.STAIR_PITCH_LIMIT))
        pitch_rate = PROFILE_BODY_PITCH_RAMP
        if front_settling:
            pitch_rate *= 0.50
        if phase.phase_name == "landing":
            pitch_rate = min(pitch_rate, 0.060)
        self.pitch_cmd = ex18.ramp_value(self.pitch_cmd, pitch_target, pitch_rate, dt)

        cmd.z_pos = self.z_cmd
        cmd.pitch = self.pitch_cmd
        runtime.pitch_ref = self.pitch_cmd
        runtime.support_h = support_h
        runtime.foot_support_h = foot_support_h
        runtime.confirmed_support_h = float(confirmed_support_h)
        return support_h, preview_h, foot_support_h


class ProfileStairGait(ex18.StairAwareGait):
    def __init__(self, frequency_hz, duty, terrain: StairProfile, swing_height=ex18.SWING_HEIGHT, phase_offset=None):
        super().__init__(frequency_hz, duty, terrain, swing_height=swing_height, phase_offset=phase_offset)
        self.terrain = terrain
        self.phase_policy = ACTIVE_PHASE_POLICY or StairPhasePolicy(terrain)
        self.foothold_planner = FootholdPlanner(terrain, self.phase_policy)
        self.swing_planner = SwingPlanner(terrain)
        self._forced_rear_leg: str | None = None
        self._forced_rear_start_s: float = float("nan")
        self._forced_rear_end_s: float = float("nan")
        self._forced_rear_mask_start_s: float = float("nan")
        self._forced_rear_retry_after_s: float = 0.0
        self._forced_rear_waiting_for_support: bool = False
        self._forced_rear_stepup_active: bool = False

    def _clear_forced_rear(self):
        self._forced_rear_leg = None
        self._forced_rear_start_s = float("nan")
        self._forced_rear_end_s = float("nan")
        self._forced_rear_mask_start_s = float("nan")
        self._forced_rear_retry_after_s = 0.0
        self._forced_rear_waiting_for_support = False
        self._forced_rear_stepup_active = False
        self.phase_policy.forced_rear_leg = None
        self.phase_policy.forced_rear_active = False
        self.phase_policy.forced_rear_phase = "idle"
        self.phase_policy.forced_rear_stepup_active = False

    def update_forced_rear_schedule(self, time_s: float, runtime=None):
        fsm = getattr(self.phase_policy, "fsm", None)
        fsm_state = str(getattr(fsm, "state", ""))
        rear_stepup_state = fsm_state.startswith("rear_swing_to_step_")
        body_commit_state = fsm_state.startswith("body_commit_step_")
        if fsm is None or not (rear_stepup_state or body_commit_state):
            self._clear_forced_rear()
            return

        target = int(getattr(fsm, "target_step", 1))
        states = getattr(self.phase_policy.estimator, "states", {})
        if body_commit_state:
            safe_lo, _ = self.phase_policy.stair_map.safe_tread_interval(target)
            pending = []
            for leg in ("RL", "RR"):
                state = states.get(leg, LegTerrainState())
                foot_x = float(getattr(state, "foot_x", np.nan))
                prep_x = float(safe_lo) - PROFILE_REAR_STEPUP_DIRECT_X - 0.010
                if (
                    int(getattr(state, "confirmed_step", 0)) < target
                    and int(getattr(state, "rear_prepare_step", 0)) < target
                    and (
                    not np.isfinite(foot_x)
                    or foot_x < prep_x
                    )
                ):
                    pending.append(leg)
        else:
            pending = [
                leg for leg in ("RL", "RR")
                if int(states.get(leg, LegTerrainState()).confirmed_step) < target
            ]
        now = float(time_s)
        active_leg = self._forced_rear_leg
        safe_lo, _ = self.phase_policy.stair_map.safe_tread_interval(target)
        schedule_vbot = getattr(runtime, "profile_vbot_for_schedule", None) if runtime is not None else None

        if rear_stepup_state and active_leg in ("RL", "RR") and not bool(self._forced_rear_stepup_active):
            self._clear_forced_rear()
            active_leg = None

        def leg_stepup_ready(name: str) -> tuple[bool, float]:
            try:
                foot_pos, _ = schedule_vbot.get_single_foot_state_in_world(name)
            except Exception:
                return False, -float("inf")
            if foot_pos is None:
                return False, -float("inf")
            foot_pos = np.asarray(foot_pos, dtype=float).reshape(3)
            foot_x = float(foot_pos[0])
            return foot_x + PROFILE_REAR_STEPUP_DIRECT_X >= float(safe_lo) - 0.010, foot_x

        def candidate_support(name: str) -> tuple[float, float]:
            support_margin = float(getattr(runtime, "support_margin_xy", np.nan)) if runtime is not None else np.nan
            support_center_y = float(getattr(runtime, "support_center_y", np.nan)) if runtime is not None else np.nan
            if schedule_vbot is not None and name in ex18.LEG_INDEX:
                try:
                    candidate_mask = np.ones(4, dtype=int)
                    candidate_mask[ex18.LEG_INDEX[name]] = 0
                    samples = ex18.get_support_foot_samples(schedule_vbot, self.terrain, candidate_mask)
                    points_xy = np.asarray([sample[1][0:2] for sample in samples], dtype=float).reshape(-1, 2)
                    com_xy = ex18._support_com_xy(schedule_vbot)
                    if len(points_xy) >= 3:
                        support_margin = float(ex18._support_polygon_margin_xy(com_xy, points_xy))
                        support_center_y = float(np.mean(points_xy[:, 1]))
                except Exception:
                    pass
            return support_margin, support_center_y

        def support_score(name: str) -> float:
            margin, center_y = candidate_support(name)
            margin_score = -0.20 if not np.isfinite(margin) else float(np.clip(margin, -0.08, 0.12))
            center_penalty = 0.0 if not np.isfinite(center_y) else 0.45 * abs(float(center_y))
            state = states.get(name, LegTerrainState())
            edge = float(getattr(state, "edge_distance", 0.0))
            edge_score = 0.0 if not np.isfinite(edge) else 0.10 * float(np.clip(edge, -0.02, 0.12))
            return margin_score + edge_score - center_penalty

        active_cycle_leg = active_leg in ("RL", "RR") and np.isfinite(self._forced_rear_end_s)
        cycle_complete = False
        if active_cycle_leg:
            if now < self._forced_rear_end_s:
                if body_commit_state and not bool(self._forced_rear_stepup_active):
                    roll_abs = 0.0
                    if schedule_vbot is not None:
                        try:
                            roll_abs = abs(float(schedule_vbot.current_config.compute_euler_angle_world()[0]))
                        except Exception:
                            roll_abs = 0.0
                    support_margin_now = float(getattr(runtime, "support_margin_xy", np.nan)) if runtime is not None else np.nan
                    guard_active_now = bool(getattr(runtime, "stair_guard", False)) if runtime is not None else False
                    elapsed = now - float(self._forced_rear_start_s)
                    abort_catchup = bool(
                        elapsed >= 0.080
                        and (
                            guard_active_now
                            or roll_abs > 0.18
                            or (np.isfinite(support_margin_now) and support_margin_now < 0.006)
                        )
                    )
                    if abort_catchup:
                        self._forced_rear_end_s = now
                        self._forced_rear_retry_after_s = (
                            now + PROFILE_REAR_FORCED_SWING_SETTLE_S + PROFILE_REAR_FORCED_RETRY_S
                        )
                        self.phase_policy.forced_rear_leg = active_leg
                        self.phase_policy.forced_rear_active = False
                        self.phase_policy.forced_rear_phase = "settle"
                        self.phase_policy.forced_rear_stepup_active = False
                        return
                self.phase_policy.forced_rear_leg = active_leg
                self.phase_policy.forced_rear_active = True
                self.phase_policy.forced_rear_phase = "swing"
                self.phase_policy.forced_rear_stepup_active = bool(self._forced_rear_stepup_active)
                return
            settle_until = self._forced_rear_end_s + PROFILE_REAR_FORCED_SWING_SETTLE_S
            if now < settle_until:
                self.phase_policy.forced_rear_leg = active_leg
                self.phase_policy.forced_rear_active = False
                self.phase_policy.forced_rear_phase = "settle"
                self.phase_policy.forced_rear_stepup_active = bool(self._forced_rear_stepup_active)
                return
            if now < self._forced_rear_retry_after_s:
                self.phase_policy.forced_rear_leg = active_leg
                self.phase_policy.forced_rear_active = False
                self.phase_policy.forced_rear_phase = "retry_wait"
                self.phase_policy.forced_rear_stepup_active = False
                return
            else:
                cycle_complete = True

        if not pending:
            self._clear_forced_rear()
            return

        stepup_ready = []
        for name in pending:
            ready, foot_x = leg_stepup_ready(name)
            if ready:
                stepup_ready.append((foot_x, name))

        if rear_stepup_state and stepup_ready:
            ready_legs = [name for _, name in stepup_ready]
            leg = max(ready_legs, key=support_score)
            stepup_active = True
        elif self._forced_rear_waiting_for_support and active_leg in pending and len(pending) == 1 and not cycle_complete:
            leg = active_leg
            stepup_active = bool(self._forced_rear_stepup_active)
        else:
            leg = max(pending, key=support_score)
            stepup_active = False

        support_margin, support_center_y = candidate_support(leg)
        guard_active = bool(getattr(runtime, "stair_guard", False)) if runtime is not None else False
        min_support_margin = (
            PROFILE_REAR_FORCED_MIN_SUPPORT_MARGIN
            if rear_stepup_state
            else PROFILE_REAR_BODY_CATCHUP_MIN_SUPPORT_MARGIN
        )
        max_support_center_y = (
            1.5 * PROFILE_REAR_FORCED_MAX_SUPPORT_CENTER_Y
            if rear_stepup_state
            else PROFILE_REAR_FORCED_MAX_SUPPORT_CENTER_Y
        )
        support_ready = (
            ((not guard_active) or rear_stepup_state)
            and (not np.isfinite(support_margin) or support_margin >= min_support_margin)
            and (not np.isfinite(support_center_y) or abs(support_center_y) <= max_support_center_y)
        )
        if not support_ready:
            self._forced_rear_leg = leg
            self.phase_policy.forced_rear_leg = leg
            self.phase_policy.forced_rear_active = False
            self.phase_policy.forced_rear_phase = "support_wait"
            self._forced_rear_stepup_active = bool(stepup_active)
            self.phase_policy.forced_rear_stepup_active = bool(stepup_active)
            self._forced_rear_waiting_for_support = True
            return

        self._forced_rear_leg = leg
        self._forced_rear_start_s = now
        self._forced_rear_mask_start_s = float("nan")
        self._forced_rear_waiting_for_support = False
        self._forced_rear_stepup_active = bool(stepup_active)
        forced_swing_time = max(float(self.swing_time), float(PROFILE_REAR_FORCED_SWING_TIME))
        self._forced_rear_end_s = now + forced_swing_time
        self._forced_rear_retry_after_s = self._forced_rear_end_s + PROFILE_REAR_FORCED_SWING_SETTLE_S + PROFILE_REAR_FORCED_RETRY_S
        self.phase_policy.forced_rear_leg = leg
        self.phase_policy.forced_rear_active = True
        self.phase_policy.forced_rear_phase = "swing"
        self.phase_policy.forced_rear_stepup_active = bool(stepup_active)
        if runtime is not None:
            runtime.forced_rear_leg = leg
            runtime.forced_rear_active = True
            runtime.forced_rear_phase = "swing"
            runtime.forced_rear_stepup_active = int(bool(stepup_active))


    def compute_current_mask(self, time):
        mask = np.asarray(super().compute_current_mask(time), dtype=int).reshape(4)
        fsm = getattr(self.phase_policy, "fsm", None)
        fsm_state = str(getattr(fsm, "state", ""))
        rear_stepup_state = fsm_state.startswith("rear_swing_to_step_")
        body_commit_state = fsm_state.startswith("body_commit_step_")
        if fsm is None or not (rear_stepup_state or body_commit_state):
            return mask
        target = int(getattr(fsm, "target_step", 1))
        states = getattr(self.phase_policy.estimator, "states", {})
        if body_commit_state:
            safe_lo, _ = self.phase_policy.stair_map.safe_tread_interval(target)
            rear_pending = []
            for leg in ("RL", "RR"):
                state = states.get(leg, LegTerrainState())
                foot_x = float(getattr(state, "foot_x", np.nan))
                prep_x = float(safe_lo) - PROFILE_REAR_STEPUP_DIRECT_X - 0.010
                if (
                    int(getattr(state, "confirmed_step", 0)) < target
                    and int(getattr(state, "rear_prepare_step", 0)) < target
                    and (
                    not np.isfinite(foot_x)
                    or foot_x < prep_x
                    )
                ):
                    rear_pending.append(leg)
        else:
            rear_pending = [
                leg for leg in ("RL", "RR")
                if int(states.get(leg, LegTerrainState()).confirmed_step) < target
            ]
        forced_leg = self._forced_rear_leg
        if (
            forced_leg in ("RL", "RR")
            and np.isfinite(self._forced_rear_start_s)
            and float(self._forced_rear_start_s) <= float(time) < float(self._forced_rear_end_s)
        ):
            if not np.isfinite(self._forced_rear_mask_start_s):
                self._forced_rear_mask_start_s = float(time)
                forced_swing_time = max(float(self.swing_time), float(PROFILE_REAR_FORCED_SWING_TIME))
                self._forced_rear_start_s = float(time)
                self._forced_rear_end_s = float(time) + forced_swing_time
                self._forced_rear_retry_after_s = (
                    self._forced_rear_end_s
                    + PROFILE_REAR_FORCED_SWING_SETTLE_S
                    + PROFILE_REAR_FORCED_RETRY_S
                )
            forced_mask = np.ones(4, dtype=int)
            forced_mask[ex18.LEG_INDEX[forced_leg]] = 0
            return forced_mask
        if not rear_pending:
            if body_commit_state:
                return np.ones(4, dtype=int)
            return mask
        return np.ones(4, dtype=int)

    def compute_touchdown_world_for_traj_purpose_only(self, go2: PinVBotModel, leg: str, terrain_height_fn=None):
        nominal_td = Gait.compute_touchdown_world_for_traj_purpose_only(
            self,
            go2,
            leg,
            terrain_height_fn=self.terrain.height,
        )
        touchdown, debug = self.foothold_planner.plan(go2, leg, nominal_td)
        self.last_swing_debug[leg] = {**self.last_swing_debug.get(leg, {}), **debug}
        return touchdown

    def compute_swing_traj_and_touchdown(self, go2: PinVBotModel, leg: str):
        foot_pos, _ = go2.get_single_foot_state_in_world(leg)
        _, nominal_td = Gait.compute_swing_traj_and_touchdown(self, go2, leg)
        nominal_td = np.asarray(nominal_td, dtype=float).reshape(3).copy()
        td, foothold_debug = self.foothold_planner.plan(go2, leg, nominal_td)
        td = np.asarray(td, dtype=float).reshape(3)
        takeoff_step = self.terrain.step_index_at(foot_pos[0], foot_pos[1])
        touchdown_step = self.terrain.step_index_at(td[0], td[1])
        t_swing = float(self.swing_time)
        if (
            leg in ("RL", "RR")
            and leg == self._forced_rear_leg
            and bool(getattr(self.phase_policy, "forced_rear_active", False))
            and str(getattr(self.phase_policy.fsm, "state", "")).startswith(("body_commit_step_", "rear_swing_to_step_"))
        ):
            t_swing = max(t_swing, float(PROFILE_REAR_FORCED_SWING_TIME))
        foot_traj, swing_debug = self.swing_planner.make(
            foot_pos,
            td,
            t_swing,
            takeoff_step,
            touchdown_step,
        )
        projected = bool(foothold_debug.get("foothold_projected", np.linalg.norm(td[:2] - nominal_td[:2]) > 1e-5))
        reason = str(foothold_debug.get("projection_reason", "none"))
        debug = {}
        debug.update(
            {
                "takeoff_step": int(takeoff_step),
                "touchdown_step": int(touchdown_step),
                "target_step": int(foothold_debug.get("target_step", touchdown_step)),
                "foothold_projected": projected,
                "projection_reason": reason if projected else "none",
            }
        )
        debug = {**foothold_debug, **swing_debug, **debug}
        self.last_swing_debug[leg] = debug
        return foot_traj, td


ACTIVE_PROFILE_NAME = "uniform06"
ACTIVE_PHASE_POLICY: StairPhasePolicy | None = None
ACTIVE_BODY_PLANNER: BodyReferencePlanner | None = None
ACTIVE_HEADLESS_START_DELAY_S = 0.0
ACTIVE_HEADLESS_DELAY_TARGET_X_VEL: float | None = None


def _smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _ensure_profile_runtime(runtime, mask=None):
    if hasattr(runtime, "profile_touchdown_age"):
        return
    init_mask = np.ones(4, dtype=int) if mask is None else np.asarray(mask, dtype=int).reshape(4)
    runtime.profile_contact_mask = init_mask.copy()
    runtime.profile_touchdown_age = {}
    runtime.profile_stance_blend = {}
    runtime.profile_force_blend = {}
    for leg_i, leg in enumerate(ex18.MPC_LEG_ORDER):
        in_stance = int(init_mask[leg_i]) == 1
        runtime.profile_touchdown_age[leg] = PROFILE_TOUCHDOWN_SETTLE_S if in_stance else np.nan
        runtime.profile_stance_blend[leg] = 1.0 if in_stance else 0.0
        runtime.profile_force_blend[leg] = 1.0 if in_stance else 0.0
    runtime.profile_filtered_mpc_force = np.zeros(12, dtype=float)
    runtime.profile_prev_tau_cmd = None
    runtime.profile_tau_rate_limited = False
    runtime.profile_force_rate_limited = False


def _profile_reset_smoothing(runtime):
    for name in (
        "profile_contact_mask",
        "profile_touchdown_age",
        "profile_stance_blend",
        "profile_force_blend",
        "profile_filtered_mpc_force",
        "profile_prev_tau_cmd",
        "profile_tau_rate_limited",
        "profile_force_rate_limited",
        "profile_smoothing_active",
        "profile_prev_x_eff",
        "profile_prev_y_eff",
        "profile_prev_yaw_eff",
        "profile_entry_damping",
        "profile_guard_recovery",
        "leg_terrain_states",
        "fsm_state",
        "fsm_target_step",
        "fsm_hold_reason",
        "confirmed_support_h",
    ):
        if hasattr(runtime, name):
            delattr(runtime, name)


def _profile_update_contact_state(runtime, contact_mask, dt: float):
    mask = np.asarray(contact_mask, dtype=int).reshape(4)
    _ensure_profile_runtime(runtime, mask)
    prev = np.asarray(getattr(runtime, "profile_contact_mask", mask), dtype=int).reshape(4)
    for leg_i, leg in enumerate(ex18.MPC_LEG_ORDER):
        contact_now = int(mask[leg_i]) == 1
        contact_prev = int(prev[leg_i]) == 1
        if contact_now:
            if not contact_prev:
                age = 0.0
            else:
                age_prev = runtime.profile_touchdown_age.get(leg, PROFILE_TOUCHDOWN_SETTLE_S)
                age = PROFILE_TOUCHDOWN_SETTLE_S if not np.isfinite(age_prev) else float(age_prev) + float(dt)
            age = float(np.clip(age, 0.0, PROFILE_TOUCHDOWN_SETTLE_S))
            blend = PROFILE_TOUCHDOWN_MIN_FORCE_BLEND + (
                1.0 - PROFILE_TOUCHDOWN_MIN_FORCE_BLEND
            ) * _smoothstep01(age / max(1e-6, PROFILE_TOUCHDOWN_SETTLE_S))
        else:
            age = np.nan
            blend = 0.0
        runtime.profile_touchdown_age[leg] = age
        runtime.profile_stance_blend[leg] = float(blend)
        runtime.profile_force_blend[leg] = float(blend)
    runtime.profile_contact_mask = mask.copy()


def _runtime_settling_legs(runtime):
    if runtime is None or not hasattr(runtime, "profile_touchdown_age"):
        return []
    if not bool(getattr(runtime, "profile_smoothing_active", False)):
        return []
    legs = []
    for leg in ex18.MPC_LEG_ORDER:
        age = float(runtime.profile_touchdown_age.get(leg, np.nan))
        if np.isfinite(age) and age < PROFILE_TOUCHDOWN_SETTLE_S:
            legs.append(leg)
    return legs


def _profile_sync_forced_rear_runtime(runtime, gait):
    policy = getattr(gait, "phase_policy", None)
    if runtime is None or policy is None:
        return
    runtime.forced_rear_leg = str(getattr(policy, "forced_rear_leg", "") or "")
    runtime.forced_rear_active = int(bool(getattr(policy, "forced_rear_active", False)))
    runtime.forced_rear_phase = str(getattr(policy, "forced_rear_phase", "idle"))
    runtime.forced_rear_stepup_active = int(bool(getattr(policy, "forced_rear_stepup_active", False)))
    runtime.forced_rear_start_s = float(getattr(gait, "_forced_rear_start_s", np.nan))
    runtime.forced_rear_end_s = float(getattr(gait, "_forced_rear_end_s", np.nan))
    runtime.forced_rear_retry_after_s = float(getattr(gait, "_forced_rear_retry_after_s", np.nan))


def _profile_ramp_runtime_value(runtime, attr: str, target: float, rate: float, dt: float) -> float:
    target = float(target)
    prev = getattr(runtime, attr, None)
    if prev is None or not np.isfinite(float(prev)):
        value = target
    else:
        max_delta = max(0.0, float(rate)) * float(dt)
        value = float(prev) + float(np.clip(target - float(prev), -max_delta, max_delta))
    setattr(runtime, attr, value)
    return value


def _profile_start_walking_phase(runtime, time_now_s: float, vbot, leg_controller, traj, gait):
    ex18.start_walking_phase(runtime, time_now_s, vbot, leg_controller, traj, gait)
    runtime.profile_walk_start_time_s = float(time_now_s)
    runtime.profile_start_phase_hold_s = float(PROFILE_START_PHASE_HOLD_S)
    if PROFILE_START_PHASE_HOLD_S > 0.0:
        runtime.gait_start_time_s = float(time_now_s) + float(PROFILE_START_PHASE_HOLD_S)
        runtime.gait_time_s = 0.0
        leg_controller.last_mask = gait.compute_current_mask(0.0).reshape(4).copy()


def _profile_start_age(runtime, time_now_s: float) -> float:
    start_t = getattr(runtime, "profile_walk_start_time_s", None)
    if start_t is None:
        return np.inf
    return max(0.0, float(time_now_s) - float(start_t))


def _profile_apply_entry_damping(vbot, terrain, runtime, x_eff: float, y_eff: float, stair_mode: bool, stair_guard: bool):
    base_pos = np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)
    roll, _, _ = vbot.current_config.compute_euler_angle_world()
    in_entry = (
        float(terrain.start_x) - PROFILE_ENTRY_DAMPING_BEFORE_X
        <= float(base_pos[0])
        <= float(terrain.start_x) + PROFILE_ENTRY_DAMPING_AFTER_X
    )
    if not in_entry or stair_guard:
        runtime.profile_entry_damping = False
        return float(x_eff), float(y_eff)

    y_abs = abs(float(base_pos[1]))
    roll_abs = abs(float(roll))
    risk = max(
        y_abs / max(1e-6, PROFILE_ENTRY_Y_SLOW),
        roll_abs / max(1e-6, PROFILE_ENTRY_ROLL_SLOW),
    )
    if risk <= 0.75 and not stair_mode:
        runtime.profile_entry_damping = False
        return float(x_eff), float(y_eff)

    runtime.profile_entry_damping = True
    if x_eff > 0.0:
        if risk > 1.65:
            x_eff = min(float(x_eff), 0.006)
        elif risk > 1.0:
            x_eff = min(float(x_eff), 0.014)
        else:
            x_eff = min(float(x_eff), 0.026)
    y_center = -PROFILE_ENTRY_Y_KP * float(base_pos[1])
    y_eff = float(np.clip(float(y_eff) + y_center, -PROFILE_ENTRY_Y_LIMIT, PROFILE_ENTRY_Y_LIMIT))
    return float(x_eff), float(y_eff)


def _profile_apply_guard_recovery(vbot, cmd, runtime, x_eff: float, y_eff: float, yaw_rate_eff: float):
    if not bool(getattr(runtime, "stair_guard", False)):
        runtime.profile_guard_recovery = False
        return float(x_eff), float(y_eff), float(yaw_rate_eff)

    base_pos = np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)
    roll, _, _ = vbot.current_config.compute_euler_angle_world()
    y_abs = abs(float(base_pos[1]))
    roll_abs = abs(float(roll))
    severity = max(
        y_abs / max(1e-6, float(ex18.STAIR_GUARD_Y)),
        roll_abs / max(1e-6, float(ex18.STAIR_GUARD_ROLL)),
    )

    runtime.profile_guard_recovery = True
    if float(cmd.x_vel) > ex18.STAND_VEL_EPS:
        alpha = float(np.clip((severity - 1.0) / 1.5, 0.0, 1.0))
        recovery_x = PROFILE_GUARD_RECOVERY_X_MAX + alpha * (
            PROFILE_GUARD_RECOVERY_X_MIN - PROFILE_GUARD_RECOVERY_X_MAX
        )
        x_eff = min(max(float(x_eff), recovery_x), recovery_x)

    y_center = -PROFILE_GUARD_RECOVERY_Y_KP * float(base_pos[1])
    y_eff = float(np.clip(float(cmd.y_vel) + y_center, -PROFILE_GUARD_RECOVERY_Y_LIMIT, PROFILE_GUARD_RECOVERY_Y_LIMIT))
    yaw_rate_eff = float(yaw_rate_eff) * PROFILE_GUARD_YAW_SCALE
    return float(x_eff), float(y_eff), float(yaw_rate_eff)


def _profile_guard_should_hold_stance(vbot, runtime, stair_mode: bool) -> bool:
    if not stair_mode or not bool(getattr(runtime, "stair_guard", False)):
        return False
    base_pos = np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)
    roll, _, _ = vbot.current_config.compute_euler_angle_world()
    support_margin = float(getattr(runtime, "support_margin_xy", np.nan))
    return bool(
        abs(float(roll)) > PROFILE_GUARD_STANCE_ROLL
        or abs(float(base_pos[1])) > PROFILE_GUARD_STANCE_Y
        or (np.isfinite(support_margin) and support_margin < PROFILE_GUARD_STANCE_SUPPORT_MARGIN)
    )


def _profile_apply_force_shaping(runtime, mpc_force_raw, contact_mask, dt: float) -> np.ndarray:
    mask = np.asarray(contact_mask, dtype=int).reshape(4)
    _ensure_profile_runtime(runtime, mask)
    raw = np.asarray(mpc_force_raw, dtype=float).reshape(12)
    if not bool(getattr(runtime, "profile_smoothing_active", False)):
        runtime.profile_filtered_mpc_force = raw.copy()
        runtime.profile_force_rate_limited = False
        return raw
    shaped = np.zeros(12, dtype=float)
    for leg_i, leg in enumerate(ex18.MPC_LEG_ORDER):
        leg_slice = ex18.LEG_SLICE[leg]
        if int(mask[leg_i]) == 1:
            blend = float(runtime.profile_force_blend.get(leg, 1.0))
            shaped[leg_slice] = raw[leg_slice] * blend
    prev = np.asarray(runtime.profile_filtered_mpc_force, dtype=float).reshape(12)
    max_delta = PROFILE_FORCE_RATE_LIMIT * float(dt)
    filtered = prev + np.clip(shaped - prev, -max_delta, max_delta)
    runtime.profile_force_rate_limited = bool(np.any(np.abs(filtered - shaped) > 1e-8))
    runtime.profile_filtered_mpc_force = filtered.copy()
    return filtered


def _profile_apply_tau_rate_limit(runtime, tau_cmd, contact_mask, dt: float) -> np.ndarray:
    tau = np.asarray(tau_cmd, dtype=float).reshape(12)
    if not bool(getattr(runtime, "profile_smoothing_active", False)):
        runtime.profile_prev_tau_cmd = tau.copy()
        runtime.profile_tau_rate_limited = False
        return tau
    prev = getattr(runtime, "profile_prev_tau_cmd", None)
    if prev is None:
        runtime.profile_prev_tau_cmd = tau.copy()
        runtime.profile_tau_rate_limited = False
        return tau
    prev = np.asarray(prev, dtype=float).reshape(12)
    max_delta = PROFILE_TAU_RATE_LIMIT * float(dt)
    mask = np.asarray(contact_mask, dtype=int).reshape(4)
    limited = tau.copy()
    for leg_i, leg in enumerate(ex18.MPC_LEG_ORDER):
        leg_slice = ex18.LEG_SLICE[leg]
        if int(mask[leg_i]) == 1:
            limited[leg_slice] = prev[leg_slice] + np.clip(
                tau[leg_slice] - prev[leg_slice],
                -max_delta[leg_slice],
                max_delta[leg_slice],
            )
    runtime.profile_tau_rate_limited = bool(np.any(np.abs(limited - tau) > 1e-8))
    runtime.profile_prev_tau_cmd = limited.copy()
    return limited


def _compute_support_metrics(vbot, terrain, runtime, contact_mask):
    samples = ex18.get_support_foot_samples(vbot, terrain, contact_mask)
    points_xy = np.asarray([sample[1][0:2] for sample in samples], dtype=float).reshape(-1, 2)
    com_xy = ex18._support_com_xy(vbot)
    runtime.support_count = len(samples)
    if len(samples) >= 2:
        xs = points_xy[:, 0]
        runtime.support_margin_x = float(min(com_xy[0] - np.min(xs), np.max(xs) - com_xy[0]))
        runtime.support_forward_margin_x = float(np.max(xs) - com_xy[0])
        runtime.support_margin_xy = ex18._support_polygon_margin_xy(com_xy, points_xy)
        runtime.support_center_y = float(np.mean(points_xy[:, 1]))
    else:
        runtime.support_margin_x = np.nan
        runtime.support_forward_margin_x = np.nan
        runtime.support_margin_xy = np.nan
        runtime.support_center_y = np.nan
    return samples, com_xy


def update_profile_reference_command(vbot, terrain, cmd, runtime, contact_mask, stair_mode, x_eff, y_eff, dt):
    x_eff, y_eff = ORIGINAL_UPDATE_STAIR_REFERENCE(
        vbot,
        terrain,
        cmd,
        runtime,
        contact_mask,
        stair_mode,
        x_eff,
        y_eff,
        dt,
    )
    samples, _ = _compute_support_metrics(vbot, terrain, runtime, contact_mask)
    phase = ACTIVE_PHASE_POLICY.update(vbot, contact_mask, runtime) if ACTIVE_PHASE_POLICY is not None else PhaseCommand()
    runtime.phase_command = phase
    if ACTIVE_BODY_PLANNER is not None:
        ACTIVE_BODY_PLANNER.z_cmd = float(cmd.z_pos)
        ACTIVE_BODY_PLANNER.pitch_cmd = float(cmd.pitch)
        ACTIVE_BODY_PLANNER.update(vbot, cmd, runtime, contact_mask, phase, dt)
    base = np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)
    support_h = terrain.body_support_height(base)
    foot_positions = {}
    mask = np.asarray(contact_mask, dtype=int).reshape(-1)
    for leg_i, leg in enumerate(ex18.MPC_LEG_ORDER):
        if leg_i < mask.size and int(mask[leg_i]) == 1:
            foot_positions[leg] = vbot.get_single_foot_state_in_world(leg)[0]
    runtime.support_h = support_h
    runtime.foot_support_h = terrain.feet_support_height(foot_positions)

    if stair_mode and x_eff > 0.0:
        x_eff = min(x_eff, float(phase.x_vel_limit))
        settling_legs = _runtime_settling_legs(runtime)
        if settling_legs:
            x_eff = min(x_eff, PROFILE_TOUCHDOWN_X_LIMIT)
            if any(leg in settling_legs for leg in ("FL", "FR")):
                y_eff *= PROFILE_TOUCHDOWN_Y_SCALE
    if phase.phase_name in ("front_step_up", "rear_catch_up"):
        y_eff *= float(phase.y_gain_scale)
    if str(getattr(phase, "fsm_state", "")).startswith(("rear_swing_to_step_", "rear_confirmed_step_")):
        rear_done = 0
        for leg in ("RL", "RR"):
            state = getattr(runtime, "leg_terrain_states", {}).get(leg)
            if state is not None and int(state.confirmed_step) >= int(getattr(phase, "fsm_target_step", 1)):
                rear_done += 1
        y_limit = 0.060 if rear_done == 1 else 0.040
        y_center = -0.85 * float(base[1])
        y_eff = float(np.clip(float(y_eff) + y_center, -y_limit, y_limit))

    runtime.x_eff = x_eff
    return x_eff, y_eff


def _profile_apply_body_commit_support_shift(vbot, terrain, runtime, x_eff: float, y_eff: float) -> tuple[float, float]:
    fsm_state = str(getattr(runtime, "fsm_state", ""))
    forced_phase = str(getattr(runtime, "forced_rear_phase", ""))
    forced_leg = str(getattr(runtime, "forced_rear_leg", ""))
    if (
        not fsm_state.startswith("body_commit_step_")
        or forced_phase != "support_wait"
        or forced_leg not in ("RL", "RR")
        or bool(getattr(runtime, "stair_guard", False))
    ):
        return float(x_eff), float(y_eff)
    candidate_mask = np.ones(4, dtype=int)
    candidate_mask[ex18.LEG_INDEX[forced_leg]] = 0
    try:
        samples = ex18.get_support_foot_samples(vbot, terrain, candidate_mask)
        points_xy = np.asarray([sample[1][0:2] for sample in samples], dtype=float).reshape(-1, 2)
        if len(points_xy) < 3:
            return float(x_eff), float(y_eff)
        com_xy = ex18._support_com_xy(vbot)
    except Exception:
        return float(x_eff), float(y_eff)

    target_xy = np.mean(points_xy, axis=0)
    delta = np.asarray(target_xy - com_xy, dtype=float).reshape(2)
    x_cmd = float(np.clip(0.30 * max(0.0, delta[0]), 0.0, 0.026))
    y_cmd = float(np.clip(0.45 * delta[1], -0.055, 0.055))
    runtime.profile_body_commit_shift_x = x_cmd
    runtime.profile_body_commit_shift_y = y_cmd
    return float(max(float(x_eff), x_cmd)), float(np.clip(float(y_eff) + y_cmd, -0.055, 0.055))


def _profile_apply_rear_stepup_support_shift(vbot, terrain, runtime, x_eff: float, y_eff: float) -> tuple[float, float]:
    fsm_state = str(getattr(runtime, "fsm_state", ""))
    forced_phase = str(getattr(runtime, "forced_rear_phase", ""))
    forced_leg = str(getattr(runtime, "forced_rear_leg", ""))
    if (
        not fsm_state.startswith("rear_swing_to_step_")
        or forced_phase != "support_wait"
        or forced_leg not in ("RL", "RR")
    ):
        runtime.profile_rear_stepup_shift_x = 0.0
        runtime.profile_rear_stepup_shift_y = 0.0
        return float(x_eff), float(y_eff)
    candidate_mask = np.ones(4, dtype=int)
    candidate_mask[ex18.LEG_INDEX[forced_leg]] = 0
    try:
        samples = ex18.get_support_foot_samples(vbot, terrain, candidate_mask)
        points_xy = np.asarray([sample[1][0:2] for sample in samples], dtype=float).reshape(-1, 2)
        if len(points_xy) < 3:
            return float(x_eff), float(y_eff)
        com_xy = ex18._support_com_xy(vbot)
    except Exception:
        return float(x_eff), float(y_eff)

    target_xy = np.mean(points_xy, axis=0)
    delta = np.asarray(target_xy - com_xy, dtype=float).reshape(2)
    x_cmd = float(np.clip(0.35 * max(0.0, delta[0]), 0.0, 0.020))
    y_cmd = float(np.clip(0.60 * delta[1], -0.045, 0.045))
    runtime.profile_rear_stepup_shift_x = x_cmd
    runtime.profile_rear_stepup_shift_y = y_cmd
    return float(max(float(x_eff), x_cmd)), float(np.clip(float(y_eff) + y_cmd, -0.045, 0.045))


def compute_profile_effective_velocity_command(vbot, terrain, cmd, stair_mode, stair_guard=False):
    x_eff, y_eff = ORIGINAL_COMPUTE_EFFECTIVE_VELOCITY(vbot, terrain, cmd, stair_mode, stair_guard)
    base_pos = np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)
    roll, _, _ = vbot.current_config.compute_euler_angle_world()
    near_entry = float(base_pos[0]) >= float(terrain.start_x) - PROFILE_ENTRY_DAMPING_BEFORE_X
    if near_entry and not stair_guard and x_eff > 0.0:
        y_abs = abs(float(base_pos[1]))
        roll_abs = abs(float(roll))
        x_eff = min(x_eff, PROFILE_APPROACH_X_LIMIT)
        if y_abs > 0.080 or roll_abs > 0.22:
            x_eff = min(x_eff, 0.006)
        elif y_abs > 0.050 or roll_abs > 0.16:
            x_eff = min(x_eff, 0.018)
        elif y_abs > PROFILE_ENTRY_Y_SLOW or roll_abs > 0.10:
            x_eff = min(x_eff, PROFILE_APPROACH_RISK_X_LIMIT)
    return x_eff, y_eff


def compute_profile_effective_yaw_rate_command(vbot, cmd, stair_mode):
    return ORIGINAL_COMPUTE_EFFECTIVE_YAW(vbot, cmd, stair_mode)


def compute_profile_control_tick(
    vbot: PinVBotModel,
    mujoco_vbot,
    leg_controller,
    traj,
    gait,
    stand_gait,
    mpc,
    cmd,
    terrain,
    ctrl_i: int,
    u_opt: np.ndarray,
    runtime,
):
    time_now_s = float(mujoco_vbot.data.time)

    mujoco_vbot.update_pin_with_mujoco(vbot)
    if ACTIVE_HEADLESS_DELAY_TARGET_X_VEL is not None:
        cmd.target_x_vel = 0.0 if time_now_s < ACTIVE_HEADLESS_START_DELAY_S else float(ACTIVE_HEADLESS_DELAY_TARGET_X_VEL)
        cmd.clamp()
    ex18.update_command_filter(cmd, ex18.CTRL_DT)
    base_pos = np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)
    roll, _, _ = vbot.current_config.compute_euler_angle_world()
    prev_stair_mode = runtime.stair_mode
    prev_stair_guard = runtime.stair_guard
    stair_now = ex18.compute_stair_mode_with_hysteresis(terrain, base_pos[0], prev_stair_mode)
    stair_guard = ex18.update_stair_guard(runtime, base_pos[1], roll, stair_now, cmd)
    mode_changed = (stair_now != prev_stair_mode) or (stair_guard != prev_stair_guard)
    x_eff, y_eff = compute_profile_effective_velocity_command(vbot, terrain, cmd, stair_now, stair_guard)
    yaw_rate_eff = compute_profile_effective_yaw_rate_command(vbot, cmd, stair_now)
    x_eff, y_eff = _profile_apply_entry_damping(vbot, terrain, runtime, x_eff, y_eff, stair_now, stair_guard)
    x_eff, y_eff, yaw_rate_eff = _profile_apply_guard_recovery(vbot, cmd, runtime, x_eff, y_eff, yaw_rate_eff)
    runtime.x_eff = x_eff
    runtime.yaw_rate_eff = yaw_rate_eff
    if ex18.command_is_idle(cmd):
        cmd.pitch = ex18.ramp_value(cmd.pitch, 0.0, ex18.STAIR_PITCH_RAMP, ex18.CTRL_DT)
        runtime.pitch_ref = cmd.pitch
        runtime.yaw_rate_eff = 0.0
        runtime.reset()
        _profile_reset_smoothing(runtime)
        tau_cmd = ex18.compute_stand_pd_torque(vbot, mujoco_vbot)
        return tau_cmd, u_opt, "stand_pd", np.zeros(12, dtype=float), y_eff, np.zeros(4, dtype=int)

    if stair_guard and ex18.STAIR_GUARD_USE_STAND_GAIT and not stair_now:
        runtime.walking = False
        runtime.stair_mode = stair_now
        runtime.x_eff = 0.0
        runtime.leg_outputs.clear()
        _profile_reset_smoothing(runtime)
        tau_cmd = ex18.compute_stand_pd_torque(vbot, mujoco_vbot)
        return tau_cmd, u_opt, "stair_guard", np.zeros(12, dtype=float), y_eff, np.ones(4, dtype=int)

    gait_fsm = getattr(getattr(gait, "phase_policy", None), "fsm", None)
    profile_fsm_state = str(getattr(gait_fsm, "state", ""))
    rear_event_state = profile_fsm_state.startswith(("body_commit_step_", "rear_swing_to_step_"))
    forced_rear_stepup_in_progress = bool(
        getattr(runtime, "forced_rear_active", False)
        and getattr(runtime, "forced_rear_stepup_active", False)
        and profile_fsm_state.startswith("rear_swing_to_step_")
    )
    if stair_guard and forced_rear_stepup_in_progress:
        x_eff = 0.0
        yaw_rate_eff = 0.0
        runtime.x_eff = 0.0
        runtime.yaw_rate_eff = 0.0

    prev_guard_stance_hold = bool(getattr(runtime, "profile_guard_stance_hold", False))
    guard_stance_hold = (
        _profile_guard_should_hold_stance(vbot, runtime, stair_now)
        and not forced_rear_stepup_in_progress
        and not rear_event_state
    )
    runtime.profile_guard_stance_hold = guard_stance_hold
    if guard_stance_hold != prev_guard_stance_hold:
        u_opt = np.zeros_like(u_opt)
    active_gait = (
        gait
        if rear_event_state
        else (stand_gait if guard_stance_hold else (stand_gait if stair_guard and ex18.STAIR_GUARD_USE_STAND_GAIT else gait))
    )
    if stair_guard:
        mode = "stair_guard"
    else:
        mode = "stair_crawl" if stair_now else "crawl"
    if not runtime.walking:
        _profile_start_walking_phase(runtime, time_now_s, vbot, leg_controller, traj, active_gait)
        _profile_reset_smoothing(runtime)
    elif ex18.RESET_GAIT_ON_STAIR_ENTRY and stair_now and not runtime.stair_mode:
        _profile_start_walking_phase(runtime, time_now_s, vbot, leg_controller, traj, active_gait)
        _profile_reset_smoothing(runtime)
        u_opt = np.zeros_like(u_opt)
    elif mode_changed:
        u_opt = np.zeros_like(u_opt)
    runtime.stair_mode = stair_now
    runtime.profile_vbot_for_schedule = vbot

    gait_time_s = max(0.0, time_now_s - runtime.gait_start_time_s)
    runtime.gait_time_s = gait_time_s
    if hasattr(active_gait, "update_forced_rear_schedule"):
        active_gait.update_forced_rear_schedule(gait_time_s, runtime)
        _profile_sync_forced_rear_runtime(runtime, active_gait)
    contact_mask_for_command = active_gait.compute_current_mask(gait_time_s).reshape(-1)
    runtime.profile_smoothing_active = bool(
        stair_now or base_pos[0] >= float(terrain.start_x) - PROFILE_ENTRY_DAMPING_BEFORE_X
    )
    _profile_update_contact_state(runtime, contact_mask_for_command, ex18.CTRL_DT)
    if (
        stair_now
        and ex18.STAIR_MOVE_ONLY_ALL_STANCE
        and not np.all(contact_mask_for_command == 1)
        and x_eff > ex18.STAIR_SWING_X_VEL_LIMIT
    ):
        x_eff = ex18.STAIR_SWING_X_VEL_LIMIT
        runtime.x_eff = x_eff

    x_eff, y_eff = update_profile_reference_command(
        vbot,
        terrain,
        cmd,
        runtime,
        contact_mask_for_command,
        stair_now,
        x_eff,
        y_eff,
        ex18.CTRL_DT,
    )
    x_eff, y_eff = _profile_apply_body_commit_support_shift(vbot, terrain, runtime, x_eff, y_eff)
    x_eff, y_eff = _profile_apply_rear_stepup_support_shift(vbot, terrain, runtime, x_eff, y_eff)
    if stair_guard and forced_rear_stepup_in_progress:
        x_eff = 0.0
        yaw_rate_eff = 0.0
        runtime.x_eff = 0.0
        runtime.yaw_rate_eff = 0.0
    if hasattr(active_gait, "update_forced_rear_schedule"):
        active_gait.update_forced_rear_schedule(gait_time_s, runtime)
        _profile_sync_forced_rear_runtime(runtime, active_gait)
        scheduled_mask = active_gait.compute_current_mask(gait_time_s).reshape(-1)
        if not np.array_equal(scheduled_mask, contact_mask_for_command):
            contact_mask_for_command = scheduled_mask
            _profile_update_contact_state(runtime, contact_mask_for_command, ex18.CTRL_DT)
    rear_event_locked = bool(
        str(getattr(runtime, "forced_rear_leg", "")) in ("RL", "RR")
        and (
            (
                str(getattr(runtime, "fsm_state", "")).startswith("body_commit_step_")
                and str(getattr(runtime, "forced_rear_phase", "")) in ("swing", "settle", "retry_wait")
            )
        )
    )
    if rear_event_locked:
        x_eff = 0.0
        y_eff = 0.0
        yaw_rate_eff = 0.0
        runtime.profile_prev_x_eff = 0.0
        runtime.profile_prev_y_eff = 0.0
        runtime.profile_prev_yaw_eff = 0.0
    phase = getattr(runtime, "phase_command", PhaseCommand())
    settling_legs = _runtime_settling_legs(runtime)
    start_age = _profile_start_age(runtime, time_now_s)
    if x_eff > 0.0 and start_age < PROFILE_START_SOFT_WINDOW_S:
        alpha = _smoothstep01(start_age / max(1e-6, PROFILE_START_SOFT_WINDOW_S))
        start_limit = PROFILE_START_X_LIMIT_MIN + alpha * (PROFILE_START_X_LIMIT_MAX - PROFILE_START_X_LIMIT_MIN)
        x_eff = min(float(x_eff), float(start_limit))
    if guard_stance_hold and x_eff > 0.0:
        support_margin = float(getattr(runtime, "support_margin_xy", np.nan))
        if np.isfinite(support_margin) and support_margin < 0.0:
            x_eff = min(float(x_eff), PROFILE_GUARD_RECOVERY_X_MIN)
        else:
            x_eff = min(float(x_eff), PROFILE_GUARD_RECOVERY_X_MAX)
    yaw_scale = float(getattr(phase, "yaw_gain_scale", 1.0))
    if any(leg in settling_legs for leg in ("FL", "FR")):
        yaw_scale *= PROFILE_TOUCHDOWN_YAW_SCALE
    yaw_rate_eff *= yaw_scale
    if bool(getattr(runtime, "profile_entry_damping", False)):
        yaw_rate_eff *= 0.45
    if bool(getattr(runtime, "profile_guard_recovery", False)):
        yaw_rate_eff *= PROFILE_GUARD_YAW_SCALE
    x_eff = _profile_ramp_runtime_value(runtime, "profile_prev_x_eff", x_eff, PROFILE_X_EFF_RAMP, ex18.CTRL_DT)
    y_eff = _profile_ramp_runtime_value(runtime, "profile_prev_y_eff", y_eff, PROFILE_Y_EFF_RAMP, ex18.CTRL_DT)
    yaw_rate_eff = _profile_ramp_runtime_value(
        runtime,
        "profile_prev_yaw_eff",
        yaw_rate_eff,
        PROFILE_YAW_EFF_RAMP,
        ex18.CTRL_DT,
    )
    if rear_event_locked:
        x_eff = 0.0
        y_eff = 0.0
        yaw_rate_eff = 0.0
        runtime.profile_prev_x_eff = 0.0
        runtime.profile_prev_y_eff = 0.0
        runtime.profile_prev_yaw_eff = 0.0
    runtime.yaw_rate_eff = yaw_rate_eff
    runtime.x_eff = x_eff

    if (runtime.gait_ctrl_i % ex18.STEPS_PER_MPC) == 0 or not np.any(u_opt):
        ex18.reset_com_traj_start_to_current(vbot, traj)
        z_ref_fn, pitch_ref_fn = ex18.make_terrain_mpc_reference_functions(terrain, cmd)
        traj.generate_traj(
            vbot,
            active_gait,
            gait_time_s,
            x_eff,
            y_eff,
            cmd.z_pos,
            yaw_rate_eff,
            time_step=ex18.MPC_DT,
            terrain_height_fn=terrain.height,
            pitch_des_body=cmd.pitch if ex18.STAIR_POSTURE_REFERENCE else 0.0,
            z_traj_world_fn=z_ref_fn,
            pitch_traj_world_fn=pitch_ref_fn,
        )
        try:
            sol = mpc.solve_QP(vbot, traj, False)
            n_horizon = traj.N
            w_opt = sol["x"].full().flatten()
            u_opt = w_opt[12 * n_horizon :].reshape((12, n_horizon), order="F")
            runtime.profile_qp_fail_count = int(getattr(runtime, "profile_qp_fail_count", 0))
        except RuntimeError:
            runtime.profile_qp_fail_count = int(getattr(runtime, "profile_qp_fail_count", 0)) + 1
            runtime.stair_guard = True
            runtime.x_eff = 0.0
            if not np.any(u_opt):
                u_opt = np.zeros((12, traj.N), dtype=float)

    mpc_force_raw = u_opt[:, 0]
    contact_mask = active_gait.compute_current_mask(gait_time_s).reshape(-1)
    mpc_force_now = _profile_apply_force_shaping(runtime, mpc_force_raw, contact_mask, ex18.CTRL_DT)

    tau_raw = np.zeros(12, dtype=float)
    joint_bias = ex18.compute_joint_bias(vbot) if ex18.STANCE_BIAS_COMP else None
    leg_outputs = {}
    for leg, leg_slice in ex18.LEG_SLICE.items():
        out = leg_controller.compute_leg_torque(
            leg,
            vbot,
            active_gait,
            mpc_force_now[leg_slice],
            gait_time_s,
        )
        leg_outputs[leg] = out
        tau_leg = np.asarray(out.tau, dtype=float).reshape(3)
        if ex18.STANCE_BIAS_COMP and contact_mask[ex18.LEG_INDEX[leg]] == 1:
            tau_leg = tau_leg + joint_bias[vbot.get_leg_joint_vcols(leg)]
        tau_raw[leg_slice] = tau_leg

    tau_cmd = np.clip(tau_raw, -ex18.TAU_LIM, ex18.TAU_LIM)
    tau_cmd = np.clip(
        _profile_apply_tau_rate_limit(runtime, tau_cmd, contact_mask, ex18.CTRL_DT),
        -ex18.TAU_LIM,
        ex18.TAU_LIM,
    )
    runtime.leg_outputs = leg_outputs
    runtime.swing_debug = dict(getattr(active_gait, "last_swing_debug", {}))
    runtime.gait_ctrl_i += 1
    return tau_cmd, u_opt, mode, mpc_force_now, y_eff, contact_mask


class ProfileRunLogger:
    def __init__(self):
        ex18.LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = ex18.LOG_DIR / f"{ex18.LOG_PREFIX}_{stamp}.csv"
        self.file = self.path.open("w", newline="")
        self.prev_time_s: float | None = None
        self.prev_tau = np.zeros(12, dtype=float)
        self.prev_mpc_force = np.zeros(12, dtype=float)
        self.prev_contact_mask: np.ndarray | None = None
        self.prev_phase_name = ""
        self.prev_z_cmd: float | None = None
        self.prev_pitch_cmd: float | None = None
        self.prev_x_eff: float | None = None
        self.prev_yaw_eff: float | None = None
        base_fields = [
            "time_s",
            "phase_name",
            "fsm_state",
            "fsm_target_step",
            "fsm_hold_reason",
            "phase_transition",
            "stair_mode",
            "stair_guard",
            "contact_mask",
            "x_vel_eff",
            "dx_vel_eff",
            "y_vel_eff",
            "yaw_rate_eff",
            "dyaw_rate_eff",
            "z_cmd",
            "dz_cmd",
            "pitch_cmd",
            "dpitch_cmd",
            "support_h",
            "foot_support_h",
            "support_margin_x",
            "support_margin_xy",
            "support_forward_margin_x",
            "support_center_y",
            "rear_stepup_shift_x",
            "rear_stepup_shift_y",
            "forced_rear_leg",
            "forced_rear_active",
            "forced_rear_stepup_active",
            "forced_rear_phase",
            "forced_rear_start_s",
            "forced_rear_end_s",
            "forced_rear_retry_after_s",
            "base_x",
            "base_y",
            "base_z",
            "roll",
            "pitch",
            "yaw",
            "base_vx",
            "base_vy",
            "base_vz",
            "base_wx",
            "base_wy",
            "base_wz",
            "mode",
            "target_x_vel",
            "x_vel",
            "tau_max",
            "tau_sat_frac",
            "tau_rate_limited",
            "force_rate_limited",
            "qp_fail_count",
            "mpc_update_ms",
            "mpc_solve_ms",
        ]
        leg_fields = []
        for leg in ex18.MPC_LEG_ORDER:
            p = leg.lower()
            leg_fields.extend(
                [
                    f"{p}_contact_cmd",
                    f"{p}_touchdown_event",
                    f"{p}_liftoff_event",
                    f"{p}_touchdown_age",
                    f"{p}_stance_blend",
                    f"{p}_force_blend",
                    f"{p}_mpc_fx",
                    f"{p}_mpc_fy",
                    f"{p}_mpc_fz",
                    f"{p}_mpc_force_delta",
                    f"{p}_foot_nom_x",
                    f"{p}_foot_nom_z",
                    f"{p}_foot_des_x",
                    f"{p}_foot_des_y",
                    f"{p}_foot_des_z",
                    f"{p}_foot_now_x",
                    f"{p}_foot_now_y",
                    f"{p}_foot_now_z",
                    f"{p}_foot_vx",
                    f"{p}_foot_vy",
                    f"{p}_foot_vz",
                    f"{p}_foot_err",
                    f"{p}_foot_err_x",
                    f"{p}_foot_err_y",
                    f"{p}_foot_err_z",
                    f"{p}_terrain_clearance",
                    f"{p}_takeoff_step",
                    f"{p}_touchdown_step",
                    f"{p}_target_step",
                    f"{p}_foothold_projected",
                    f"{p}_projection_reason",
                    f"{p}_projection_delta_x",
                    f"{p}_projection_delta_y",
                    f"{p}_terrain_peak",
                    f"{p}_swing_apex_z",
                    f"{p}_swing_clearance",
                    f"{p}_riser_phase",
                    f"{p}_peak_phase",
                    f"{p}_terminal_vz_limit",
                    f"{p}_observed_step_index",
                    f"{p}_confirmed_step",
                    f"{p}_rear_prepare_step",
                    f"{p}_contact_confidence",
                    f"{p}_on_safe_tread",
                    f"{p}_near_edge",
                    f"{p}_edge_distance",
                    f"{p}_touchdown_confirmed",
                    f"{p}_target_step_source",
                    f"{p}_is_forced_rear",
                    f"{p}_candidate_score",
                    f"{p}_selected_candidate_score",
                    f"{p}_replan_reason",
                ]
            )
            if PROFILE_DEBUG_JOINTS:
                for joint in JOINT_SHORT_NAMES:
                    leg_fields.extend(
                        [
                            f"{p}_{joint}_q",
                            f"{p}_{joint}_dq",
                            f"{p}_{joint}_tau",
                            f"{p}_{joint}_tau_rate",
                            f"{p}_{joint}_tau_sat",
                        ]
                    )
        self.writer = csv.DictWriter(self.file, fieldnames=base_fields + leg_fields)
        self.writer.writeheader()

    def log(self, time_s, vbot, cmd, terrain, tau_cmd, mpc, mode, mpc_force_now=None, y_eff=0.0, runtime=None, contact_mask=None):
        base = np.asarray(vbot.current_config.base_pos, dtype=float).reshape(3)
        base_vel = np.asarray(vbot.current_config.base_vel, dtype=float).reshape(3)
        base_ang_vel = np.asarray(vbot.current_config.base_ang_vel, dtype=float).reshape(3)
        roll, pitch, yaw = vbot.current_config.compute_euler_angle_world()
        tau = np.asarray(tau_cmd, dtype=float).reshape(12)
        tau_abs = np.abs(tau)
        mpc_force = np.zeros(12, dtype=float)
        if mpc_force_now is not None:
            mpc_force = np.asarray(mpc_force_now, dtype=float).reshape(12)
        support_h = float(getattr(runtime, "support_h", terrain.body_support_height(base)))
        foot_support_h = float(getattr(runtime, "foot_support_h", 0.0))
        phase_name = getattr(runtime, "phase_name", "")
        if not phase_name:
            stairs_end_x = float(terrain.start_x) + int(terrain.step_count) * float(terrain.step_depth)
            if base[0] >= stairs_end_x + 0.15:
                phase_name = "landing"
            elif bool(getattr(runtime, "stair_mode", False)):
                phase_name = "body_commit"
            else:
                phase_name = "approach"
        mask = np.asarray(contact_mask if contact_mask is not None else [0, 0, 0, 0], dtype=int).reshape(4)
        dt_log = 0.0
        if self.prev_time_s is not None:
            dt_log = max(1e-9, float(time_s) - float(self.prev_time_s))
        tau_rate = (tau - self.prev_tau) / dt_log if dt_log > 0.0 else np.zeros_like(tau)
        mpc_force_delta = mpc_force - self.prev_mpc_force if dt_log > 0.0 else np.zeros_like(mpc_force)
        prev_mask = self.prev_contact_mask if self.prev_contact_mask is not None else mask.copy()
        x_eff_now = float(getattr(runtime, "x_eff", cmd.x_vel))
        yaw_eff_now = float(getattr(runtime, "yaw_rate_eff", cmd.yaw_rate))
        dz_cmd = 0.0 if self.prev_z_cmd is None else float(cmd.z_pos) - self.prev_z_cmd
        dpitch_cmd = 0.0 if self.prev_pitch_cmd is None else float(cmd.pitch) - self.prev_pitch_cmd
        dx_eff = 0.0 if self.prev_x_eff is None else x_eff_now - self.prev_x_eff
        dyaw_eff = 0.0 if self.prev_yaw_eff is None else yaw_eff_now - self.prev_yaw_eff
        phase_transition = int(bool(self.prev_phase_name) and phase_name != self.prev_phase_name)
        row = {
            "time_s": f"{float(time_s):.4f}",
            "phase_name": phase_name or "approach",
            "fsm_state": str(getattr(runtime, "fsm_state", phase_name or "approach")),
            "fsm_target_step": int(getattr(runtime, "fsm_target_step", 0)),
            "fsm_hold_reason": str(getattr(runtime, "fsm_hold_reason", "none")),
            "phase_transition": phase_transition,
            "stair_mode": int(bool(getattr(runtime, "stair_mode", False))),
            "stair_guard": int(bool(getattr(runtime, "stair_guard", False))),
            "contact_mask": "".join(str(int(v)) for v in mask),
            "x_vel_eff": f"{x_eff_now:.4f}",
            "dx_vel_eff": f"{dx_eff:.4f}",
            "y_vel_eff": f"{float(y_eff):.4f}",
            "yaw_rate_eff": f"{yaw_eff_now:.4f}",
            "dyaw_rate_eff": f"{dyaw_eff:.4f}",
            "z_cmd": f"{float(cmd.z_pos):.4f}",
            "dz_cmd": f"{dz_cmd:.4f}",
            "pitch_cmd": f"{float(cmd.pitch):.4f}",
            "dpitch_cmd": f"{dpitch_cmd:.4f}",
            "support_h": f"{support_h:.4f}",
            "foot_support_h": f"{foot_support_h:.4f}",
            "support_margin_x": f"{float(getattr(runtime, 'support_margin_x', np.nan)):.4f}",
            "support_margin_xy": f"{float(getattr(runtime, 'support_margin_xy', np.nan)):.4f}",
            "support_forward_margin_x": f"{float(getattr(runtime, 'support_forward_margin_x', np.nan)):.4f}",
            "support_center_y": f"{float(getattr(runtime, 'support_center_y', np.nan)):.4f}",
            "rear_stepup_shift_x": f"{float(getattr(runtime, 'profile_rear_stepup_shift_x', 0.0)):.4f}",
            "rear_stepup_shift_y": f"{float(getattr(runtime, 'profile_rear_stepup_shift_y', 0.0)):.4f}",
            "forced_rear_leg": str(getattr(runtime, "forced_rear_leg", "")),
            "forced_rear_active": int(bool(getattr(runtime, "forced_rear_active", False))),
            "forced_rear_stepup_active": int(bool(getattr(runtime, "forced_rear_stepup_active", False))),
            "forced_rear_phase": str(getattr(runtime, "forced_rear_phase", "idle")),
            "forced_rear_start_s": f"{float(getattr(runtime, 'forced_rear_start_s', np.nan)):.4f}",
            "forced_rear_end_s": f"{float(getattr(runtime, 'forced_rear_end_s', np.nan)):.4f}",
            "forced_rear_retry_after_s": f"{float(getattr(runtime, 'forced_rear_retry_after_s', np.nan)):.4f}",
            "base_x": f"{base[0]:.4f}",
            "base_y": f"{base[1]:.4f}",
            "base_z": f"{base[2]:.4f}",
            "roll": f"{roll:.4f}",
            "pitch": f"{pitch:.4f}",
            "yaw": f"{yaw:.4f}",
            "base_vx": f"{base_vel[0]:.4f}",
            "base_vy": f"{base_vel[1]:.4f}",
            "base_vz": f"{base_vel[2]:.4f}",
            "base_wx": f"{base_ang_vel[0]:.4f}",
            "base_wy": f"{base_ang_vel[1]:.4f}",
            "base_wz": f"{base_ang_vel[2]:.4f}",
            "mode": mode,
            "target_x_vel": f"{float(cmd.target_x_vel):.4f}",
            "x_vel": f"{float(cmd.x_vel):.4f}",
            "tau_max": f"{float(np.max(tau_abs)):.4f}",
            "tau_sat_frac": f"{float(np.mean(tau_abs >= 0.98 * ex18.TAU_LIM)):.4f}",
            "tau_rate_limited": int(bool(getattr(runtime, "profile_tau_rate_limited", False))),
            "force_rate_limited": int(bool(getattr(runtime, "profile_force_rate_limited", False))),
            "qp_fail_count": int(getattr(runtime, "profile_qp_fail_count", 0)),
            "mpc_update_ms": f"{float(getattr(mpc, 'update_time', 0.0)):.4f}",
            "mpc_solve_ms": f"{float(getattr(mpc, 'solve_time', 0.0)):.4f}",
        }
        for leg_i, leg in enumerate(ex18.MPC_LEG_ORDER):
            p = leg.lower()
            leg_slice = ex18.LEG_SLICE[leg]
            foot_pos, foot_vel = vbot.get_single_foot_state_in_world(leg)
            foot_pos = np.asarray(foot_pos, dtype=float).reshape(3)
            foot_vel = np.asarray(foot_vel, dtype=float).reshape(3)
            contact_now = int(mask[leg_i])
            contact_prev = int(prev_mask[leg_i]) if leg_i < len(prev_mask) else contact_now
            row[f"{p}_contact_cmd"] = contact_now
            row[f"{p}_touchdown_event"] = int(contact_prev == 0 and contact_now == 1)
            row[f"{p}_liftoff_event"] = int(contact_prev == 1 and contact_now == 0)
            touchdown_age = getattr(runtime, "profile_touchdown_age", {}).get(leg, np.nan) if runtime is not None else np.nan
            row[f"{p}_touchdown_age"] = f"{float(touchdown_age):.4f}" if np.isfinite(touchdown_age) else "nan"
            row[f"{p}_stance_blend"] = f"{float(getattr(runtime, 'profile_stance_blend', {}).get(leg, contact_now)):.4f}"
            row[f"{p}_force_blend"] = f"{float(getattr(runtime, 'profile_force_blend', {}).get(leg, contact_now)):.4f}"
            row[f"{p}_mpc_fx"] = f"{float(mpc_force[leg_slice][0]):.4f}"
            row[f"{p}_mpc_fy"] = f"{float(mpc_force[leg_slice][1]):.4f}"
            row[f"{p}_mpc_fz"] = f"{float(mpc_force[leg_slice][2]):.4f}"
            row[f"{p}_mpc_force_delta"] = f"{float(np.linalg.norm(mpc_force_delta[leg_slice])):.4f}"
            out = getattr(runtime, "leg_outputs", {}).get(leg) if runtime is not None else None
            if out is not None:
                des = np.asarray(out.pos_des, dtype=float).reshape(3)
                now = np.asarray(out.pos_now, dtype=float).reshape(3)
                foot_err = des - now
                row[f"{p}_foot_des_x"] = f"{des[0]:.4f}"
                row[f"{p}_foot_des_y"] = f"{des[1]:.4f}"
                row[f"{p}_foot_des_z"] = f"{des[2]:.4f}"
                row[f"{p}_foot_now_x"] = f"{now[0]:.4f}"
                row[f"{p}_foot_now_y"] = f"{now[1]:.4f}"
                row[f"{p}_foot_now_z"] = f"{now[2]:.4f}"
                row[f"{p}_foot_err"] = f"{float(np.linalg.norm(foot_err)):.4f}"
                row[f"{p}_foot_err_x"] = f"{foot_err[0]:.4f}"
                row[f"{p}_foot_err_y"] = f"{foot_err[1]:.4f}"
                row[f"{p}_foot_err_z"] = f"{foot_err[2]:.4f}"
            else:
                for name in (
                    "foot_des_x",
                    "foot_des_y",
                    "foot_des_z",
                    "foot_now_x",
                    "foot_now_y",
                    "foot_now_z",
                    "foot_err",
                    "foot_err_x",
                    "foot_err_y",
                    "foot_err_z",
                ):
                    row[f"{p}_{name}"] = "nan"
            row[f"{p}_foot_vx"] = f"{foot_vel[0]:.4f}"
            row[f"{p}_foot_vy"] = f"{foot_vel[1]:.4f}"
            row[f"{p}_foot_vz"] = f"{foot_vel[2]:.4f}"
            row[f"{p}_terrain_clearance"] = f"{float(foot_pos[2] - terrain.height(foot_pos[0], foot_pos[1])):.4f}"
            dbg = getattr(runtime, "swing_debug", {}).get(leg, {}) if runtime is not None else {}
            row[f"{p}_foot_nom_x"] = f"{float(dbg.get('foot_nom_x', np.nan)):.4f}"
            row[f"{p}_foot_nom_z"] = f"{float(dbg.get('foot_nom_z', np.nan)):.4f}"
            row[f"{p}_takeoff_step"] = str(int(dbg.get("takeoff_step", -1)))
            row[f"{p}_touchdown_step"] = str(int(dbg.get("touchdown_step", -1)))
            row[f"{p}_target_step"] = str(int(dbg.get("target_step", -1)))
            row[f"{p}_foothold_projected"] = int(bool(dbg.get("foothold_projected", False)))
            row[f"{p}_projection_reason"] = str(dbg.get("projection_reason", "none"))
            row[f"{p}_projection_delta_x"] = f"{float(dbg.get('projection_delta_x', np.nan)):.4f}"
            row[f"{p}_projection_delta_y"] = f"{float(dbg.get('projection_delta_y', np.nan)):.4f}"
            row[f"{p}_terrain_peak"] = f"{float(dbg.get('terrain_peak', np.nan)):.4f}"
            row[f"{p}_swing_apex_z"] = f"{float(dbg.get('swing_apex_z', np.nan)):.4f}"
            row[f"{p}_swing_clearance"] = f"{float(dbg.get('swing_clearance', np.nan)):.4f}"
            row[f"{p}_riser_phase"] = f"{float(dbg.get('riser_phase', np.nan)):.4f}"
            row[f"{p}_peak_phase"] = f"{float(dbg.get('peak_phase', np.nan)):.4f}"
            row[f"{p}_terminal_vz_limit"] = f"{float(dbg.get('terminal_vz_limit', np.nan)):.4f}"
            leg_state = getattr(runtime, "leg_terrain_states", {}).get(leg) if runtime is not None else None
            if leg_state is not None:
                row[f"{p}_observed_step_index"] = int(leg_state.observed_step_index)
                row[f"{p}_confirmed_step"] = int(leg_state.confirmed_step)
                row[f"{p}_rear_prepare_step"] = int(getattr(leg_state, "rear_prepare_step", 0))
                row[f"{p}_contact_confidence"] = f"{float(leg_state.contact_confidence):.4f}"
                row[f"{p}_on_safe_tread"] = int(bool(leg_state.on_safe_tread))
                row[f"{p}_near_edge"] = int(bool(leg_state.near_edge))
                row[f"{p}_edge_distance"] = f"{float(leg_state.edge_distance):.4f}"
                row[f"{p}_touchdown_confirmed"] = int(bool(leg_state.touchdown_confirmed))
            else:
                row[f"{p}_observed_step_index"] = -1
                row[f"{p}_confirmed_step"] = -1
                row[f"{p}_rear_prepare_step"] = -1
                row[f"{p}_contact_confidence"] = "nan"
                row[f"{p}_on_safe_tread"] = 0
                row[f"{p}_near_edge"] = 0
                row[f"{p}_edge_distance"] = "nan"
                row[f"{p}_touchdown_confirmed"] = 0
            row[f"{p}_target_step_source"] = str(dbg.get("target_step_source", "unknown"))
            row[f"{p}_is_forced_rear"] = int(str(getattr(runtime, "forced_rear_leg", "")) == leg)
            row[f"{p}_candidate_score"] = f"{float(dbg.get('candidate_score', np.nan)):.4f}"
            row[f"{p}_selected_candidate_score"] = f"{float(dbg.get('selected_candidate_score', np.nan)):.4f}"
            row[f"{p}_replan_reason"] = str(dbg.get("replan_reason", "none"))
            if PROFILE_DEBUG_JOINTS:
                q_leg = np.asarray(getattr(vbot.current_config, f"{leg}_joint_angle"), dtype=float).reshape(3)
                dq_leg = np.asarray(getattr(vbot.current_config, f"{leg}_joint_vel"), dtype=float).reshape(3)
                tau_leg = tau[leg_slice]
                tau_rate_leg = tau_rate[leg_slice]
                tau_lim_leg = np.asarray(ex18.TAU_LIM[leg_slice], dtype=float).reshape(3)
                for joint_i, joint in enumerate(JOINT_SHORT_NAMES):
                    row[f"{p}_{joint}_q"] = f"{float(q_leg[joint_i]):.4f}"
                    row[f"{p}_{joint}_dq"] = f"{float(dq_leg[joint_i]):.4f}"
                    row[f"{p}_{joint}_tau"] = f"{float(tau_leg[joint_i]):.4f}"
                    row[f"{p}_{joint}_tau_rate"] = f"{float(tau_rate_leg[joint_i]):.4f}"
                    row[f"{p}_{joint}_tau_sat"] = int(abs(float(tau_leg[joint_i])) >= 0.98 * float(tau_lim_leg[joint_i]))
        self.writer.writerow(row)
        self.file.flush()
        self.prev_time_s = float(time_s)
        self.prev_tau = tau.copy()
        self.prev_mpc_force = mpc_force.copy()
        self.prev_contact_mask = mask.copy()
        self.prev_phase_name = phase_name
        self.prev_z_cmd = float(cmd.z_pos)
        self.prev_pitch_cmd = float(cmd.pitch)
        self.prev_x_eff = x_eff_now
        self.prev_yaw_eff = yaw_eff_now

    def close(self):
        self.file.close()


ORIGINAL_COMPUTE_EFFECTIVE_VELOCITY = ex18.compute_effective_velocity_command
ORIGINAL_COMPUTE_EFFECTIVE_YAW = ex18.compute_effective_yaw_rate_command
ORIGINAL_UPDATE_STAIR_REFERENCE = ex18.update_stair_reference_command
ORIGINAL_STAIR_AWARE_GAIT = ex18.StairAwareGait
ORIGINAL_RUN_LOGGER = ex18.RunLogger
ORIGINAL_COMPUTE_CONTROL_TICK = ex18.compute_control_tick


def augment_profile_log(path: Path, profile_name: str):
    path = Path(path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
        base_fields = list(rows[0].keys()) if rows else []

    extra_fields = ["phase_name", "fsm_state", "fsm_target_step", "fsm_hold_reason"]
    for leg in ex18.MPC_LEG_ORDER:
        p = leg.lower()
        extra_fields.extend(
            [
                f"{p}_foot_nom_x",
                f"{p}_foot_nom_z",
                f"{p}_target_step",
                f"{p}_projection_reason",
                f"{p}_projection_delta_x",
                f"{p}_projection_delta_y",
                f"{p}_terrain_peak",
                f"{p}_riser_phase",
                f"{p}_peak_phase",
                f"{p}_touchdown_age",
                f"{p}_stance_blend",
                f"{p}_force_blend",
                f"{p}_observed_step_index",
                f"{p}_confirmed_step",
                f"{p}_contact_confidence",
                f"{p}_on_safe_tread",
                f"{p}_near_edge",
                f"{p}_edge_distance",
                f"{p}_touchdown_confirmed",
                f"{p}_target_step_source",
                f"{p}_candidate_score",
                f"{p}_selected_candidate_score",
                f"{p}_replan_reason",
            ]
        )
    fieldnames = base_fields + [name for name in extra_fields if name not in base_fields]

    terrain = Uniform06StairProfile() if profile_name == "uniform06" else VariableStairProfile()
    stairs_end_x = terrain.start_x + terrain.step_count * terrain.step_depth
    for row in rows:
        base_x = float(row.get("base_x", "0") or 0.0)
        if "phase_name" not in base_fields:
            if base_x >= stairs_end_x + 0.15:
                row["phase_name"] = "landing"
            elif int(float(row.get("stair_mode", 0) or 0)):
                row["phase_name"] = "body_commit"
            else:
                row["phase_name"] = "approach"
        row.setdefault("fsm_state", row.get("phase_name", "approach"))
        row.setdefault("fsm_target_step", "0")
        row.setdefault("fsm_hold_reason", "legacy_log")
        for leg in ex18.MPC_LEG_ORDER:
            p = leg.lower()
            row.setdefault(f"{p}_foot_nom_x", row.get(f"{p}_foot_des_x", "nan"))
            row.setdefault(f"{p}_foot_nom_z", row.get(f"{p}_foot_des_z", "nan"))
            row.setdefault(f"{p}_target_step", row.get(f"{p}_touchdown_step", "-1"))
            projected = str(row.get(f"{p}_foothold_projected", "0")) not in ("0", "False", "false", "")
            row.setdefault(f"{p}_projection_reason", "profile_safe_tread" if projected else "none")
            row.setdefault(f"{p}_projection_delta_x", "nan")
            row.setdefault(f"{p}_projection_delta_y", "nan")
            row.setdefault(f"{p}_terrain_peak", "nan")
            row.setdefault(f"{p}_riser_phase", "nan")
            row.setdefault(f"{p}_peak_phase", "nan")
            row.setdefault(f"{p}_touchdown_age", "nan")
            row.setdefault(f"{p}_stance_blend", row.get(f"{p}_contact_cmd", "0"))
            row.setdefault(f"{p}_force_blend", row.get(f"{p}_contact_cmd", "0"))
            row.setdefault(f"{p}_observed_step_index", row.get(f"{p}_touchdown_step", "-1"))
            row.setdefault(f"{p}_confirmed_step", row.get(f"{p}_touchdown_step", "-1"))
            row.setdefault(f"{p}_contact_confidence", "nan")
            row.setdefault(f"{p}_on_safe_tread", "0")
            row.setdefault(f"{p}_near_edge", "0")
            row.setdefault(f"{p}_edge_distance", "nan")
            row.setdefault(f"{p}_touchdown_confirmed", "0")
            row.setdefault(f"{p}_target_step_source", "legacy")
            row.setdefault(f"{p}_candidate_score", "nan")
            row.setdefault(f"{p}_selected_candidate_score", "nan")
            row.setdefault(f"{p}_replan_reason", "none")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_float(row, key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _max_abs(rows, key: str) -> tuple[float, int]:
    best = 0.0
    best_i = -1
    for i, row in enumerate(rows):
        value = _csv_float(row, key)
        if np.isfinite(value) and abs(value) > abs(best):
            best = value
            best_i = i
    return best, best_i


def analyze_profile_log(path: Path, window_s: float = 0.5) -> dict:
    path = Path(path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"path": str(path), "rows": 0}

    times = [_csv_float(row, "time_s", 0.0) for row in rows]
    phase_counts = Counter(row.get("phase_name", "") for row in rows)
    fsm_counts = Counter(row.get("fsm_state", row.get("phase_name", "")) for row in rows)
    fsm_hold_counts = Counter(row.get("fsm_hold_reason", "none") or "none" for row in rows)
    landing_times = [
        times[i]
        for i, row in enumerate(rows)
        if row.get("fsm_state") == "landing_confirmed" or row.get("phase_name") == "landing"
    ]
    guard_rows = sum(1 for row in rows if str(row.get("stair_guard", "")).lower() in ("1", "true", "yes"))
    zero_x_rows = sum(1 for row in rows if abs(_csv_float(row, "x_vel_eff", 0.0)) < 1e-5)
    projection_reasons = Counter()
    replan_reasons = Counter()
    unsafe_foothold_rows = 0
    edge_foothold_rows = 0
    confirmed_progress = {}
    projection_delta_peak = 0.0
    for row in rows:
        for leg in ex18.MPC_LEG_ORDER:
            p = leg.lower()
            reason = row.get(f"{p}_projection_reason", "none") or "none"
            if reason != "none":
                projection_reasons[reason] += 1
            replan = row.get(f"{p}_replan_reason", "none") or "none"
            if replan != "none":
                replan_reasons[replan] += 1
            if int(_csv_float(row, f"{p}_on_safe_tread", 1.0)) == 0 and _csv_float(row, f"{p}_observed_step_index", 0.0) > 0:
                unsafe_foothold_rows += 1
            if int(_csv_float(row, f"{p}_near_edge", 0.0)) == 1:
                edge_foothold_rows += 1
            dx = abs(_csv_float(row, f"{p}_projection_delta_x", 0.0))
            dy = abs(_csv_float(row, f"{p}_projection_delta_y", 0.0))
            projection_delta_peak = max(projection_delta_peak, dx, dy)
    for leg in ex18.MPC_LEG_ORDER:
        p = leg.lower()
        confirmed_values = [_csv_float(row, f"{p}_confirmed_step", -1.0) for row in rows]
        confirmed_progress[leg] = int(max(confirmed_values)) if confirmed_values else -1
    tau_rate_limited_rows = sum(1 for row in rows if int(_csv_float(row, "tau_rate_limited", 0.0)) == 1)
    force_rate_limited_rows = sum(1 for row in rows if int(_csv_float(row, "force_rate_limited", 0.0)) == 1)
    qp_fail_rows = sum(1 for row in rows if int(_csv_float(row, "qp_fail_count", 0.0)) > 0)

    hip_report = {}
    worst_hip_rate = {"leg": "", "value": 0.0, "time_s": 0.0, "phase": ""}
    touchdown_windows = []
    for leg in ex18.MPC_LEG_ORDER:
        p = leg.lower()
        hip_dq, hip_dq_i = _max_abs(rows, f"{p}_hip_dq")
        hip_tau, hip_tau_i = _max_abs(rows, f"{p}_hip_tau")
        hip_tau_rate, hip_tau_rate_i = _max_abs(rows, f"{p}_hip_tau_rate")
        hip_sat_rows = sum(1 for row in rows if int(_csv_float(row, f"{p}_hip_tau_sat", 0.0)) == 1)
        hip_report[leg] = {
            "hip_dq_peak": hip_dq,
            "hip_dq_time_s": times[hip_dq_i] if hip_dq_i >= 0 else float("nan"),
            "hip_tau_peak": hip_tau,
            "hip_tau_time_s": times[hip_tau_i] if hip_tau_i >= 0 else float("nan"),
            "hip_tau_rate_peak": hip_tau_rate,
            "hip_tau_rate_time_s": times[hip_tau_rate_i] if hip_tau_rate_i >= 0 else float("nan"),
            "hip_tau_sat_rows": hip_sat_rows,
        }
        if abs(hip_tau_rate) > abs(worst_hip_rate["value"]):
            worst_hip_rate = {
                "leg": leg,
                "value": hip_tau_rate,
                "time_s": times[hip_tau_rate_i] if hip_tau_rate_i >= 0 else float("nan"),
                "phase": rows[hip_tau_rate_i].get("phase_name", "") if hip_tau_rate_i >= 0 else "",
            }
        for i, row in enumerate(rows):
            if int(_csv_float(row, f"{p}_touchdown_event", 0.0)) != 1:
                continue
            t0 = times[i]
            win = [j for j, t in enumerate(times) if abs(t - t0) <= window_s]
            if not win:
                continue
            max_hip_rate = max(abs(_csv_float(rows[j], f"{p}_hip_tau_rate", 0.0)) for j in win)
            min_clearance = min(_csv_float(rows[j], f"{p}_terrain_clearance", float("inf")) for j in win)
            max_down_vz = min(_csv_float(rows[j], f"{p}_foot_vz", 0.0) for j in win)
            max_mpc_delta = max(_csv_float(rows[j], f"{p}_mpc_force_delta", 0.0) for j in win)
            max_foot_err = max(_csv_float(rows[j], f"{p}_foot_err", 0.0) for j in win)
            min_stance_blend = min(_csv_float(rows[j], f"{p}_stance_blend", 1.0) for j in win)
            min_force_blend = min(_csv_float(rows[j], f"{p}_force_blend", 1.0) for j in win)
            touchdown_windows.append(
                {
                    "leg": leg,
                    "time_s": t0,
                    "phase": row.get("phase_name", ""),
                    "takeoff_step": row.get(f"{p}_takeoff_step", ""),
                    "touchdown_step": row.get(f"{p}_touchdown_step", ""),
                    "hip_tau_rate_peak": max_hip_rate,
                    "terrain_clearance_min": min_clearance,
                    "foot_vz_min": max_down_vz,
                    "mpc_force_delta_peak": max_mpc_delta,
                    "foot_err_peak": max_foot_err,
                    "stance_blend_min": min_stance_blend,
                    "force_blend_min": min_force_blend,
                }
            )
    touchdown_windows.sort(key=lambda item: item["hip_tau_rate_peak"], reverse=True)

    dz_peak, dz_i = _max_abs(rows, "dz_cmd")
    dpitch_peak, dpitch_i = _max_abs(rows, "dpitch_cmd")
    dx_peak, dx_i = _max_abs(rows, "dx_vel_eff")
    dyaw_peak, dyaw_i = _max_abs(rows, "dyaw_rate_eff")
    roll_peak, _ = _max_abs(rows, "roll")
    pitch_peak, _ = _max_abs(rows, "pitch")
    yaw_peak, _ = _max_abs(rows, "yaw")
    final_z = _csv_float(rows[-1], "base_z")
    fell = None
    if final_z < 0.12:
        fell = f"base_z_low:{final_z:.3f}"
    elif abs(roll_peak) > 1.20:
        fell = f"roll_large:{roll_peak:.3f}"
    elif abs(pitch_peak) > 1.20:
        fell = f"pitch_large:{pitch_peak:.3f}"

    cause_tags = []
    if not landing_times:
        last_confirmed = min(confirmed_progress.values()) if confirmed_progress else -1
        if last_confirmed < 1:
            cause_tags.append("touchdown_not_confirmed")
        elif confirmed_progress.get("RL", 0) < confirmed_progress.get("FL", 0) or confirmed_progress.get("RR", 0) < confirmed_progress.get("FR", 0):
            cause_tags.append("rear_not_caught_up")
    if unsafe_foothold_rows > 0:
        cause_tags.append("unsafe_foothold")
    if edge_foothold_rows > 0:
        cause_tags.append("edge_contact")
    if any("support_margin" in key or "body_commit" in key for key in fsm_hold_counts):
        cause_tags.append("body_commit_too_early")
    if any(info["hip_tau_sat_rows"] > 0 for info in hip_report.values()):
        cause_tags.append("hip_tau_saturation")
    if touchdown_windows:
        worst_td = touchdown_windows[0]
        if worst_td["foot_vz_min"] < -0.35 or worst_td["terrain_clearance_min"] < -0.005:
            cause_tags.append("touchdown_impact")
        if worst_td["mpc_force_delta_peak"] > 35.0:
            cause_tags.append("mpc_force_jump")
        if worst_td["foot_err_peak"] > 0.08:
            cause_tags.append("foot_tracking_error")
    if abs(dz_peak) > 0.015 or abs(dpitch_peak) > 0.015:
        cause_tags.append("body_ref_jump")
    if abs(dx_peak) > 0.02 or abs(dyaw_peak) > 0.03:
        cause_tags.append("phase_policy_jump")
    if projection_delta_peak > 0.055:
        cause_tags.append("foothold_projection_jump")
    elif projection_reasons:
        cause_tags.append("foothold_projection_active")
    if abs(roll_peak) > 0.35 and abs(yaw_peak) > 0.18:
        cause_tags.append("yaw_roll_coupling")
    if qp_fail_rows > 0:
        cause_tags.append("mpc_qp_failure")
    if not cause_tags:
        cause_tags.append("no_single_large_log_signature")

    return {
        "path": str(path),
        "rows": len(rows),
        "landing_time_s": min(landing_times) if landing_times else None,
        "fell": fell,
        "final_base_x": _csv_float(rows[-1], "base_x"),
        "final_base_y": _csv_float(rows[-1], "base_y"),
        "final_base_z": final_z,
        "guard_rows": guard_rows,
        "zero_x_rows": zero_x_rows,
        "tau_rate_limited_rows": tau_rate_limited_rows,
        "force_rate_limited_rows": force_rate_limited_rows,
        "qp_fail_rows": qp_fail_rows,
        "phase_counts": dict(phase_counts),
        "fsm_counts": dict(fsm_counts),
        "fsm_hold_reason_counts": dict(fsm_hold_counts),
        "confirmed_step_progress": confirmed_progress,
        "unsafe_foothold_rows": unsafe_foothold_rows,
        "edge_foothold_rows": edge_foothold_rows,
        "projection_reasons": dict(projection_reasons),
        "replan_reasons": dict(replan_reasons),
        "projection_delta_peak": projection_delta_peak,
        "attitude_peaks": {"roll": roll_peak, "pitch": pitch_peak, "yaw": yaw_peak},
        "command_jumps": {
            "dz_cmd_peak": dz_peak,
            "dz_cmd_time_s": times[dz_i] if dz_i >= 0 else float("nan"),
            "dpitch_cmd_peak": dpitch_peak,
            "dpitch_cmd_time_s": times[dpitch_i] if dpitch_i >= 0 else float("nan"),
            "dx_vel_eff_peak": dx_peak,
            "dx_vel_eff_time_s": times[dx_i] if dx_i >= 0 else float("nan"),
            "dyaw_rate_eff_peak": dyaw_peak,
            "dyaw_rate_eff_time_s": times[dyaw_i] if dyaw_i >= 0 else float("nan"),
        },
        "hip_report": hip_report,
        "worst_hip_tau_rate": worst_hip_rate,
        "worst_touchdown_windows": touchdown_windows[:8],
        "cause_tags": cause_tags,
    }


def print_profile_log_analysis(path: Path):
    report = analyze_profile_log(path)
    print("\nEX21 diagnostic analysis")
    print(f"Log: {report.get('path')}")
    print(
        "Summary: "
        f"rows={report.get('rows')} landing_t={report.get('landing_time_s')} "
        f"fell={report.get('fell')} "
        f"final_x={report.get('final_base_x'):.4f} final_y={report.get('final_base_y'):.4f} "
        f"final_z={report.get('final_base_z'):.4f} "
        f"guard_rows={report.get('guard_rows')} zero_x_rows={report.get('zero_x_rows')}"
    )
    print(
        "Limiter rows: "
        f"tau_rate={report.get('tau_rate_limited_rows')} "
        f"force_rate={report.get('force_rate_limited_rows')} "
        f"qp_fail={report.get('qp_fail_rows')}"
    )
    print(f"Attitude peaks: {report.get('attitude_peaks')}")
    print(f"Phase distribution: {report.get('phase_counts')}")
    print(f"FSM distribution: {report.get('fsm_counts')}")
    print(f"FSM hold reasons: {report.get('fsm_hold_reason_counts')}")
    print(f"Confirmed step progress: {report.get('confirmed_step_progress')}")
    print(
        "Unsafe/edge foothold rows: "
        f"unsafe={report.get('unsafe_foothold_rows')} edge={report.get('edge_foothold_rows')}"
    )
    print(
        f"Projection reasons: {report.get('projection_reasons')} "
        f"delta_peak={report.get('projection_delta_peak'):.4f}"
    )
    print(f"Replan reasons: {report.get('replan_reasons')}")
    print(f"Command jumps: {report.get('command_jumps')}")
    print(f"Worst hip tau-rate: {report.get('worst_hip_tau_rate')}")
    print("Hip peaks:")
    for leg, info in report.get("hip_report", {}).items():
        print(f"  {leg}: {info}")
    print("Worst touchdown windows:")
    for item in report.get("worst_touchdown_windows", []):
        print(f"  {item}")
    print(f"Preliminary cause tags: {report.get('cause_tags')}\n")


def configure_runtime_logging(log_decim: int | None = None, debug_joints: bool = True):
    global PROFILE_DEBUG_JOINTS
    PROFILE_DEBUG_JOINTS = bool(debug_joints)
    if log_decim is not None:
        ex18.LOG_CTRL_DECIM = max(1, int(log_decim))


def configure_profile(profile_name: str):
    global ACTIVE_PROFILE_NAME, ACTIVE_PHASE_POLICY, ACTIVE_BODY_PLANNER
    if profile_name not in PROFILE_STEP_HEIGHTS:
        raise ValueError(f"unknown profile {profile_name!r}")

    ex20.configure_active_high_stairs()
    profile_cls = Uniform06StairProfile if profile_name == "uniform06" else VariableStairProfile
    terrain = profile_cls()
    ACTIVE_PROFILE_NAME = profile_name
    ACTIVE_PHASE_POLICY = StairPhasePolicy(terrain)
    ACTIVE_BODY_PLANNER = BodyReferencePlanner(terrain)

    ex18.LOG_PREFIX = f"ex21_profile_{profile_name}"
    ex18.STAIRS_XML_PATH = PROFILE_XML[profile_name]
    ex18.StairTerrain = profile_cls
    ex18.StairAwareGait = ProfileStairGait
    ex18.RunLogger = ProfileRunLogger
    ex18.update_stair_reference_command = update_profile_reference_command
    ex18.compute_effective_velocity_command = compute_profile_effective_velocity_command
    ex18.compute_effective_yaw_rate_command = compute_profile_effective_yaw_rate_command
    ex18.compute_control_tick = compute_profile_control_tick

    ex18.GAIT_DUTY = 0.84
    ex18.GAIT_HZ = 0.60
    ex18.GAIT_T = 1.0 / ex18.GAIT_HZ
    ex18.MPC_DT = ex18.GAIT_T / 16
    ex18.MPC_HZ = 1.0 / ex18.MPC_DT
    ex18.STEPS_PER_MPC = max(1, int(ex18.CTRL_HZ // ex18.MPC_HZ))
    ex18.USE_STAIR_AWARE_FOOTHOLD = True
    ex18.USE_TERRAIN_AWARE_SWING = True
    ex18.USE_ACTIVE_STAIR_SWING = True
    ex18.STAIR_TERRAIN_AWARE_MPC_REFERENCE = False
    ex18.STAIR_MPC_TERRAIN_PITCH_REFERENCE = False
    ex18.STAIR_MOVE_ONLY_ALL_STANCE = False
    ex18.RESET_GAIT_ON_STAIR_ENTRY = False
    ex18.STAIR_SUPPORT_MARGIN_CONTROL = True
    ex18.Z_POS_RAMP = PROFILE_BODY_Z_RAMP
    ex18.STAIR_PITCH_RAMP = PROFILE_BODY_PITCH_RAMP
    ex18.STAIR_APPROACH_SPEED_ZONE_X = 0.35
    ex18.STAIR_X_VEL_LIMIT = 0.034
    ex18.STAIR_APPROACH_X_VEL_LIMIT = 0.055
    ex18.STAIR_RECOVERY_X_VEL_LIMIT = 0.014
    ex18.STAIR_Y_CENTER_KP = 1.45
    ex18.STAIR_Y_CENTER_VEL_LIMIT = 0.12
    ex18.STAIR_YAW_CENTER_KP = 0.42
    ex18.STAIR_YAW_RATE_LIMIT = 0.070
    ex18.STAIR_REAR_FOOTHOLD_MAX_DELTA_X = 0.20
    ex18.STAIR_GUARD_Y = 0.075
    ex18.STAIR_GUARD_Y_RELEASE = 0.035
    ex18.STAIR_GUARD_ROLL = 0.22
    ex18.STAIR_GUARD_ROLL_RELEASE = 0.13
    ex18.STAIR_GUARD_USE_STAND_GAIT = False
    ex18.STAIR_END_STOP_MARGIN_X = 0.35
    ex18.STAIR_END_SLOW_ZONE_X = 0.65
    ex18.STAIR_END_X_VEL_LIMIT = 0.0
    ex18.X_VEL_LIMIT = 0.07
    ex18.X_VEL_RAMP = 0.12
    if profile_name == "variable":
        ex18.STAIR_APPROACH_X_VEL_LIMIT = 0.055
        ex18.STAIR_X_VEL_LIMIT = 0.032
        ex18.STAIR_RECOVERY_X_VEL_LIMIT = 0.014
        ex18.STAIR_RECOVERY_Y = 0.060
        ex18.STAIR_RECOVERY_ROLL = 0.18
        ex18.STAIR_SUPPORT_SLOW_MARGIN_X = 0.110
        ex18.STAIR_SUPPORT_POLYGON_SLOW_MARGIN = 0.035
        ex18.STAIR_Y_CENTER_KP = 1.15
        ex18.STAIR_Y_CENTER_VEL_LIMIT = 0.10
        ex18.STAIR_YAW_CENTER_KP = 0.34
        ex18.STAIR_YAW_RATE_LIMIT = 0.060
        ex18.STAIR_PITCH_GAIN = 0.36
        ex18.STAIR_PITCH_LIMIT = 0.095
        ex18.STAIR_FORCE_REAR_STEP_UP = True
        ex18.STAIR_REAR_STEP_UP_BASE_MARGIN_X = 0.060
        ex18.STAIR_REAR_STEP_UP_MAX_GAP = 1
        ex18.X_VEL_LIMIT = 0.060
    return terrain


def run_headless(
    profile_name: str,
    duration_s: float,
    target_x_vel: float | None = None,
    stop_on_fall: bool = True,
    log_decim: int | None = None,
    debug_joints: bool = True,
    auto_analyze: bool = True,
    start_delay_s: float = 0.0,
):
    global ACTIVE_HEADLESS_START_DELAY_S, ACTIVE_HEADLESS_DELAY_TARGET_X_VEL
    ACTIVE_HEADLESS_START_DELAY_S = max(0.0, float(start_delay_s))
    ACTIVE_HEADLESS_DELAY_TARGET_X_VEL = float(target_x_vel) if ACTIVE_HEADLESS_START_DELAY_S > 0.0 and target_x_vel is not None else None
    configure_runtime_logging(log_decim=log_decim, debug_joints=debug_joints)
    terrain = configure_profile(profile_name)
    print(
        "EX21 profile stair climber: "
        f"profile={profile_name}, start_x={terrain.start_x:.2f}, depth={terrain.step_depth:.2f}, "
        f"heights={terrain.step_heights}, landing={terrain.landing_depth:.2f}, half_width_y={terrain.half_width_y:.2f}"
    )
    run_target_x_vel = 0.0 if ACTIVE_HEADLESS_DELAY_TARGET_X_VEL is not None else target_x_vel
    log_path, fell_reason = ex18.run_headless(duration_s, target_x_vel=run_target_x_vel, stop_on_fall=stop_on_fall)
    ACTIVE_HEADLESS_DELAY_TARGET_X_VEL = None
    augment_profile_log(log_path, profile_name)
    print(f"EX21 augmented log: {log_path}")
    if auto_analyze:
        print_profile_log_analysis(log_path)
    return log_path, fell_reason


def run_viewer(
    profile_name: str,
    target_x_vel: float | None = None,
    log_decim: int | None = None,
    debug_joints: bool = True,
):
    configure_runtime_logging(log_decim=log_decim, debug_joints=debug_joints)
    terrain = configure_profile(profile_name)
    print(
        "EX21 profile stair climber: "
        f"profile={profile_name}, start_x={terrain.start_x:.2f}, depth={terrain.step_depth:.2f}, "
        f"heights={terrain.step_heights}, landing={terrain.landing_depth:.2f}, half_width_y={terrain.half_width_y:.2f}"
    )
    if target_x_vel is None:
        ex18.run_viewer()
        return

    original_keyboard_command = ex18.KeyboardCommand

    class InitialVelocityKeyboardCommand(original_keyboard_command):
        def __post_init__(self):
            super().__post_init__()
            self.target_x_vel = float(target_x_vel)
            self.apply_limits()

    try:
        ex18.KeyboardCommand = InitialVelocityKeyboardCommand
        ex18.run_viewer()
    finally:
        ex18.KeyboardCommand = original_keyboard_command


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_STEP_HEIGHTS), default="uniform06")
    parser.add_argument("--headless", action="store_true", help="run without MuJoCo viewer")
    parser.add_argument("--duration", type=float, default=20.0, help="headless run duration in seconds")
    parser.add_argument("--target-x-vel", type=float, default=None, help="target forward velocity")
    parser.add_argument("--start-delay", type=float, default=0.0, help="headless seconds to stand before applying target-x-vel")
    parser.add_argument("--no-stop-on-fall", action="store_true", help="continue headless run after fall thresholds")
    parser.add_argument("--log-decim", type=int, default=None, help="control ticks per log row; use 1 for high-rate diagnostics")
    parser.add_argument("--no-debug-joints", action="store_true", help="disable EX21 joint/impact diagnostic columns")
    parser.add_argument("--no-auto-analyze", action="store_true", help="do not print CSV diagnostic analysis after headless runs")
    parser.add_argument("--analyze-log", type=Path, default=None, help="analyze an existing EX21 CSV log and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.analyze_log is not None:
        print_profile_log_analysis(args.analyze_log)
        return
    if args.headless:
        run_headless(
            args.profile,
            args.duration,
            target_x_vel=args.target_x_vel,
            stop_on_fall=not args.no_stop_on_fall,
            log_decim=args.log_decim,
            debug_joints=not args.no_debug_joints,
            auto_analyze=not args.no_auto_analyze,
            start_delay_s=args.start_delay,
        )
    else:
        run_viewer(
            args.profile,
            target_x_vel=args.target_x_vel,
            log_decim=args.log_decim,
            debug_joints=not args.no_debug_joints,
        )


if __name__ == "__main__":
    main()
