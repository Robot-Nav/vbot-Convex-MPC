from pathlib import Path

import numpy as np
import pinocchio as pin
from numpy import cos, sin


REPO = Path(__file__).resolve().parents[2]
MJCF_PATH = REPO / "models" / "MJCF" / "vbot" / "vbot_pinocchio.xml"

MPC_LEG_ORDER = ("FL", "FR", "RL", "RR")
PIN_LEG_ORDER = ("FR", "FL", "RR", "RL")#关节顺序和mujoco里不一样，注意区分
RIGHT_LEGS = {"FR", "RR"}
VBOT_ABAD_LINK_LENGTH = 0.0975
VBOT_HIP_LINK_LENGTH = 0.1985
VBOT_KNEE_LINK_LENGTH = 0.214
VBOT_ABAD_OFFSET_BODY = {
    "FL": np.array([0.18453, 0.051, 0.0]),
    "FR": np.array([0.18453, -0.051, 0.0]),
    "RL": np.array([-0.18453, 0.051, 0.0]),
    "RR": np.array([-0.18453, -0.051, 0.0]),
}

JOINTS = {
    "FL": ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
    "FR": ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
    "RL": ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
    "RR": ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
}
DEFAULT_JOINT_ANGLES = {
    "FL": np.array([0.0, 0.9, -1.8]),
    "FR": np.array([-0.0, 0.9, -1.8]),
    "RL": np.array([0.0, 0.9, -1.8]),
    "RR": np.array([-0.0, 0.9, -1.8]),
}


def build_freeflyer_mjcf_model(path: Path):
    try:
        model = pin.buildModelFromMJCF(
            str(path),
            pin.JointModelFreeFlyer(),
            "root_joint",
        )
    except TypeError:
        model = pin.buildModelFromMJCF(str(path), pin.JointModelFreeFlyer())
    if isinstance(model, tuple):
        model = model[0]
    return model


class VBotConfigurationState:
    def __init__(self):
        self.base_pos = np.array([0.0, 0.0, 0.462])
        self.base_quad = np.array([0.0, 0.0, 0.0, 1.0])
        self.base_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)

        for leg in MPC_LEG_ORDER:
            setattr(self, f"{leg}_joint_angle", DEFAULT_JOINT_ANGLES[leg].copy())
            setattr(self, f"{leg}_joint_vel", np.zeros(3))

    def get_q(self):
        joint_q = [getattr(self, f"{leg}_joint_angle") for leg in PIN_LEG_ORDER]
        return np.concatenate([self.base_pos, self.base_quad, *joint_q])

    def get_dq(self):
        joint_v = [getattr(self, f"{leg}_joint_vel") for leg in PIN_LEG_ORDER]
        return np.concatenate([self.base_vel, self.base_ang_vel, *joint_v])

    def update_q(self, q):
        self.base_pos = np.asarray(q[0:3], dtype=float)
        self.base_quad = np.asarray(q[3:7], dtype=float)
        joints = np.asarray(q[7:19], dtype=float)
        for i, leg in enumerate(PIN_LEG_ORDER):
            setattr(self, f"{leg}_joint_angle", joints[3 * i : 3 * i + 3])

    def update_dq(self, v):
        self.base_vel = np.asarray(v[0:3], dtype=float)
        self.base_ang_vel = np.asarray(v[3:6], dtype=float)
        joints = np.asarray(v[6:18], dtype=float)
        for i, leg in enumerate(PIN_LEG_ORDER):
            setattr(self, f"{leg}_joint_vel", joints[3 * i : 3 * i + 3])

    def compute_euler_angle_world(self):
        qx, qy, qz, qw = self.base_quad
        R = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        roll, pitch, yaw_meas = np.array(pin.rpy.matrixToRpy(R)).reshape(3)

        if not hasattr(self, "_yaw_unwrap_initialized"):
            self._yaw_unwrap_initialized = True
            self._yaw_prev_meas = yaw_meas
            self._yaw_cont = yaw_meas
        else:
            yaw_delta = (yaw_meas - self._yaw_prev_meas + np.pi) % (2 * np.pi) - np.pi
            self._yaw_cont += yaw_delta
            self._yaw_prev_meas = yaw_meas

        return np.array([roll, pitch, self._yaw_cont])


class PinVBotModel:
    def __init__(self):
        self.model = build_freeflyer_mjcf_model(MJCF_PATH)
        self.data = self.model.createData()
        self.current_config = VBotConfigurationState()
        self.q_init = self.current_config.get_q()
        self.dq_init = self.current_config.get_dq()

        self.base_id = self.model.getFrameId("base")
        for leg in MPC_LEG_ORDER:
            setattr(self, f"{leg}_foot_id", self.model.getFrameId(f"{leg}_foot"))
            setattr(self, f"{leg}_hip_id", self.model.getFrameId(f"{leg}_thigh_joint"))

        self.update_model(self.q_init, self.dq_init)

        self.x_pos_des_world = 0.0
        self.y_pos_des_world = 0.0
        self.x_vel_des_world = 0.0
        self.y_vel_des_world = 0.0
        self.yaw_rate_des_world = 0.0

    def get_leg_joint_vcols(self, leg: str):
        return [self.model.joints[self.model.getJointId(name)].idx_v for name in JOINTS[leg]]

    def get_hip_offset(self, leg: str):
        return getattr(self, f"{leg.upper()}_hip_offset")

    def get_abad_offset_body(self, leg: str):
        return VBOT_ABAD_OFFSET_BODY[leg].copy()

    def _leg_side_sign(self, leg: str):
        return -1.0 if leg in RIGHT_LEGS else 1.0

    def calc_leg_fk_hip(self, leg: str, q_leg):
        q = np.asarray(q_leg, dtype=float).reshape(3)
        l1 = self._leg_side_sign(leg) * VBOT_ABAD_LINK_LENGTH
        l2 = -VBOT_HIP_LINK_LENGTH
        l3 = -VBOT_KNEE_LINK_LENGTH

        s1, s2, s3 = np.sin(q)
        c1, c2, c3 = np.cos(q)
        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        return np.array([
            l3 * s23 + l2 * s2,
            -l3 * s1 * c23 + l1 * c1 - l2 * c2 * s1,
            l3 * c1 * c23 + l1 * s1 + l2 * c1 * c2,
        ])

    def calc_leg_fk_body(self, leg: str, q_leg):
        return self.get_abad_offset_body(leg) + self.calc_leg_fk_hip(leg, q_leg)

    def calc_leg_jacobian_body(self, leg: str, q_leg):
        q = np.asarray(q_leg, dtype=float).reshape(3)
        l1 = self._leg_side_sign(leg) * VBOT_ABAD_LINK_LENGTH
        l2 = -VBOT_HIP_LINK_LENGTH
        l3 = -VBOT_KNEE_LINK_LENGTH

        s1, s2, s3 = np.sin(q)
        c1, c2, c3 = np.cos(q)
        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        jaco = np.zeros((3, 3), dtype=float)
        jaco[0, 0] = 0.0
        jaco[1, 0] = -l3 * c1 * c23 - l2 * c1 * c2 - l1 * s1
        jaco[2, 0] = -l3 * s1 * c23 - l2 * c2 * s1 + l1 * c1
        jaco[0, 1] = l3 * c23 + l2 * c2
        jaco[1, 1] = l3 * s1 * s23 + l2 * s1 * s2
        jaco[2, 1] = -l3 * c1 * s23 - l2 * c1 * s2
        jaco[0, 2] = l3 * c23
        jaco[1, 2] = l3 * s1 * s23
        jaco[2, 2] = -l3 * c1 * s23
        return jaco

    def calc_leg_ik_body(self, leg: str, foot_pos_body):
        p = np.asarray(foot_pos_body, dtype=float).reshape(3) - self.get_abad_offset_body(leg)
        px, py, pz = p
        b2y = VBOT_ABAD_LINK_LENGTH * self._leg_side_sign(leg)
        b3z = -VBOT_HIP_LINK_LENGTH
        b4z = -VBOT_KNEE_LINK_LENGTH
        a = VBOT_ABAD_LINK_LENGTH

        c = float(np.linalg.norm(p))
        b_sq = max(c * c - a * a, 1.0e-12)
        b = float(np.sqrt(b_sq))

        lateral_sq = max(py * py + pz * pz - b2y * b2y, 1.0e-12)
        lateral = float(np.sqrt(lateral_sq))
        q1 = float(np.arctan2(pz * b2y + py * lateral, py * b2y - pz * lateral))

        temp = (b3z * b3z + b4z * b4z - b * b) / (2.0 * abs(b3z * b4z))
        q3 = float(-(np.pi - np.arccos(np.clip(temp, -1.0, 1.0))))

        a1 = py * np.sin(q1) - pz * np.cos(q1)
        a2 = px
        m1 = b4z * np.sin(q3)
        m2 = b3z + b4z * np.cos(q3)
        q2 = float(np.arctan2(m1 * a1 + m2 * a2, m1 * a2 - m2 * a1))

        return np.array([q1, q2, q3], dtype=float)

    def calc_leg_qd_body(self, leg: str, q_leg, foot_vel_body):
        J = self.calc_leg_jacobian_body(leg, q_leg)
        v = np.asarray(foot_vel_body, dtype=float).reshape(3)
        try:
            return np.linalg.solve(J, v)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(J, v, rcond=None)[0]

    def compute_com_x_vec(self):
        pos_com_world = self.pos_com_world
        rpy_com_world = self.current_config.compute_euler_angle_world()
        vel_com_world = self.vel_com_world
        omega_world = self.R_body_to_world @ self.current_config.base_ang_vel
        return np.concatenate([pos_com_world, rpy_com_world, vel_com_world, omega_world]).reshape(-1, 1)

    def update_model(self, q, dq):
        q = np.asarray(q, dtype=float).reshape(-1)
        dq = np.asarray(dq, dtype=float).reshape(-1)
        self.current_config.update_q(q)
        self.current_config.update_dq(dq)

        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)
        pin.ccrba(self.model, self.data, q, dq)
        pin.centerOfMass(self.model, self.data, q, dq)

        self.oMb = self.data.oMf[self.base_id]
        self.pos_com_world = np.asarray(self.data.com[0]).copy()
        self.vel_com_world = np.asarray(self.data.vcom[0]).copy()

        for i, leg in enumerate(MPC_LEG_ORDER, start=1):
            setattr(self, f"oMf{i}", self.data.oMf[getattr(self, f"{leg}_foot_id")])

        for leg in MPC_LEG_ORDER:
            oMh = self.data.oMf[getattr(self, f"{leg}_hip_id")]
            setattr(self, f"{leg}_hip_offset", self.oMb.actInv(oMh).translation.copy())

        yaw = self.current_config.compute_euler_angle_world()[2]
        self.R_body_to_world = np.asarray(self.oMb.rotation).copy()
        self.R_world_to_body = self.R_body_to_world.T
        self.R_z = np.array([
            [cos(yaw), -sin(yaw), 0.0],
            [sin(yaw), cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])

    def update_model_simplified(self, q, dq):
        roll, pitch, yaw = q[3:6]
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        joint_q = [DEFAULT_JOINT_ANGLES[leg] for leg in PIN_LEG_ORDER]
        q_full = np.concatenate([q[0:3], [qx, qy, qz, qw], *joint_q])
        dq_full = np.concatenate([dq[0:6], np.zeros(12)])
        self.update_model(q_full, dq_full)

    def get_foot_placement_in_world(self):
        return tuple(self.data.oMf[getattr(self, f"{leg}_foot_id")].translation.copy() for leg in MPC_LEG_ORDER)

    def get_foot_lever_world(self):
        return tuple(
            self.data.oMf[getattr(self, f"{leg}_foot_id")].translation.copy() - self.pos_com_world
            for leg in MPC_LEG_ORDER
        )

    def get_single_foot_state_in_world(self, leg: str):
        foot_id = getattr(self, f"{leg}_foot_id")
        foot_pos_world = self.data.oMf[foot_id].translation.copy()
        v6 = pin.getFrameVelocity(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return foot_pos_world, np.asarray(v6.linear).copy()

    def compute_3x3_foot_Jacobian_world(self, leg: str):
        foot_id = getattr(self, f"{leg}_foot_id")
        J_world = pin.getFrameJacobian(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J_world[0:3, self.get_leg_joint_vcols(leg)]

    def compute_full_foot_Jacobian_world(self, leg: str):
        foot_id = getattr(self, f"{leg}_foot_id")
        J_world = pin.getFrameJacobian(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J_world[0:3, :]

    def compute_Jdot_dq_world(self, leg: str):
        foot_id = getattr(self, f"{leg}_foot_id")
        Jdot = pin.getFrameJacobianTimeVariation(
            self.model,
            self.data,
            foot_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return np.asarray(Jdot[0:3, :] @ self.current_config.get_dq()).reshape(3)

    def compute_dynamcis_terms(self):
        return self.data.g, self.data.C, self.data.M
