from pathlib import Path
import os

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
import pinocchio as pin
import time

from .vbot_robot_data import MPC_LEG_ORDER, JOINTS, PinVBotModel


REPO = Path(__file__).resolve().parents[2]
XML_PATH = REPO / "models" / "MJCF" / "vbot" / "vbot_mpc_scene.xml"
ENV_XML_PATH = "VBOT_MPC_XML_PATH"
FLOATING_JOINT = "joint_fixed_world"


class MuJoCo_VBot_Model:
    def __init__(self, xml_path=None):
        if xml_path is None:
            xml_path = os.environ.get(ENV_XML_PATH, XML_PATH)
        self.xml_path = Path(xml_path)
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.viewer = None
        self.base_bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "base")
        self.base_jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, FLOATING_JOINT)
        self.base_qadr = int(self.model.jnt_qposadr[self.base_jid])
        self.base_vadr = int(self.model.jnt_dofadr[self.base_jid])

        self.joint_ids = {
            name: mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            for leg in MPC_LEG_ORDER
            for name in JOINTS[leg]
        }
        self.actuator_ids = {
            name: mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            for leg in MPC_LEG_ORDER
            for name in JOINTS[leg]
        }

    def update_with_q_pin(self, q_pin):
        q_pin = np.asarray(q_pin, dtype=float).reshape(-1)
        px, py, pz, qx, qy, qz, qw = q_pin[:7]
        self.data.qpos[self.base_qadr : self.base_qadr + 7] = [px, py, pz, qw, qx, qy, qz]

        for leg in MPC_LEG_ORDER:
            for value, joint_name in zip(getattr_slice(q_pin, leg), JOINTS[leg]):
                jid = self.joint_ids[joint_name]
                self.data.qpos[int(self.model.jnt_qposadr[jid])] = value

        mj.mj_forward(self.model, self.data)

    def set_leg_joint_torque(self, leg: str, torque):
        for value, joint_name in zip(np.asarray(torque).reshape(3), JOINTS[leg]):
            self.data.ctrl[self.actuator_ids[joint_name]] = value

    def set_joint_torque(self, torque):
        torque = np.asarray(torque, dtype=float).reshape(12)
        for i, leg in enumerate(MPC_LEG_ORDER):
            self.set_leg_joint_torque(leg, torque[3 * i : 3 * i + 3])

    def update_pin_with_mujoco(self, vbot: PinVBotModel):
        mujoco_q = np.asarray(self.data.qpos, dtype=float).reshape(-1)
        mujoco_dq = np.asarray(self.data.qvel, dtype=float).reshape(-1)

        base_q = mujoco_q[self.base_qadr : self.base_qadr + 7]
        base_v = mujoco_dq[self.base_vadr : self.base_vadr + 6]
        qw, qx, qy, qz = base_q[3:7]
        R = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        v_world = base_v[0:3]
        w_body = base_v[3:6]
        v_body = R.T @ v_world

        joint_q = []
        joint_v = []
        for leg in ("FR", "FL", "RR", "RL"):
            for joint_name in JOINTS[leg]:
                jid = self.joint_ids[joint_name]
                joint_q.append(mujoco_q[int(self.model.jnt_qposadr[jid])])
                joint_v.append(mujoco_dq[int(self.model.jnt_dofadr[jid])])

        q_pin = np.concatenate([base_q[0:3], [qx, qy, qz, qw], np.asarray(joint_q)])
        dq_pin = np.concatenate([v_body, w_body, np.asarray(joint_v)])
        vbot.update_model(q_pin, dq_pin)

    def replay_simulation(self, time_log_s, q_log, tau_log_Nm, render_dt, realtime_factor):
        model = self.model
        data_replay = mj.MjData(model)

        with mjv.launch_passive(model, data_replay) as viewer:
            viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = self.base_bid
            viewer.cam.fixedcamid = -1
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 90
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True

            while viewer.is_running():
                start_wall = time.perf_counter()
                t0 = time_log_s[0]
                next_render_t = t0
                k = 0
                while k < len(time_log_s) and viewer.is_running():
                    t = time_log_s[k]
                    if t >= next_render_t:
                        data_replay.qpos[:] = q_log[k]
                        data_replay.ctrl[:] = tau_log_Nm[k]
                        mj.mj_forward(model, data_replay)
                        viewer.sync()
                        sleep_time = start_wall + (t - t0) / realtime_factor - time.perf_counter()
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        next_render_t += render_dt
                    k += 1
                time.sleep(1)


def getattr_slice(q_pin, leg: str):
    pin_offsets = {"FR": 7, "FL": 10, "RR": 13, "RL": 16}
    start = pin_offsets[leg]
    return q_pin[start : start + 3]
