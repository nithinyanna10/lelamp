from __future__ import annotations

import mujoco
import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

# Serial kinematic chain, base to end-effector. Each body below owns exactly one
# hinge joint in the SO-ARM100 MJCF (shoulder_pan .. wrist_roll); the gripper jaw
# ("gripper" joint, on moving_jaw_so101_v1) sits past the pointing tip and is left
# out of the IK chain on purpose.
_CHAIN_BODIES = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]
_EE_SITE = "gripperframe"


def _quat_wxyz_to_rpy(quat: np.ndarray) -> tuple[float, float, float]:
    """URDF-convention (fixed-axis xyz) roll/pitch/yaw from a mujoco wxyz quaternion.

    Handles the pitch=+-90deg gimbal-lock singularity explicitly: several of this
    arm's CAD-exported link offsets land exactly on it, and the naive atan2 split
    is a coin flip on floating-point noise there (verified against MuJoCo's own
    xmat ground truth -- see tests/test_ik.py).
    """
    w, x, y, z = quat
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    cos_pitch = float(np.hypot(r00, r10))
    pitch = float(np.arctan2(-r20, cos_pitch))
    if cos_pitch > 1e-8:
        roll = float(np.arctan2(r21, r22))
        yaw = float(np.arctan2(r10, r00))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-r01, r11))
    return roll, pitch, yaw


class LampIK:
    """ikpy chain for the SO-ARM100, built by walking the MJCF's compiled kinematic
    tree (mujoco resolves body_pos/body_quat for us), so serial and sim share one
    kinematic model without a separate URDF."""

    def __init__(self, mjcf_path: str) -> None:
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        links: list[OriginLink | URDFLink] = [OriginLink()]
        active_mask = [False]

        for body_name in _CHAIN_BODIES:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            joint_id = model.body_jntadr[body_id]
            axis = model.jnt_axis[joint_id] if joint_id >= 0 else np.array([0.0, 0.0, 1.0])
            bounds = tuple(model.jnt_range[joint_id]) if joint_id >= 0 else (-np.pi, np.pi)
            rpy = _quat_wxyz_to_rpy(model.body_quat[body_id])
            links.append(
                URDFLink(
                    name=body_name,
                    origin_translation=model.body_pos[body_id].copy(),
                    origin_orientation=np.array(rpy),
                    rotation=axis.copy(),
                    bounds=bounds,
                )
            )
            active_mask.append(joint_id >= 0)

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _EE_SITE)
        rpy = _quat_wxyz_to_rpy(model.site_quat[site_id])
        links.append(
            URDFLink(
                name=_EE_SITE,
                origin_translation=model.site_pos[site_id].copy(),
                origin_orientation=np.array(rpy),
                rotation=np.array([0.0, 0.0, 1.0]),
                bounds=(0.0, 0.0),
            )
        )
        active_mask.append(False)

        self.chain = Chain(links=links, active_links_mask=active_mask, name="so_arm100")
        self._active_indices = [i for i, active in enumerate(active_mask) if active]

    def forward(self, joint_angles: list[float]) -> tuple[float, float, float]:
        full = [0.0] * len(self.chain.links)
        for idx, angle in zip(self._active_indices, joint_angles, strict=True):
            full[idx] = angle
        matrix = self.chain.forward_kinematics(full)
        return float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])

    def solve(
        self,
        target_xyz: tuple[float, float, float],
        target_orientation: tuple[float, float, float] | None = None,
        seed: list[float] | None = None,
    ) -> list[float]:
        if target_orientation is not None:
            raise NotImplementedError("orientation targeting not yet implemented")
        full_seed = [0.0] * len(self.chain.links)
        if seed is not None:
            for idx, angle in zip(self._active_indices, seed, strict=True):
                full_seed[idx] = angle
        result = self.chain.inverse_kinematics(
            target_position=list(target_xyz), initial_position=full_seed
        )
        return [float(result[i]) for i in self._active_indices]
