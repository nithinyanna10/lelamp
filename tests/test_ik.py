from __future__ import annotations

import mujoco
import numpy as np
import pytest

from lelamp.behavior.ik import LampIK

MJCF_PATH = "assets/so_arm100/scene.xml"


@pytest.fixture(scope="module")
def ik() -> LampIK:
    return LampIK(MJCF_PATH)


@pytest.fixture(scope="module")
def mujoco_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(MJCF_PATH)


def test_forward_matches_mujoco_ground_truth(ik: LampIK, mujoco_model: mujoco.MjModel) -> None:
    data = mujoco.MjData(mujoco_model)
    site_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    rng = np.random.default_rng(0)
    for _ in range(20):
        angles = [rng.uniform(lo, hi) for lo, hi in mujoco_model.jnt_range[:5]]
        data.qpos[:5] = angles
        data.qpos[5] = 0.0
        mujoco.mj_forward(mujoco_model, data)
        truth = data.site_xpos[site_id].copy()
        pred = ik.forward(angles)
        assert np.linalg.norm(np.array(pred) - truth) < 1e-6


def test_forward_matches_at_gimbal_lock_offsets(ik: LampIK, mujoco_model: mujoco.MjModel) -> None:
    """upper_arm's fixed body_quat lands exactly on the pitch=+-90deg singularity;
    this is a regression test for that specific bug (see ik.py's _quat_wxyz_to_rpy)."""
    data = mujoco.MjData(mujoco_model)
    site_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    data.qpos[:] = 0.0
    mujoco.mj_forward(mujoco_model, data)
    truth = data.site_xpos[site_id].copy()
    pred = ik.forward([0.0] * 5)
    assert np.linalg.norm(np.array(pred) - truth) < 1e-6


def test_solve_converges_to_target(ik: LampIK) -> None:
    target = (0.2, 0.05, 0.15)
    angles = ik.solve(target)
    reached = ik.forward(angles)
    assert np.linalg.norm(np.array(reached) - np.array(target)) < 1e-3


def test_solve_rejects_orientation_for_now(ik: LampIK) -> None:
    with pytest.raises(NotImplementedError):
        ik.solve((0.2, 0.0, 0.1), target_orientation=(0.0, 0.0, 0.0))
