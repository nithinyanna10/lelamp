from __future__ import annotations

import asyncio

import numpy as np
import pytest
from conftest import FakeSerialPort

from lelamp.behavior.feetech import angle_to_raw, raw_to_angle
from lelamp.behavior.motor import (
    DEFAULT_JOINT_LIMITS,
    MuJoCoMotorBackend,
    SerialMotorBackend,
    interpolate,
)

MJCF_PATH = "assets/so_arm100/scene.xml"


def test_interpolate_eases_in_and_out() -> None:
    start, end = [0.0], [1.0]
    assert interpolate(start, end, 0.0, 1.0) == [0.0]
    assert interpolate(start, end, 1.0, 1.0) == [1.0]
    early = interpolate(start, end, 0.1, 1.0)[0]
    late = interpolate(start, end, 0.9, 1.0)[0]
    linear_early, linear_late = 0.1, 0.9
    assert early < linear_early  # slow start
    assert late > linear_late  # slow finish, i.e. mirrors the slow start


def test_feetech_angle_raw_roundtrip() -> None:
    joint_range = (-1.5, 1.5)
    for angle in (-1.5, -0.3, 0.0, 0.7, 1.5):
        raw = angle_to_raw(angle, joint_range)
        assert 0 <= raw <= 4095
        assert raw_to_angle(raw, joint_range) == pytest.approx(angle, abs=1e-3)


async def test_serial_backend_writes_sync_write_packet(
    mock_serial_port: FakeSerialPort, monkeypatch: pytest.MonkeyPatch
) -> None:
    import serial as serial_module

    monkeypatch.setattr(serial_module, "Serial", lambda *a, **k: mock_serial_port)

    backend = SerialMotorBackend(port="/dev/fake", tick_hz=50)
    await backend.connect()
    try:
        target = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        await backend.move_to(target, duration_s=0.1)
        assert mock_serial_port.written
        packet = mock_serial_port.written[-1]
        assert packet[0:2] == b"\xff\xff"
        assert packet[4] == 0x83  # INSTR_SYNC_WRITE
        state = await backend.get_state()
        assert state.joint_angles[0] == pytest.approx(0.1, abs=1e-2)
    finally:
        await backend.close()


async def test_mujoco_backend_converges_to_target_headless() -> None:
    backend = MuJoCoMotorBackend(MJCF_PATH, render=False, tick_hz=200)
    await backend.connect()
    try:
        target = [0.3, -0.2, 0.1, 0.15, -0.3, 0.0]
        await backend.move_to(target, duration_s=1.0)
        state = await backend.get_state()
        assert np.allclose(state.joint_angles, target, atol=0.05)
        assert not state.is_moving
    finally:
        await backend.close()


async def test_mujoco_backend_stop_preempts_move_to() -> None:
    backend = MuJoCoMotorBackend(MJCF_PATH, render=False, tick_hz=200)
    await backend.connect()
    try:
        target = [lo * 0.9 for lo, _ in DEFAULT_JOINT_LIMITS]
        task = asyncio.create_task(backend.move_to(target, duration_s=5.0))
        await asyncio.sleep(0.1)
        await backend.stop()
        await asyncio.wait_for(task, timeout=1.0)
        state = await backend.get_state()
        assert not np.allclose(state.joint_angles, target, atol=0.05)
    finally:
        await backend.close()
