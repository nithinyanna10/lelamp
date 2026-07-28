from __future__ import annotations

from lelamp.behavior.ik import LampIK
from lelamp.behavior.motor import MotorBackend

LOOK_AT_DURATION_S = 1.5


async def wake(motor: MotorBackend) -> None:
    raise NotImplementedError


async def sleep(motor: MotorBackend) -> None:
    raise NotImplementedError


async def nod(motor: MotorBackend) -> None:
    raise NotImplementedError


async def curious_tilt(motor: MotorBackend) -> None:
    raise NotImplementedError


async def look_at(motor: MotorBackend, ik: LampIK, target: tuple[float, float, float]) -> None:
    state = await motor.get_state()
    pose_angles = ik.solve(target, seed=state.joint_angles[:5])
    await motor.move_to([*pose_angles, state.joint_angles[5]], duration_s=LOOK_AT_DURATION_S)


async def point_at(motor: MotorBackend, ik: LampIK, target: tuple[float, float, float]) -> None:
    await look_at(motor, ik, target)


async def breathe(motor: MotorBackend, base: list[float], amplitude_rad: float = 0.03) -> None:
    """Low-amplitude sinusoidal overlay during IDLE so the lamp never looks dead."""
    raise NotImplementedError
