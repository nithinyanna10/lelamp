import asyncio

from lelamp.behavior.ik import LampIK
from lelamp.behavior.motor import DEFAULT_HOME, MuJoCoMotorBackend
from lelamp.behavior.primitives import look_at
from lelamp.telemetry import init_telemetry

MJCF_PATH = "assets/so_arm100/scene.xml"


async def main() -> None:
    init_telemetry()
    motor, ik = MuJoCoMotorBackend(MJCF_PATH), LampIK(MJCF_PATH)
    await motor.connect()
    await motor.home()
    for i, (_, hi) in enumerate(motor.joint_limits[:5]):
        await motor.move_to([hi * 0.6 if j == i else 0.0 for j in range(6)], duration_s=1.2)
        await motor.move_to(DEFAULT_HOME, duration_s=1.2)
    await look_at(motor, ik, (0.3, 0.0, 0.4))
    await asyncio.sleep(5.0)
    await motor.close()


if __name__ == "__main__":
    asyncio.run(main())
