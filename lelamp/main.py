from __future__ import annotations

import asyncio
import signal
import time

import structlog

from lelamp.behavior.expressions import HOME_DURATION_S, HOME_POSE, WAKE_DURATION_S, WAKE_POSE
from lelamp.behavior.motor import MotorBackend, make_motor_backend
from lelamp.perception.camera import Frame, camera_task
from lelamp.perception.debug_overlay import LatencySample, debug_overlay_task
from lelamp.perception.face_gaze import DebugFrame, GazeEvent, face_gaze_task
from lelamp.perception.hysteresis import EngagementTransition, HysteresisGate
from lelamp.state.fsm import LampFSM, LampState
from lelamp.telemetry import init_telemetry

log = structlog.get_logger()


async def hysteresis_task(
    gate: HysteresisGate,
    in_queue: asyncio.Queue[GazeEvent],
    out_queue: asyncio.Queue[EngagementTransition],
    latency_queue: asyncio.Queue[LatencySample] | None = None,
) -> None:
    while True:
        event = await in_queue.get()
        t0 = time.monotonic()
        transition = gate.update(event.gaze_score, timestamp=event.timestamp)
        if latency_queue is not None:
            latency_queue.put_nowait(
                LatencySample(stage="gaze_to_fsm", latency_ms=(time.monotonic() - t0) * 1000.0)
            )
        if transition is not None:
            out_queue.put_nowait(transition)


async def fsm_motor_task(
    fsm: LampFSM,
    in_queue: asyncio.Queue[EngagementTransition],
    motor: MotorBackend,
    latency_queue: asyncio.Queue[LatencySample] | None = None,
) -> None:
    while True:
        event = await in_queue.get()
        transition = fsm.on_engagement_transition(event)
        if transition is None:
            continue
        # Latency budget: "engagement transition -> first motor command dispatched" < 250ms p95.
        dispatch_latency_ms = (time.monotonic() - event.timestamp) * 1000.0
        log.info(
            "fsm_transition",
            dispatch_latency_ms=round(dispatch_latency_ms, 1),
            **transition.model_dump(),
        )
        if latency_queue is not None:
            latency_queue.put_nowait(
                LatencySample(stage="fsm_to_motor", latency_ms=dispatch_latency_ms)
            )
        if transition.to_state == LampState.ENGAGED:
            await motor.move_to(WAKE_POSE, duration_s=WAKE_DURATION_S)
        elif transition.to_state == LampState.IDLE:
            await motor.move_to(HOME_POSE, duration_s=HOME_DURATION_S)


async def run() -> None:
    provider = init_telemetry()

    frame_queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=2)
    gaze_queue: asyncio.Queue[GazeEvent] = asyncio.Queue(maxsize=32)
    debug_queue: asyncio.Queue[DebugFrame] = asyncio.Queue(maxsize=2)
    engagement_queue: asyncio.Queue[EngagementTransition] = asyncio.Queue(maxsize=32)
    latency_queue: asyncio.Queue[LatencySample] = asyncio.Queue(maxsize=256)

    motor = make_motor_backend()
    await motor.connect()
    await motor.move_to(HOME_POSE, duration_s=HOME_DURATION_S)

    gate = HysteresisGate()
    fsm = LampFSM()

    tasks = [
        asyncio.create_task(camera_task(frame_queue), name="camera"),
        asyncio.create_task(face_gaze_task(frame_queue, gaze_queue, debug_queue), name="face_gaze"),
        asyncio.create_task(
            hysteresis_task(gate, gaze_queue, engagement_queue, latency_queue), name="hysteresis"
        ),
        asyncio.create_task(
            fsm_motor_task(fsm, engagement_queue, motor, latency_queue), name="fsm_motor"
        ),
        asyncio.create_task(
            debug_overlay_task(debug_queue, gate, fsm, latency_queue), name="debug_overlay"
        ),
    ]

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await asyncio.wait(
        [*tasks, asyncio.create_task(stop_event.wait())], return_when=asyncio.FIRST_COMPLETED
    )

    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await motor.close()
    provider.shutdown()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
