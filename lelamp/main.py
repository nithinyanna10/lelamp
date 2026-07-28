from __future__ import annotations

import asyncio

import structlog

from lelamp.behavior.motor import make_motor_backend
from lelamp.perception.audio import SpeechEvent, audio_task
from lelamp.perception.camera import Frame, camera_task
from lelamp.perception.face_gaze import GazeEvent, LampPosition, face_gaze_task
from lelamp.perception.scene_scan import Detections, scene_scan_task
from lelamp.state.fsm import LampFSM
from lelamp.telemetry import init_telemetry

log = structlog.get_logger()

SCENE_SCAN_VOCABULARY = ["water bottle", "mug", "phone", "keys", "book", "laptop"]


async def brain_task(
    gaze_queue: asyncio.Queue[GazeEvent],
    detections_queue: asyncio.Queue[Detections],
    speech_queue: asyncio.Queue[SpeechEvent],
    fsm: LampFSM,
) -> None:
    while True:
        gaze_event = await gaze_queue.get()
        transition = fsm.on_gaze_event(gaze_event)
        if transition is not None:
            log.info("fsm_transition", **transition.model_dump())


async def run() -> None:
    init_telemetry()

    frame_queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=2)
    gaze_queue: asyncio.Queue[GazeEvent] = asyncio.Queue(maxsize=32)
    detections_queue: asyncio.Queue[Detections] = asyncio.Queue(maxsize=8)
    speech_queue: asyncio.Queue[SpeechEvent] = asyncio.Queue(maxsize=32)

    motor = make_motor_backend()
    await motor.connect()

    lamp_position = LampPosition(x=0.0, y=0.0, z=0.0)
    fsm = LampFSM()

    tasks = [
        asyncio.create_task(camera_task(frame_queue), name="camera"),
        asyncio.create_task(
            face_gaze_task(frame_queue, gaze_queue, lamp_position), name="face_gaze"
        ),
        asyncio.create_task(
            scene_scan_task(frame_queue, detections_queue, SCENE_SCAN_VOCABULARY), name="scene_scan"
        ),
        asyncio.create_task(audio_task(speech_queue), name="audio"),
        asyncio.create_task(
            brain_task(gaze_queue, detections_queue, speech_queue, fsm), name="brain"
        ),
    ]

    try:
        await asyncio.gather(*tasks)
    finally:
        await motor.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
