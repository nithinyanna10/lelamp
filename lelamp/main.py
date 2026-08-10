from __future__ import annotations

import asyncio
import signal
import sqlite3
import time

import structlog

from lelamp.behavior.expression_player import ExpressionPlayer
from lelamp.behavior.idle_overlay import IdleOverlay
from lelamp.behavior.motor import make_motor_backend
from lelamp.memory.db import init_db
from lelamp.memory.embeddings import ClipEmbedder
from lelamp.memory.writer import write_scan
from lelamp.perception.camera import Frame, camera_task
from lelamp.perception.debug_overlay import HudState, LatencySample, debug_overlay_task
from lelamp.perception.face_gaze import DebugFrame, GazeEvent, face_gaze_task
from lelamp.perception.hysteresis import HysteresisConfig, HysteresisGate
from lelamp.perception.scene_scan import (
    Detections,
    LatestFrame,
    SceneScanRequest,
    scene_scan_task,
)
from lelamp.state.fsm import (
    Event,
    FaceLost,
    FacePresent,
    GazeDisengaged,
    GazeEngaged,
    LampFSM,
    ScanComplete,
    Tick,
)
from lelamp.telemetry import init_telemetry

log = structlog.get_logger()

DEFAULT_DB_PATH = "memory.db"

# Symmetric, short hold: "is a face in frame at all" is a coarser, faster-settling
# signal than the engage/disengage gate below (which also requires *looking at*
# the lamp, longer holds) -- reuses HysteresisGate with a different config rather
# than writing a second debouncer.
_PRESENCE_CONFIG = HysteresisConfig(
    engage_threshold=0.5, engage_hold_ms=500.0, disengage_threshold=0.5, disengage_hold_ms=500.0
)


async def perception_to_fsm_task(
    engagement_gate: HysteresisGate,
    presence_gate: HysteresisGate,
    in_queue: asyncio.Queue[GazeEvent],
    out_queue: asyncio.Queue[Event],
    latency_queue: asyncio.Queue[LatencySample] | None = None,
) -> None:
    while True:
        gaze_event = await in_queue.get()
        t0 = time.monotonic()
        engagement = engagement_gate.update(gaze_event.gaze_score, timestamp=gaze_event.timestamp)
        presence = presence_gate.update(
            1.0 if gaze_event.face_present else 0.0, timestamp=gaze_event.timestamp
        )
        if latency_queue is not None:
            latency_queue.put_nowait(
                LatencySample(stage="gaze_to_fsm", latency_ms=(time.monotonic() - t0) * 1000.0)
            )
        if engagement is not None:
            fsm_event: Event = (
                GazeEngaged(timestamp=engagement.timestamp)
                if engagement.engaged
                else GazeDisengaged(timestamp=engagement.timestamp)
            )
            out_queue.put_nowait(fsm_event)
        if presence is not None:
            fsm_event = (
                FacePresent(timestamp=presence.timestamp)
                if presence.engaged
                else FaceLost(timestamp=presence.timestamp)
            )
            out_queue.put_nowait(fsm_event)


async def tick_task(out_queue: asyncio.Queue[Event], interval_s: float) -> None:
    while True:
        await asyncio.sleep(interval_s)
        out_queue.put_nowait(Tick(timestamp=time.monotonic()))


async def fsm_event_task(fsm: LampFSM, in_queue: asyncio.Queue[Event]) -> None:
    while True:
        event = await in_queue.get()
        await fsm.handle_event(event)


async def frame_fanout_task(
    in_queue: asyncio.Queue[Frame],
    gaze_queue: asyncio.Queue[Frame],
    latest_frame: LatestFrame,
) -> None:
    """Sits between camera_task and face_gaze_task so scene_scan can read
    'whatever the current frame is' on demand, without a second consumer racing
    face_gaze_task for items on the same queue -- camera.py stays untouched."""
    while True:
        frame = await in_queue.get()
        latest_frame.frame = frame
        if gaze_queue.full():
            try:
                gaze_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        gaze_queue.put_nowait(frame)


async def scan_writer_task(
    conn: sqlite3.Connection,
    detections_queue: asyncio.Queue[Detections],
    event_queue: asyncio.Queue[Event],
    hud: HudState,
) -> None:
    """write_scan() called directly (not via memory.writer.memory_writer_task) so
    this task can also emit ScanComplete for the FSM and update the HUD's memory
    counters -- memory/writer.py itself stays untouched, it doesn't know the FSM
    or HUD exist."""
    while True:
        detections = await detections_queue.get()
        result = write_scan(conn, detections)
        log.info("scene_scan_written", **result.model_dump())

        now = time.time()
        hud.last_scan_ts = time.monotonic()
        hud.last_scan_objects = result.total_detections
        hud.last_scan_new = result.new_objects
        hud.memory_total = conn.execute("SELECT COUNT(*) AS n FROM objects").fetchone()["n"]
        hud.memory_recent = conn.execute(
            "SELECT COUNT(*) AS n FROM objects WHERE last_seen_ts >= ?", (now - 300.0,)
        ).fetchone()["n"]

        event_queue.put_nowait(
            ScanComplete(
                timestamp=time.monotonic(),
                num_objects=result.total_detections,
                num_new=result.new_objects,
            )
        )


async def run() -> None:
    provider = init_telemetry()

    raw_frame_queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=2)
    gaze_frame_queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=2)
    gaze_queue: asyncio.Queue[GazeEvent] = asyncio.Queue(maxsize=32)
    debug_queue: asyncio.Queue[DebugFrame] = asyncio.Queue(maxsize=2)
    event_queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=64)
    latency_queue: asyncio.Queue[LatencySample] = asyncio.Queue(maxsize=256)
    scan_request_queue: asyncio.Queue[SceneScanRequest] = asyncio.Queue(maxsize=4)
    detections_queue: asyncio.Queue[Detections] = asyncio.Queue(maxsize=4)
    latest_frame = LatestFrame()

    motor = make_motor_backend()
    await motor.connect()

    idle_overlay = IdleOverlay()
    idle_overlay.set_enabled(False)  # SLEEPING is the boot state
    player = ExpressionPlayer(motor, idle_overlay)
    await player.preempt("sleep")

    engagement_gate = HysteresisGate()
    presence_gate = HysteresisGate(_PRESENCE_CONFIG)
    conn = init_db(DEFAULT_DB_PATH)
    embedder = ClipEmbedder()
    hud = HudState()
    fsm = LampFSM(player, scan_request_queue, idle_overlay)

    tasks = [
        asyncio.create_task(camera_task(raw_frame_queue), name="camera"),
        asyncio.create_task(
            frame_fanout_task(raw_frame_queue, gaze_frame_queue, latest_frame), name="frame_fanout"
        ),
        asyncio.create_task(
            face_gaze_task(gaze_frame_queue, gaze_queue, debug_queue), name="face_gaze"
        ),
        asyncio.create_task(
            perception_to_fsm_task(
                engagement_gate, presence_gate, gaze_queue, event_queue, latency_queue
            ),
            name="perception_to_fsm",
        ),
        asyncio.create_task(tick_task(event_queue, fsm.timings.tick_interval_s), name="tick"),
        asyncio.create_task(fsm_event_task(fsm, event_queue), name="fsm_event"),
        asyncio.create_task(
            debug_overlay_task(debug_queue, engagement_gate, fsm, player, hud, latency_queue),
            name="debug_overlay",
        ),
        asyncio.create_task(
            scene_scan_task(scan_request_queue, latest_frame, detections_queue, embedder),
            name="scene_scan",
        ),
        asyncio.create_task(
            scan_writer_task(conn, detections_queue, event_queue, hud), name="scan_writer"
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
    conn.close()
    provider.shutdown()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
