"""Step 4 end-to-end demo: webcam -> YOLO-World scene scan -> CLIP embeddings ->
memory DB, then a handful of example queries against what got stored.

Opens the webcam and the MuJoCo sim window (lamp sits at home pose -- no sweep
yet, see scene_scan.py's TODO), runs a few scene scans back-to-back to report
p50/p95 latency, writes them to memory.db, then runs 5 example queries.
Ctrl-C to stop.
"""

from __future__ import annotations

import asyncio
import statistics
import time

from lelamp.behavior.motor import DEFAULT_HOME, MuJoCoMotorBackend
from lelamp.memory import query as q
from lelamp.memory.db import init_db
from lelamp.memory.embeddings import ClipEmbedder
from lelamp.memory.writer import write_scan
from lelamp.perception.camera import Frame, camera_task
from lelamp.perception.scene_scan import scan_scene
from lelamp.telemetry import init_telemetry

MJCF_PATH = "assets/so_arm100/scene.xml"
DB_PATH = "memory.db"
NUM_SCANS = 5


async def _grab_frame(frame_queue: asyncio.Queue[Frame]) -> Frame:
    return await frame_queue.get()


async def main() -> None:
    init_telemetry()

    motor = MuJoCoMotorBackend(MJCF_PATH)
    await motor.connect()
    await motor.move_to(DEFAULT_HOME, duration_s=1.0)

    frame_queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=2)
    camera = asyncio.create_task(camera_task(frame_queue))
    await asyncio.sleep(1.0)  # let the webcam warm up

    conn = init_db(DB_PATH)
    embedder = ClipEmbedder()

    try:
        latencies_ms: list[float] = []
        detections = None
        for i in range(NUM_SCANS):
            frame = await _grab_frame(frame_queue)
            t0 = time.monotonic()
            detections = await scan_scene(frame, embedder)
            latencies_ms.append((time.monotonic() - t0) * 1000.0)
            print(f"scan {i + 1}/{NUM_SCANS}: {len(detections.detections)} detections")
            for det in detections.detections:
                print(
                    f"  {det.class_name} conf={det.confidence:.2f} "
                    f"pos={det.position_xy_normalized}"
                )

        assert detections is not None
        result = write_scan(conn, detections)
        print(f"\nScanResult: {result}")

        sorted_latencies = sorted(latencies_ms)
        p50 = statistics.median(sorted_latencies)
        p95 = sorted_latencies[min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95))]
        print(f"scan latency: p50={p50:.1f}ms p95={p95:.1f}ms\n")

        examples = [
            (
                "query_memory('a bottle')",
                q.query_memory(conn, embedder, q.QueryMemoryInput(description="a bottle")),
            ),
            (
                "query_memory('something to write with')",
                q.query_memory(
                    conn, embedder, q.QueryMemoryInput(description="something to write with")
                ),
            ),
            ("query_by_class('laptop')", q.query_by_class(conn, "laptop")),
            ("query_recent(300s)", q.query_recent(conn, time_window_seconds=300)),
            ("describe_current_memory()", q.describe_current_memory(conn)),
        ]
        for label, coro in examples:
            result_value = await coro
            print(f"{label}:")
            if isinstance(result_value, q.QueryMemoryResult):
                for match in result_value.matches:
                    print(f"  {match.object.class_name} similarity={match.similarity:.3f}")
            elif isinstance(result_value, list):
                for record in result_value:
                    print(f"  {record.class_name} last_seen={record.last_seen_ts:.0f}")
            else:
                print(f"  {result_value}")
    finally:
        camera.cancel()
        await asyncio.gather(camera, return_exceptions=True)
        await motor.close()
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())
