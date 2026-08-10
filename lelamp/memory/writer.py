from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import structlog
from pydantic import BaseModel

from lelamp.memory.db import (
    BBox2D,
    ObjectRecord,
    SceneRecord,
    SightingRecord,
    find_dedupe_match,
    get_object,
    insert_scene,
    insert_sighting,
    upsert_object,
)
from lelamp.perception.scene_scan import Detections
from lelamp.telemetry import get_tracer

_tracer = get_tracer(__name__)
log = structlog.get_logger()

CROPS_DIR = Path("memory/crops")


class ScanResult(BaseModel):
    new_objects: int
    updated_objects: int
    total_detections: int
    scene_id: str


def _save_crop(object_id: int, image_crop: np.ndarray, crops_dir: Path) -> str:
    crops_dir.mkdir(parents=True, exist_ok=True)
    path = crops_dir / f"{object_id}.jpg"
    cv2.imwrite(str(path), image_crop)
    return str(path)


def write_scan(
    conn: sqlite3.Connection, detections: Detections, crops_dir: Path = CROPS_DIR
) -> ScanResult:
    """Dedupe against object_vecs (same class, cosine > 0.85, position within 0.3
    normalized, last seen within 30 min) and either update the matched object or
    insert a new one; always inserts a sighting row and a scenes row."""
    with _tracer.start_as_current_span("memory.write_scan") as span:
        new_objects = 0
        updated_objects = 0
        now = detections.timestamp

        for det in detections.detections:
            match_id = find_dedupe_match(
                conn, det.class_name, det.position_xy_normalized, det.embedding or [], now=now
            )
            if match_id is not None:
                existing = get_object(conn, match_id)
                assert existing is not None
                existing.last_seen_ts = now
                existing.position_xy_normalized = det.position_xy_normalized
                existing.confidence = max(existing.confidence, det.confidence)
                existing.sighting_count += 1
                existing.embedding = det.embedding or existing.embedding
                object_id = upsert_object(conn, existing)
                updated_objects += 1
            else:
                record = ObjectRecord(
                    class_name=det.class_name,
                    first_seen_ts=now,
                    last_seen_ts=now,
                    position_xy_normalized=det.position_xy_normalized,
                    confidence=det.confidence,
                    embedding=det.embedding or [],
                )
                object_id = upsert_object(conn, record)
                record.id = object_id
                record.image_crop_path = _save_crop(object_id, det.image_crop, crops_dir)
                upsert_object(conn, record)
                new_objects += 1

            insert_sighting(
                conn,
                SightingRecord(
                    object_id=object_id,
                    ts=now,
                    bbox=BBox2D(
                        x_min=det.bbox.x_min,
                        y_min=det.bbox.y_min,
                        x_max=det.bbox.x_max,
                        y_max=det.bbox.y_max,
                    ),
                    frame_id=detections.frame_id,
                    scene_id=detections.scene_id,
                ),
            )

        class_names = sorted({d.class_name for d in detections.detections})
        insert_scene(
            conn,
            SceneRecord(
                id=detections.scene_id,
                ts=now,
                summary_text=", ".join(class_names) or "nothing",
                num_objects=len(detections.detections),
            ),
        )

        result = ScanResult(
            new_objects=new_objects,
            updated_objects=updated_objects,
            total_detections=len(detections.detections),
            scene_id=detections.scene_id,
        )
        span.set_attribute("new_objects", new_objects)
        span.set_attribute("updated_objects", updated_objects)
        span.set_attribute("total_detections", result.total_detections)
        return result


async def memory_writer_task(
    conn: sqlite3.Connection,
    in_queue: asyncio.Queue[Detections],
    crops_dir: Path = CROPS_DIR,
) -> None:
    # ponytail: write_scan runs inline, not via asyncio.to_thread -- sqlite3
    # connections are thread-affine, and `conn` is created on this same loop's thread.
    while True:
        detections = await in_queue.get()
        result = write_scan(conn, detections, crops_dir)
        log.info("scene_scan_written", **result.model_dump())
