from __future__ import annotations

import math
import sqlite3
import struct
import time

import sqlite_vec
from pydantic import BaseModel

_EMBEDDING_DIM = 512


def _deserialize_float32(blob: bytes) -> list[float]:
    # sqlite_vec only ships serialize_float32; unpack the raw little-endian float32 blob ourselves.
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class BBox2D(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class ObjectRecord(BaseModel):
    id: int | None = None
    class_name: str
    first_seen_ts: float
    last_seen_ts: float
    position_xy_normalized: tuple[float, float]  # image-frame [-1, 1]; 3D comes in step 7
    confidence: float
    sighting_count: int = 1
    embedding: list[float]
    image_crop_path: str | None = None


class SightingRecord(BaseModel):
    id: int | None = None
    object_id: int
    ts: float
    bbox: BBox2D
    frame_id: int
    scene_id: str


class SceneRecord(BaseModel):
    id: str
    ts: float
    summary_text: str
    num_objects: int = 0
    # ponytail: scene-level semantic search (scene_vecs) deferred, no caller needs it yet.
    embedding: list[float] | None = None


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT NOT NULL,
    first_seen_ts REAL NOT NULL,
    last_seen_ts REAL NOT NULL,
    position_x REAL NOT NULL,
    position_y REAL NOT NULL,
    confidence REAL NOT NULL,
    sighting_count INTEGER NOT NULL DEFAULT 1,
    image_crop_path TEXT
);

CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id INTEGER NOT NULL REFERENCES objects(id),
    ts REAL NOT NULL,
    bbox_x_min REAL NOT NULL,
    bbox_y_min REAL NOT NULL,
    bbox_x_max REAL NOT NULL,
    bbox_y_max REAL NOT NULL,
    frame_id INTEGER NOT NULL,
    scene_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scenes (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    summary_text TEXT NOT NULL,
    num_objects INTEGER NOT NULL DEFAULT 0
);

-- free-text facts for the `remember` LLM tool (step 7)
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    text TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS object_vecs USING vec0(
    object_id INTEGER PRIMARY KEY,
    embedding FLOAT[{_EMBEDDING_DIM}]
);
"""


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _row_to_object(row: sqlite3.Row, embedding: list[float]) -> ObjectRecord:
    return ObjectRecord(
        id=row["id"],
        class_name=row["class_name"],
        first_seen_ts=row["first_seen_ts"],
        last_seen_ts=row["last_seen_ts"],
        position_xy_normalized=(row["position_x"], row["position_y"]),
        confidence=row["confidence"],
        sighting_count=row["sighting_count"],
        embedding=embedding,
        image_crop_path=row["image_crop_path"],
    )


def get_object(conn: sqlite3.Connection, object_id: int) -> ObjectRecord | None:
    row = conn.execute("SELECT * FROM objects WHERE id = ?", (object_id,)).fetchone()
    if row is None:
        return None
    vec_row = conn.execute(
        "SELECT embedding FROM object_vecs WHERE object_id = ?", (object_id,)
    ).fetchone()
    embedding = _deserialize_float32(vec_row["embedding"]) if vec_row else []
    return _row_to_object(row, embedding)


def find_dedupe_match(
    conn: sqlite3.Connection,
    class_name: str,
    position_xy: tuple[float, float],
    embedding: list[float],
    position_threshold: float = 0.3,
    cosine_threshold: float = 0.85,
    max_age_s: float = 1800.0,
    now: float | None = None,
) -> int | None:
    """Top-3 nearest neighbors by embedding, then filter to same class_name,
    position within `position_threshold` (normalized image coords), and
    last_seen_ts within `max_age_s`. Returns the closest object_id, or None."""
    now = time.time() if now is None else now
    rows = conn.execute(
        "SELECT object_id, distance FROM object_vecs WHERE embedding MATCH ? AND k = 3 "
        "ORDER BY distance",
        (sqlite_vec.serialize_float32(embedding),),
    ).fetchall()
    best: tuple[float, int] | None = None
    for row in rows:
        # vec0's default metric is L2 over these (L2-normalized) vectors;
        # cosine = 1 - l2_distance**2 / 2 avoids depending on distance_metric= support.
        cosine = 1.0 - (row["distance"] ** 2) / 2.0
        if cosine <= cosine_threshold:
            continue
        obj = conn.execute(
            "SELECT class_name, position_x, position_y, last_seen_ts FROM objects WHERE id = ?",
            (row["object_id"],),
        ).fetchone()
        if obj is None or obj["class_name"] != class_name:
            continue
        if now - obj["last_seen_ts"] > max_age_s:
            continue
        dist = math.dist(position_xy, (obj["position_x"], obj["position_y"]))
        if dist > position_threshold:
            continue
        if best is None or cosine > best[0]:
            best = (cosine, row["object_id"])
    return best[1] if best else None


def upsert_object(conn: sqlite3.Connection, record: ObjectRecord) -> int:
    """Persists whatever's on `record`: inserts if `record.id` is None, else overwrites
    the row by id. Dedupe/merge decisions (running-max confidence, sighting_count++)
    are the caller's (memory/writer.py) job -- this just writes what it's given."""
    px, py = record.position_xy_normalized
    if record.id is None:
        cur = conn.execute(
            "INSERT INTO objects "
            "(class_name, first_seen_ts, last_seen_ts, position_x, position_y, "
            " confidence, sighting_count, image_crop_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.class_name,
                record.first_seen_ts,
                record.last_seen_ts,
                px,
                py,
                record.confidence,
                record.sighting_count,
                record.image_crop_path,
            ),
        )
        object_id = int(cur.lastrowid)  # type: ignore[arg-type]
        conn.execute(
            "INSERT INTO object_vecs (object_id, embedding) VALUES (?, ?)",
            (object_id, sqlite_vec.serialize_float32(record.embedding)),
        )
    else:
        object_id = record.id
        conn.execute(
            "UPDATE objects SET class_name = ?, first_seen_ts = ?, last_seen_ts = ?, "
            "position_x = ?, position_y = ?, confidence = ?, sighting_count = ?, "
            "image_crop_path = ? WHERE id = ?",
            (
                record.class_name,
                record.first_seen_ts,
                record.last_seen_ts,
                px,
                py,
                record.confidence,
                record.sighting_count,
                record.image_crop_path,
                object_id,
            ),
        )
        conn.execute(
            "UPDATE object_vecs SET embedding = ? WHERE object_id = ?",
            (sqlite_vec.serialize_float32(record.embedding), object_id),
        )
    conn.commit()
    return object_id


def insert_sighting(conn: sqlite3.Connection, record: SightingRecord) -> int:
    cur = conn.execute(
        "INSERT INTO sightings "
        "(object_id, ts, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max, frame_id, scene_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.object_id,
            record.ts,
            record.bbox.x_min,
            record.bbox.y_min,
            record.bbox.x_max,
            record.bbox.y_max,
            record.frame_id,
            record.scene_id,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)  # type: ignore[arg-type]


def insert_scene(conn: sqlite3.Connection, record: SceneRecord) -> str:
    conn.execute(
        "INSERT INTO scenes (id, ts, summary_text, num_objects) VALUES (?, ?, ?, ?)",
        (record.id, record.ts, record.summary_text, record.num_objects),
    )
    conn.commit()
    return record.id
