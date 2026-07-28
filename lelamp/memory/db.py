from __future__ import annotations

import sqlite3

from pydantic import BaseModel


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
    position_xyz_base_frame: tuple[float, float, float]
    confidence: float
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
    embedding: list[float]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class TEXT NOT NULL,
    first_seen_ts REAL NOT NULL,
    last_seen_ts REAL NOT NULL,
    position_x REAL NOT NULL,
    position_y REAL NOT NULL,
    position_z REAL NOT NULL,
    confidence REAL NOT NULL,
    embedding BLOB NOT NULL,
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
    embedding BLOB NOT NULL
);
"""


def init_db(path: str) -> sqlite3.Connection:
    raise NotImplementedError


def find_dedupe_match(
    conn: sqlite3.Connection,
    class_name: str,
    position_xyz: tuple[float, float, float],
    embedding: list[float],
    iou_threshold: float = 0.5,
    cosine_threshold: float = 0.85,
) -> int | None:
    raise NotImplementedError


def upsert_object(conn: sqlite3.Connection, record: ObjectRecord) -> int:
    raise NotImplementedError


def insert_sighting(conn: sqlite3.Connection, record: SightingRecord) -> int:
    raise NotImplementedError
