from __future__ import annotations

import sqlite3
import time

import sqlite_vec
from pydantic import BaseModel

from lelamp.memory.db import ObjectRecord, get_object
from lelamp.memory.embeddings import ClipEmbedder
from lelamp.telemetry import get_tracer

_tracer = get_tracer(__name__)


class TimeRange(BaseModel):
    start_ts: float | None = None
    end_ts: float | None = None


class QueryMemoryInput(BaseModel):
    description: str
    time_range: TimeRange | None = None
    spatial_hint: str | None = None
    top_k: int = 5


class ObjectMatch(BaseModel):
    object: ObjectRecord
    similarity: float


class QueryMemoryResult(BaseModel):
    matches: list[ObjectMatch]


class DescribeSceneResult(BaseModel):
    summary_text: str
    object_classes: list[str]


class PointAtInput(BaseModel):
    object_id: int


class RememberInput(BaseModel):
    fact: str


def _in_time_range(ts: float, time_range: TimeRange | None) -> bool:
    if time_range is None:
        return True
    if time_range.start_ts is not None and ts < time_range.start_ts:
        return False
    return not (time_range.end_ts is not None and ts > time_range.end_ts)


def _matches_spatial_hint(position: tuple[float, float], hint: str | None) -> bool:
    # ponytail: naive quadrant heuristic on normalized image position. Good enough for
    # "on the left" / "up top" type hints; upgrade to real 3D once step 7 adds it.
    if hint is None:
        return True
    x, y = position
    hint = hint.lower()
    if "left" in hint:
        return x < -0.15
    if "right" in hint:
        return x > 0.15
    if "top" in hint or "up" in hint:
        return y < -0.15
    if "bottom" in hint or "down" in hint:
        return y > 0.15
    return True


def _humanize_age(seconds_ago: float) -> str:
    if seconds_ago < 60:
        return f"{int(seconds_ago)} seconds ago"
    if seconds_ago < 3600:
        return f"{int(seconds_ago // 60)} minutes ago"
    return f"{int(seconds_ago // 3600)} hours ago"


async def query_memory(
    conn: sqlite3.Connection,
    embedder: ClipEmbedder,
    params: QueryMemoryInput,
) -> QueryMemoryResult:
    """Semantic search: CLIP-embeds the description, KNN over object_vecs, then
    filters by time_range / spatial_hint. Latency budget: < 200ms."""
    with _tracer.start_as_current_span("memory.query_memory") as span:
        span.set_attribute("description", params.description)
        query_embedding = embedder.embed_text(params.description)
        rows = conn.execute(
            "SELECT object_id, distance FROM object_vecs WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (sqlite_vec.serialize_float32(query_embedding), max(params.top_k * 3, params.top_k)),
        ).fetchall()

        matches: list[ObjectMatch] = []
        for row in rows:
            obj = get_object(conn, row["object_id"])
            if obj is None:
                continue
            if not _in_time_range(obj.last_seen_ts, params.time_range):
                continue
            if not _matches_spatial_hint(obj.position_xy_normalized, params.spatial_hint):
                continue
            similarity = 1.0 - (row["distance"] ** 2) / 2.0
            matches.append(ObjectMatch(object=obj, similarity=similarity))
            if len(matches) >= params.top_k:
                break

        span.set_attribute("num_matches", len(matches))
        return QueryMemoryResult(matches=matches)


async def query_by_class(conn: sqlite3.Connection, class_name: str) -> list[ObjectRecord]:
    """Structured lookup, no embedding involved. Latency budget: < 50ms."""
    with _tracer.start_as_current_span("memory.query_by_class") as span:
        span.set_attribute("class_name", class_name)
        rows = conn.execute(
            "SELECT id FROM objects WHERE class_name = ? ORDER BY last_seen_ts DESC",
            (class_name,),
        ).fetchall()
        records = [get_object(conn, row["id"]) for row in rows]
        return [r for r in records if r is not None]


async def query_recent(
    conn: sqlite3.Connection, time_window_seconds: float, limit: int = 20
) -> list[ObjectRecord]:
    with _tracer.start_as_current_span("memory.query_recent") as span:
        cutoff = time.time() - time_window_seconds
        span.set_attribute("cutoff_ts", cutoff)
        rows = conn.execute(
            "SELECT id FROM objects WHERE last_seen_ts >= ? ORDER BY last_seen_ts DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
        records = [get_object(conn, row["id"]) for row in rows]
        return [r for r in records if r is not None]


async def describe_current_memory(conn: sqlite3.Connection) -> str:
    with _tracer.start_as_current_span("memory.describe_current_memory"):
        total = conn.execute("SELECT COUNT(*) AS n FROM objects").fetchone()["n"]
        if total == 0:
            return "You haven't seen any objects yet."
        rows = conn.execute(
            "SELECT class_name, last_seen_ts FROM objects ORDER BY last_seen_ts DESC LIMIT 3"
        ).fetchall()
        now = time.time()
        recent = ", ".join(
            f"{row['class_name']} ({_humanize_age(now - row['last_seen_ts'])})" for row in rows
        )
        return f"You have seen {total} objects. Most recent: {recent}."


async def describe_current_scene(conn: sqlite3.Connection, scene_id: str) -> DescribeSceneResult:
    with _tracer.start_as_current_span("memory.describe_current_scene") as span:
        span.set_attribute("scene_id", scene_id)
        scene = conn.execute(
            "SELECT summary_text FROM scenes WHERE id = ?", (scene_id,)
        ).fetchone()
        summary_text = scene["summary_text"] if scene else "no scene recorded"
        rows = conn.execute(
            "SELECT DISTINCT o.class_name FROM sightings s "
            "JOIN objects o ON o.id = s.object_id WHERE s.scene_id = ?",
            (scene_id,),
        ).fetchall()
        return DescribeSceneResult(
            summary_text=summary_text, object_classes=[row["class_name"] for row in rows]
        )


async def point_at_tool(conn: sqlite3.Connection, params: PointAtInput) -> ObjectRecord:
    """Returns the object's stored (2D, normalized) position; step 7 maps that to a
    joint target -- no 3D reconstruction here."""
    with _tracer.start_as_current_span("memory.point_at") as span:
        span.set_attribute("object_id", params.object_id)
        obj = get_object(conn, params.object_id)
        if obj is None:
            raise KeyError(f"no object with id {params.object_id}")
        return obj


async def remember(conn: sqlite3.Connection, params: RememberInput) -> None:
    with _tracer.start_as_current_span("memory.remember") as span:
        span.set_attribute("fact", params.fact)
        conn.execute("INSERT INTO facts (ts, text) VALUES (?, ?)", (time.time(), params.fact))
        conn.commit()
