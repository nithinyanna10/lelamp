from __future__ import annotations

import sqlite3

from pydantic import BaseModel

from lelamp.memory.db import ObjectRecord
from lelamp.memory.embeddings import ClipEmbedder


class TimeRange(BaseModel):
    start_ts: float | None = None
    end_ts: float | None = None


class QueryMemoryInput(BaseModel):
    description: str
    time_range: TimeRange | None = None
    spatial_hint: str | None = None


class QueryMemoryResult(BaseModel):
    matches: list[ObjectRecord]


class DescribeSceneResult(BaseModel):
    summary_text: str
    object_classes: list[str]


class PointAtInput(BaseModel):
    object_id: int


class RememberInput(BaseModel):
    fact: str


async def query_memory(
    conn: sqlite3.Connection,
    embedder: ClipEmbedder,
    params: QueryMemoryInput,
) -> QueryMemoryResult:
    raise NotImplementedError


async def describe_current_scene(conn: sqlite3.Connection, scene_id: str) -> DescribeSceneResult:
    raise NotImplementedError


async def point_at_tool(conn: sqlite3.Connection, params: PointAtInput) -> ObjectRecord:
    raise NotImplementedError


async def remember(conn: sqlite3.Connection, params: RememberInput) -> None:
    raise NotImplementedError
