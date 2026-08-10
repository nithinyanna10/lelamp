from __future__ import annotations

import time

import numpy as np
import pytest

from lelamp.memory import query as q
from lelamp.memory.db import init_db
from lelamp.memory.writer import write_scan
from lelamp.perception.scene_scan import BBox, Detection, Detections


def _unit_embedding(seed: int) -> list[float]:
    v = np.random.RandomState(seed).randn(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


class FakeEmbedder:
    """Returns whatever embedding _unit_embedding(1) is -- makes 'bottle' the
    best match by construction, since that's what seeds object 1 below."""

    def embed_text(self, text: str) -> list[float]:
        return _unit_embedding(1)


@pytest.fixture
def conn(tmp_path):  # type: ignore[no-untyped-def]
    connection = init_db(":memory:")
    now = time.time()
    detections = Detections(
        frame_id=1,
        timestamp=now,
        scene_id="scene-a",
        detections=[
            Detection(
                class_name="bottle",
                confidence=0.9,
                bbox=BBox(x_min=0, y_min=0, x_max=10, y_max=10),
                position_xy_normalized=(0.1, 0.1),
                image_crop=np.zeros((10, 10, 3), dtype=np.uint8),
                embedding=_unit_embedding(1),
            ),
            Detection(
                class_name="laptop",
                confidence=0.8,
                bbox=BBox(x_min=0, y_min=0, x_max=10, y_max=10),
                position_xy_normalized=(-0.6, 0.0),
                image_crop=np.zeros((10, 10, 3), dtype=np.uint8),
                embedding=_unit_embedding(2),
            ),
        ],
    )
    write_scan(connection, detections, tmp_path)
    yield connection
    connection.close()


async def test_query_memory_ranks_by_similarity(conn) -> None:  # type: ignore[no-untyped-def]
    result = await q.query_memory(conn, FakeEmbedder(), q.QueryMemoryInput(description="a bottle"))
    assert result.matches[0].object.class_name == "bottle"
    assert result.matches[0].similarity == pytest.approx(1.0, abs=1e-4)


async def test_query_memory_spatial_hint_filters(conn) -> None:  # type: ignore[no-untyped-def]
    result = await q.query_memory(
        conn, FakeEmbedder(), q.QueryMemoryInput(description="x", spatial_hint="left")
    )
    assert [m.object.class_name for m in result.matches] == ["laptop"]


async def test_query_by_class(conn) -> None:  # type: ignore[no-untyped-def]
    matches = await q.query_by_class(conn, "laptop")
    assert len(matches) == 1
    assert matches[0].class_name == "laptop"
    assert await q.query_by_class(conn, "phone") == []


async def test_query_recent(conn) -> None:  # type: ignore[no-untyped-def]
    matches = await q.query_recent(conn, time_window_seconds=3600)
    assert {m.class_name for m in matches} == {"bottle", "laptop"}
    assert await q.query_recent(conn, time_window_seconds=0) == []


async def test_describe_current_memory(conn) -> None:  # type: ignore[no-untyped-def]
    text = await q.describe_current_memory(conn)
    assert "2 objects" in text
    assert "bottle" in text and "laptop" in text


async def test_describe_current_scene(conn) -> None:  # type: ignore[no-untyped-def]
    result = await q.describe_current_scene(conn, "scene-a")
    assert set(result.object_classes) == {"bottle", "laptop"}


async def test_point_at_tool(conn) -> None:  # type: ignore[no-untyped-def]
    bottle_id = conn.execute(
        "SELECT id FROM objects WHERE class_name = 'bottle'"
    ).fetchone()["id"]
    record = await q.point_at_tool(conn, q.PointAtInput(object_id=bottle_id))
    assert record.class_name == "bottle"

    with pytest.raises(KeyError):
        await q.point_at_tool(conn, q.PointAtInput(object_id=99999))


async def test_remember_persists_fact(conn) -> None:  # type: ignore[no-untyped-def]
    await q.remember(conn, q.RememberInput(fact="the mug is a gift"))
    row = conn.execute("SELECT text FROM facts").fetchone()
    assert row["text"] == "the mug is a gift"
