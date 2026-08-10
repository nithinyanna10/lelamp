from __future__ import annotations

import numpy as np
import pytest

from lelamp.memory.db import (
    BBox2D,
    ObjectRecord,
    SightingRecord,
    find_dedupe_match,
    get_object,
    init_db,
    insert_sighting,
    upsert_object,
)


def _unit_embedding(seed: int) -> list[float]:
    v = np.random.RandomState(seed).randn(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


@pytest.fixture
def conn():  # type: ignore[no-untyped-def]
    connection = init_db(":memory:")
    yield connection
    connection.close()


def test_schema_creates_expected_tables(conn) -> None:  # type: ignore[no-untyped-def]
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
    }
    assert {"objects", "sightings", "scenes", "facts", "object_vecs"} <= tables


def test_upsert_object_inserts_and_get_object_round_trips(conn) -> None:  # type: ignore[no-untyped-def]
    record = ObjectRecord(
        class_name="bottle",
        first_seen_ts=1.0,
        last_seen_ts=1.0,
        position_xy_normalized=(0.2, -0.1),
        confidence=0.9,
        embedding=_unit_embedding(1),
    )
    object_id = upsert_object(conn, record)
    assert object_id == 1

    fetched = get_object(conn, object_id)
    assert fetched is not None
    assert fetched.class_name == "bottle"
    assert fetched.position_xy_normalized == pytest.approx((0.2, -0.1))
    assert len(fetched.embedding) == 512


def test_upsert_object_updates_existing_row(conn) -> None:  # type: ignore[no-untyped-def]
    record = ObjectRecord(
        class_name="mug",
        first_seen_ts=1.0,
        last_seen_ts=1.0,
        position_xy_normalized=(0.0, 0.0),
        confidence=0.5,
        embedding=_unit_embedding(2),
    )
    object_id = upsert_object(conn, record)

    fetched = get_object(conn, object_id)
    assert fetched is not None
    fetched.last_seen_ts = 5.0
    fetched.confidence = 0.99
    fetched.sighting_count += 1
    upsert_object(conn, fetched)

    updated = get_object(conn, object_id)
    assert updated is not None
    assert updated.last_seen_ts == 5.0
    assert updated.confidence == 0.99
    assert updated.sighting_count == 2
    assert conn.execute("SELECT COUNT(*) AS n FROM objects").fetchone()["n"] == 1


def test_insert_sighting(conn) -> None:  # type: ignore[no-untyped-def]
    object_id = upsert_object(
        conn,
        ObjectRecord(
            class_name="pen",
            first_seen_ts=1.0,
            last_seen_ts=1.0,
            position_xy_normalized=(0.0, 0.0),
            confidence=0.5,
            embedding=_unit_embedding(3),
        ),
    )
    sighting_id = insert_sighting(
        conn,
        SightingRecord(
            object_id=object_id,
            ts=1.0,
            bbox=BBox2D(x_min=0, y_min=0, x_max=10, y_max=10),
            frame_id=1,
            scene_id="scene-a",
        ),
    )
    assert sighting_id == 1
    row = conn.execute("SELECT object_id FROM sightings WHERE id = ?", (sighting_id,)).fetchone()
    assert row["object_id"] == object_id


def test_find_dedupe_match_returns_top_match_within_thresholds(conn) -> None:  # type: ignore[no-untyped-def]
    embedding = _unit_embedding(42)
    object_id = upsert_object(
        conn,
        ObjectRecord(
            class_name="bottle",
            first_seen_ts=100.0,
            last_seen_ts=100.0,
            position_xy_normalized=(0.1, 0.1),
            confidence=0.8,
            embedding=embedding,
        ),
    )
    # A distinct, dissimilar object so top-3 KNN has something else to rank against.
    upsert_object(
        conn,
        ObjectRecord(
            class_name="laptop",
            first_seen_ts=100.0,
            last_seen_ts=100.0,
            position_xy_normalized=(-0.8, -0.8),
            confidence=0.8,
            embedding=_unit_embedding(7),
        ),
    )

    match = find_dedupe_match(
        conn, "bottle", (0.12, 0.09), embedding, now=110.0
    )
    assert match == object_id


def test_find_dedupe_match_rejects_wrong_class(conn) -> None:  # type: ignore[no-untyped-def]
    embedding = _unit_embedding(42)
    upsert_object(
        conn,
        ObjectRecord(
            class_name="bottle",
            first_seen_ts=100.0,
            last_seen_ts=100.0,
            position_xy_normalized=(0.1, 0.1),
            confidence=0.8,
            embedding=embedding,
        ),
    )
    assert find_dedupe_match(conn, "mug", (0.1, 0.1), embedding, now=110.0) is None


def test_find_dedupe_match_rejects_far_position(conn) -> None:  # type: ignore[no-untyped-def]
    embedding = _unit_embedding(42)
    upsert_object(
        conn,
        ObjectRecord(
            class_name="bottle",
            first_seen_ts=100.0,
            last_seen_ts=100.0,
            position_xy_normalized=(0.1, 0.1),
            confidence=0.8,
            embedding=embedding,
        ),
    )
    assert find_dedupe_match(conn, "bottle", (0.9, 0.9), embedding, now=110.0) is None


def test_find_dedupe_match_rejects_stale_sighting(conn) -> None:  # type: ignore[no-untyped-def]
    embedding = _unit_embedding(42)
    upsert_object(
        conn,
        ObjectRecord(
            class_name="bottle",
            first_seen_ts=100.0,
            last_seen_ts=100.0,
            position_xy_normalized=(0.1, 0.1),
            confidence=0.8,
            embedding=embedding,
        ),
    )
    far_future = 100.0 + 3600.0
    assert find_dedupe_match(conn, "bottle", (0.1, 0.1), embedding, now=far_future) is None
