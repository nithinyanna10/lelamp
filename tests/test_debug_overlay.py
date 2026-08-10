from __future__ import annotations

import asyncio
from collections import deque

import numpy as np
from conftest import FakeExpressionPlayer

from lelamp.perception.debug_overlay import (
    HudState,
    _draw_panel,
    _LatencyTracker,
    _seeking_countdown_s,
)
from lelamp.perception.face_gaze import DebugFrame
from lelamp.perception.hysteresis import HysteresisGate
from lelamp.state.config import FSMTimings
from lelamp.state.fsm import LampFSM, LampState


def _frame() -> DebugFrame:
    return DebugFrame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        face_present=True,
        gaze_score=0.6,
        num_faces=1,
        inference_ms=5.0,
    )


def _fsm() -> LampFSM:
    scan_queue: asyncio.Queue = asyncio.Queue()
    return LampFSM(FakeExpressionPlayer(), scan_queue)


def test_draw_panel_produces_expected_shape() -> None:
    panel = _draw_panel(
        _frame(),
        HysteresisGate(),
        _fsm(),
        fps=30.0,
        score_history=deque([0.1, 0.5, 0.9], maxlen=100),
        latency=_LatencyTracker(),
        current_expr_name="searching",
        current_expr_started_at=0.0,
        hud=HudState(),
        height=480,
    )
    assert panel.shape == (480, 300, 3)
    assert panel.dtype == np.uint8


def test_draw_panel_handles_no_current_expression_and_no_scan_yet() -> None:
    panel = _draw_panel(
        _frame(),
        HysteresisGate(),
        _fsm(),
        fps=0.0,
        score_history=deque(maxlen=100),
        latency=_LatencyTracker(),
        current_expr_name=None,
        current_expr_started_at=0.0,
        hud=HudState(),
        height=480,
    )
    assert panel.shape == (480, 300, 3)


def test_seeking_countdown_only_shown_in_disengaging_and_seeking_1_2() -> None:
    fsm = _fsm()
    timings = FSMTimings()

    fsm.state = LampState.DISENGAGING
    fsm._state_entered_at = 0.0
    assert _seeking_countdown_s(fsm, now=1.0) == timings.disengaging_to_seeking1_s - 1.0

    fsm.state = LampState.SEEKING_2
    fsm._state_entered_at = 0.0
    assert _seeking_countdown_s(fsm, now=1.0) == timings.seeking2_to_seeking3_s - 1.0

    fsm.state = LampState.SEEKING_3
    assert _seeking_countdown_s(fsm, now=1.0) is None

    fsm.state = LampState.ENGAGED
    assert _seeking_countdown_s(fsm, now=1.0) is None


def test_hud_state_scan_and_memory_counts_are_plain_mutable_fields() -> None:
    hud = HudState()
    assert hud.memory_total == 0
    hud.memory_total = 12
    hud.memory_recent = 5
    hud.last_scan_ts = 100.0
    hud.last_scan_objects = 7
    hud.last_scan_new = 2
    # No validation to fight -- HudState is a plain mutable box, not a pydantic model.
    assert (hud.memory_total, hud.memory_recent) == (12, 5)
