from __future__ import annotations

import asyncio

import pytest
from conftest import FakeExpressionPlayer

from lelamp.perception.scene_scan import SceneScanRequest
from lelamp.state.config import FSMTimings
from lelamp.state.fsm import (
    FaceLost,
    FacePresent,
    GazeDisengaged,
    GazeEngaged,
    LampFSM,
    LampState,
    ScanComplete,
    Tick,
)

T = FSMTimings()  # placeholder timings from state/config.py, used as-is


def _fsm(player: FakeExpressionPlayer) -> tuple[LampFSM, asyncio.Queue[SceneScanRequest]]:
    scan_queue: asyncio.Queue[SceneScanRequest] = asyncio.Queue()
    fsm = LampFSM(player, scan_queue)
    return fsm, scan_queue


def _drain(queue: asyncio.Queue[SceneScanRequest]) -> list[SceneScanRequest]:
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


async def test_fsm_starts_sleeping(mock_expression_player: FakeExpressionPlayer) -> None:
    fsm, _ = _fsm(mock_expression_player)
    assert fsm.state == LampState.SLEEPING


# ----------------------------------------------------------------------------
# T1: SLEEPING -> IDLE on sustained face presence, plays wake then loops
# breathe_deep (the loop is what makes idle_overlay's wobble actually tick).
# ----------------------------------------------------------------------------
async def test_face_present_wakes_sleeping_to_idle(
    mock_expression_player: FakeExpressionPlayer,
) -> None:
    fsm, _ = _fsm(mock_expression_player)
    await fsm.handle_event(FacePresent(timestamp=1.0))
    assert fsm.state == LampState.IDLE
    assert mock_expression_player.calls == [("preempt", "wake"), ("preempt", "breathe_deep")]
    assert fsm.get_state_history()[-1].reason == "face_present"


async def test_face_present_is_a_noop_outside_sleeping(
    mock_expression_player: FakeExpressionPlayer,
) -> None:
    fsm, _ = _fsm(mock_expression_player)
    await fsm.handle_event(FacePresent(timestamp=1.0))  # SLEEPING -> IDLE
    mock_expression_player.calls.clear()
    await fsm.handle_event(FacePresent(timestamp=2.0))  # already IDLE
    assert fsm.state == LampState.IDLE
    assert mock_expression_player.calls == []


async def test_face_lost_never_transitions(mock_expression_player: FakeExpressionPlayer) -> None:
    fsm, _ = _fsm(mock_expression_player)
    await fsm.handle_event(FacePresent(timestamp=1.0))
    state_before = fsm.state
    await fsm.handle_event(FaceLost(timestamp=2.0))
    assert fsm.state == state_before  # IDLE->SLEEPING is time-based (Tick), not FaceLost


# ----------------------------------------------------------------------------
# T2: IDLE -> ENGAGED on gaze engaged, plays notice_user, then cascades
# straight into SCANNING (plays searching, queues a scan request).
# ----------------------------------------------------------------------------
async def test_gaze_engaged_from_idle_cascades_through_scanning(
    mock_expression_player: FakeExpressionPlayer,
) -> None:
    fsm, scan_queue = _fsm(mock_expression_player)
    await fsm.handle_event(FacePresent(timestamp=1.0))
    mock_expression_player.calls.clear()

    await fsm.handle_event(GazeEngaged(timestamp=2.0))

    assert fsm.state == LampState.SCANNING
    assert mock_expression_player.calls == [("preempt", "notice_user"), ("preempt", "searching")]
    requests = _drain(scan_queue)
    assert len(requests) == 1
    reasons = [t.reason for t in fsm.get_state_history()]
    assert reasons[-2:] == ["gaze_engaged", "first_entered"]


# ----------------------------------------------------------------------------
# T3: SCANNING -> ENGAGED on scan complete, plays acknowledge.
# ----------------------------------------------------------------------------
async def test_scan_complete_returns_to_engaged(
    mock_expression_player: FakeExpressionPlayer,
) -> None:
    fsm, _ = _fsm(mock_expression_player)
    await fsm.handle_event(FacePresent(timestamp=1.0))
    await fsm.handle_event(GazeEngaged(timestamp=2.0))  # -> SCANNING
    mock_expression_player.calls.clear()

    await fsm.handle_event(ScanComplete(timestamp=3.0, num_objects=7, num_new=2))

    assert fsm.state == LampState.ENGAGED
    assert mock_expression_player.calls == [("preempt", "acknowledge")]


async def test_scan_complete_is_a_noop_outside_scanning(
    mock_expression_player: FakeExpressionPlayer,
) -> None:
    fsm, _ = _fsm(mock_expression_player)
    await fsm.handle_event(ScanComplete(timestamp=1.0))
    assert fsm.state == LampState.SLEEPING


# ----------------------------------------------------------------------------
# T4: ENGAGED -> DISENGAGING on gaze disengaged, plays home.
# ----------------------------------------------------------------------------
async def test_gaze_disengaged_from_engaged_goes_to_disengaging(
    mock_expression_player: FakeExpressionPlayer,
) -> None:
    fsm, _ = _fsm(mock_expression_player)
    await fsm.handle_event(FacePresent(timestamp=1.0))
    await fsm.handle_event(GazeEngaged(timestamp=2.0))
    await fsm.handle_event(ScanComplete(timestamp=3.0))  # settle in ENGAGED
    mock_expression_player.calls.clear()

    await fsm.handle_event(GazeDisengaged(timestamp=4.0))

    assert fsm.state == LampState.DISENGAGING
    assert mock_expression_player.calls == [("preempt", "home")]


async def test_gaze_disengaged_is_a_noop_outside_engaged(
    mock_expression_player: FakeExpressionPlayer,
) -> None:
    fsm, _ = _fsm(mock_expression_player)
    await fsm.handle_event(GazeDisengaged(timestamp=1.0))
    assert fsm.state == LampState.SLEEPING


# ----------------------------------------------------------------------------
# T5/T7/T9/T11/T12: timer-based escalation and IDLE/SEEKING_3 -> SLEEPING.
# Timestamps are fully caller-controlled via Tick.timestamp, independent of any
# real clock -- this is the "injectable clock" testability the spec asks for.
# ----------------------------------------------------------------------------
async def test_disengaging_escalates_to_seeking_1_after_timeout(
    mock_expression_player: FakeExpressionPlayer,
) -> None:
    fsm, _ = _fsm(mock_expression_player)
    await fsm.handle_event(FacePresent(timestamp=0.0))
    await fsm.handle_event(GazeEngaged(timestamp=1.0))
    await fsm.handle_event(ScanComplete(timestamp=2.0))
    await fsm.handle_event(GazeDisengaged(timestamp=10.0))  # -> DISENGAGING at t=10
    mock_expression_player.calls.clear()

    await fsm.handle_event(Tick(timestamp=10.0 + T.disengaging_to_seeking1_s - 1.0))
    assert fsm.state == LampState.DISENGAGING  # not yet

    await fsm.handle_event(Tick(timestamp=10.0 + T.disengaging_to_seeking1_s))
    assert fsm.state == LampState.SEEKING_1
    assert mock_expression_player.calls == [("play_chain", "seeking_1")]


async def test_seeking_1_escalates_to_seeking_2_after_timeout(
    mock_expression_player: FakeExpressionPlayer,
) -> None:
    fsm, _ = _fsm(mock_expression_player)
    await fsm.handle_event(FacePresent(timestamp=0.0))
    await fsm.handle_event(GazeEngaged(timestamp=1.0))
    await fsm.handle_event(ScanComplete(timestamp=2.0))
    await fsm.handle_event(GazeDisengaged(timestamp=10.0))
    await fsm.handle_event(Tick(timestamp=10.0 + T.disengaging_to_seeking1_s))  # -> SEEKING_1
    entered = fsm.state_entered_at
    mock_expression_player.calls.clear()

    await fsm.handle_event(Tick(timestamp=entered + T.seeking1_to_seeking2_s))
    assert fsm.state == LampState.SEEKING_2
    assert mock_expression_player.calls == [("play_chain", "seeking_2")]


async def test_seeking_2_escalates_to_seeking_3_after_timeout(
    mock_expression_player: FakeExpressionPlayer,
) -> None:
    fsm, _ = _fsm(mock_expression_player)
    fsm.state = LampState.SEEKING_2
    fsm._state_entered_at = 0.0  # jump straight there rather than replaying T5+T6+T7
    mock_expression_player.calls.clear()

    await fsm.handle_event(Tick(timestamp=T.seeking2_to_seeking3_s))
    assert fsm.state == LampState.SEEKING_3
    assert mock_expression_player.calls == [("play_chain", "seeking_3")]


async def test_seeking_3_times_out_to_sleeping(
    mock_expression_player: FakeExpressionPlayer,
) -> None:
    fsm, _ = _fsm(mock_expression_player)
    fsm.state = LampState.SEEKING_3
    mock_expression_player.calls.clear()

    await fsm.handle_event(Tick(timestamp=T.seeking3_to_sleeping_s))
    assert fsm.state == LampState.SLEEPING
    assert mock_expression_player.calls == [("preempt", "sleep")]


async def test_idle_times_out_to_sleeping(mock_expression_player: FakeExpressionPlayer) -> None:
    fsm, _ = _fsm(mock_expression_player)
    await fsm.handle_event(FacePresent(timestamp=0.0))  # -> IDLE at t=0
    mock_expression_player.calls.clear()

    await fsm.handle_event(Tick(timestamp=T.idle_to_sleeping_s - 1.0))
    assert fsm.state == LampState.IDLE  # not yet

    await fsm.handle_event(Tick(timestamp=T.idle_to_sleeping_s))
    assert fsm.state == LampState.SLEEPING
    assert mock_expression_player.calls == [("preempt", "sleep")]


async def test_tick_is_a_noop_in_engaged(mock_expression_player: FakeExpressionPlayer) -> None:
    fsm, _ = _fsm(mock_expression_player)
    await fsm.handle_event(FacePresent(timestamp=0.0))
    await fsm.handle_event(GazeEngaged(timestamp=1.0))
    await fsm.handle_event(ScanComplete(timestamp=2.0))
    mock_expression_player.calls.clear()

    await fsm.handle_event(Tick(timestamp=1_000_000.0))  # arbitrarily far in the future
    assert fsm.state == LampState.ENGAGED
    assert mock_expression_player.calls == []


# ----------------------------------------------------------------------------
# T6/T8/T10/T13: re-engaging from any of DISENGAGING/SEEKING_1/2/3 goes back to
# ENGAGED (and, per the generic "first entered" rule, cascades to SCANNING again).
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "state,expected_entry_expr",
    [
        (LampState.DISENGAGING, "notice_user"),
        (LampState.SEEKING_1, "notice_user"),
        (LampState.SEEKING_2, "excited"),
        (LampState.SEEKING_3, "excited"),
    ],
)
async def test_gaze_engaged_re_engages_from_any_seeking_state(
    mock_expression_player: FakeExpressionPlayer,
    state: LampState,
    expected_entry_expr: str,
) -> None:
    fsm, scan_queue = _fsm(mock_expression_player)
    fsm.state = state
    mock_expression_player.calls.clear()

    await fsm.handle_event(GazeEngaged(timestamp=100.0))

    assert fsm.state == LampState.SCANNING
    assert mock_expression_player.calls == [
        ("preempt", expected_entry_expr),
        ("preempt", "searching"),
    ]
    assert len(_drain(scan_queue)) == 1


async def test_gaze_engaged_is_a_noop_already_engaged(
    mock_expression_player: FakeExpressionPlayer,
) -> None:
    fsm, scan_queue = _fsm(mock_expression_player)
    fsm.state = LampState.ENGAGED
    mock_expression_player.calls.clear()

    await fsm.handle_event(GazeEngaged(timestamp=1.0))

    assert fsm.state == LampState.ENGAGED
    assert mock_expression_player.calls == []
    assert _drain(scan_queue) == []


# ----------------------------------------------------------------------------
# T14: reset() -> IDLE, plays home, from any state. Not wired to any of the six
# event types -- callers (e.g. main.py's exception handling) invoke it directly.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("state", list(LampState))
async def test_reset_returns_to_idle_from_any_state(
    mock_expression_player: FakeExpressionPlayer, state: LampState
) -> None:
    fsm, _ = _fsm(mock_expression_player)
    fsm.state = state
    mock_expression_player.calls.clear()

    await fsm.reset(now=1.0)

    assert fsm.state == LampState.IDLE
    assert mock_expression_player.calls == [("preempt", "home")]


# ----------------------------------------------------------------------------
# state_history: bounded to 20, ordered oldest -> newest.
# ----------------------------------------------------------------------------
async def test_state_history_is_bounded_and_ordered(
    mock_expression_player: FakeExpressionPlayer,
) -> None:
    fsm, _ = _fsm(mock_expression_player)
    for i in range(30):
        await fsm.reset(now=float(i))

    history = fsm.get_state_history()
    assert len(history) == 20
    assert [t.timestamp for t in history] == sorted(t.timestamp for t in history)
    assert history[-1].timestamp == 29.0


# ----------------------------------------------------------------------------
# Injectable clock: dispatch-latency measurement uses the injected clock, not
# the real one -- mirrors ExpressionPlayer's own `clock` param.
# ----------------------------------------------------------------------------
async def test_custom_clock_is_honored(mock_expression_player: FakeExpressionPlayer) -> None:
    calls: list[float] = []

    def fake_clock() -> float:
        calls.append(1.0)
        return 42.0

    scan_queue: asyncio.Queue[SceneScanRequest] = asyncio.Queue()
    fsm = LampFSM(mock_expression_player, scan_queue, clock=fake_clock)

    await fsm.handle_event(FacePresent(timestamp=1.0))

    assert fsm.state == LampState.IDLE
    assert calls  # the injected clock was actually invoked, not time.monotonic
