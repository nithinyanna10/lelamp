from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum

import structlog
from pydantic import BaseModel

from lelamp.behavior.expression_player import ExpressionPlayer
from lelamp.behavior.idle_overlay import IdleOverlay
from lelamp.perception.scene_scan import SceneScanRequest
from lelamp.state.config import FSMTimings
from lelamp.telemetry import get_tracer

_tracer = get_tracer(__name__)
_log = structlog.get_logger(__name__)


class LampState(StrEnum):
    SLEEPING = "sleeping"
    IDLE = "idle"
    ENGAGED = "engaged"
    SCANNING = "scanning"
    DISENGAGING = "disengaging"
    SEEKING_1 = "seeking_1"
    SEEKING_2 = "seeking_2"
    SEEKING_3 = "seeking_3"
    # Typed now so step 6/7 (audio, LLM) don't need an enum refactor -- no
    # transitions reach these yet.
    LISTENING = "listening"
    SPEAKING = "speaking"


class StateTransition(BaseModel):
    timestamp: float
    from_state: LampState
    to_state: LampState
    reason: str


# ----------------------------------------------------------------------------
# Events. All carry `timestamp` so handle_event never needs its own clock to
# compute "now" -- only _record_transition's dispatch-latency measurement does.
# ----------------------------------------------------------------------------


class GazeEngaged(BaseModel):
    timestamp: float


class GazeDisengaged(BaseModel):
    timestamp: float


class FacePresent(BaseModel):
    timestamp: float


class FaceLost(BaseModel):
    timestamp: float


class ScanComplete(BaseModel):
    timestamp: float
    num_objects: int = 0
    num_new: int = 0


class Tick(BaseModel):
    timestamp: float


Event = GazeEngaged | GazeDisengaged | FacePresent | FaceLost | ScanComplete | Tick

HISTORY_LEN = 20


class LampFSM:
    """Event-driven: handle_event() dispatches on the event type + current state,
    never polls. Time-based transitions (SEEKING escalation, IDLE->SLEEPING) are
    driven by Tick events -- main.py fires one every `timings.tick_interval_s`,
    tests just construct Tick(timestamp=...) directly.

    Every transition plays an expression via ExpressionPlayer.preempt()/play_chain()
    (never the weaker play()) so it always cuts off whatever's currently running --
    including the IDLE loop, see _go()'s `loop_after` docstring below.
    """

    def __init__(
        self,
        player: ExpressionPlayer,
        scan_request_queue: asyncio.Queue[SceneScanRequest],
        idle_overlay: IdleOverlay | None = None,
        timings: FSMTimings | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.state = LampState.SLEEPING
        self.timings = timings or FSMTimings()
        self._player = player
        self._scan_request_queue = scan_request_queue
        self._idle_overlay = idle_overlay
        self._clock = clock
        self._state_entered_at = 0.0
        self._history: deque[StateTransition] = deque(maxlen=HISTORY_LEN)
        # Holds the fire-and-forget breathe_deep task so it isn't GC'd mid-flight
        # (asyncio only weakly references un-awaited tasks). Never explicitly
        # cancelled from here: every later preempt()/play_chain() call force-cancels
        # whatever ExpressionPlayer is currently running, including this loop --
        # see the module docstring above.
        self._loop_task: asyncio.Task[None] | None = None

    def get_state_history(self) -> list[StateTransition]:
        return list(self._history)

    @property
    def state_entered_at(self) -> float:
        return self._state_entered_at

    async def handle_event(self, event: Event) -> None:
        now = event.timestamp
        if isinstance(event, Tick):
            await self._on_tick(now)
        elif isinstance(event, FacePresent):
            if self.state == LampState.SLEEPING:
                await self._go(
                    LampState.IDLE, "face_present", now, expr="wake", loop_after="breathe_deep"
                )
        elif isinstance(event, FaceLost):
            pass  # IDLE->SLEEPING is time-based (Tick), not edge-triggered on FaceLost
        elif isinstance(event, GazeEngaged):
            if self.state in (LampState.IDLE, LampState.DISENGAGING, LampState.SEEKING_1):
                await self._enter_engaged("gaze_engaged", "notice_user", now)
            elif self.state in (LampState.SEEKING_2, LampState.SEEKING_3):
                await self._enter_engaged("gaze_engaged", "excited", now)
            # ENGAGED/SCANNING: already engaged, no-op.
        elif isinstance(event, GazeDisengaged):
            if self.state == LampState.ENGAGED:
                await self._go(LampState.DISENGAGING, "gaze_disengaged", now, expr="home")
        elif isinstance(event, ScanComplete):
            if self.state == LampState.SCANNING:
                await self._go(LampState.ENGAGED, "scan_complete", now, expr="acknowledge")

    async def reset(self, now: float | None = None) -> None:
        """any -> IDLE, play home. Not wired to any of the 6 event types (none of
        them mean "error") -- callers (e.g. main.py's exception handling) invoke
        this directly."""
        now = now if now is not None else self._clock()
        await self._go(LampState.IDLE, "reset", now, expr="home")

    async def _on_tick(self, now: float) -> None:
        elapsed = now - self._state_entered_at
        t = self.timings
        if self.state == LampState.DISENGAGING and elapsed >= t.disengaging_to_seeking1_s:
            await self._go(LampState.SEEKING_1, "timer", now, chain="seeking_1")
        elif self.state == LampState.SEEKING_1 and elapsed >= t.seeking1_to_seeking2_s:
            await self._go(LampState.SEEKING_2, "timer", now, chain="seeking_2")
        elif self.state == LampState.SEEKING_2 and elapsed >= t.seeking2_to_seeking3_s:
            await self._go(LampState.SEEKING_3, "timer", now, chain="seeking_3")
        elif self.state == LampState.SEEKING_3 and elapsed >= t.seeking3_to_sleeping_s:
            await self._go(LampState.SLEEPING, "timer", now, expr="sleep")
        elif self.state == LampState.IDLE and elapsed >= t.idle_to_sleeping_s:
            await self._go(LampState.SLEEPING, "timer", now, expr="sleep")

    async def _enter_engaged(self, reason: str, entry_expr: str, now: float) -> None:
        await self._go(LampState.ENGAGED, reason, now, expr=entry_expr)
        # "ENGAGED, first entered -> SCANNING": always cascades right after the
        # entry expression finishes, regardless of which state we engaged from.
        await self._go(
            LampState.SCANNING, "first_entered", self._clock(), expr="searching", scan=True
        )

    async def _go(
        self,
        to_state: LampState,
        reason: str,
        now: float,
        *,
        expr: str | None = None,
        chain: str | None = None,
        scan: bool = False,
        loop_after: str | None = None,
    ) -> None:
        self._record_transition(to_state, reason, now)
        if self._idle_overlay is not None:
            self._idle_overlay.set_enabled(to_state != LampState.SLEEPING)
        if scan:
            self._scan_request_queue.put_nowait(SceneScanRequest(timestamp=now))
        if expr is not None:
            await self._player.preempt(expr)
        elif chain is not None:
            await self._player.play_chain(chain)
        if loop_after is not None:
            self._loop_task = asyncio.create_task(self._player.preempt(loop_after))
            await asyncio.sleep(0)  # let it actually start before we return

    def _record_transition(self, to_state: LampState, reason: str, now: float) -> None:
        with _tracer.start_as_current_span("fsm.transition") as span:
            span.set_attribute("from_state", str(self.state))
            span.set_attribute("to_state", str(to_state))
            span.set_attribute("trigger", reason)
            dispatch_latency_ms = (self._clock() - now) * 1000.0
            span.set_attribute("dispatch_latency_ms", dispatch_latency_ms)
            _log.info(
                "fsm_transition",
                from_state=str(self.state),
                to_state=str(to_state),
                reason=reason,
                timestamp=now,
                dispatch_latency_ms=round(dispatch_latency_ms, 2),
            )
            self._history.append(
                StateTransition(
                    timestamp=now, from_state=self.state, to_state=to_state, reason=reason
                )
            )
            self.state = to_state
            self._state_entered_at = now
